"""Human ground truth: the part of the harness a model cannot produce.

Labels are stored separately from the golden set so that regenerating or editing
a profile does not silently invalidate the judgement someone made about it. Each
label records who made it, when, and why, because a rating with no rationale
cannot be reviewed, disputed, or learned from six months later.

**The scale is a five-point tier, not a 0-100 score.** Humans are no better at
producing calibrated numbers than models are: asked for a score out of a hundred
you get clustering on multiples of five and drift across a long labelling session.
Asked to choose between "would interview" and "probably not", with each level
defined, people are consistent. Spearman and Kendall both work on tiers, and the
tie-handling in :mod:`hirelens.evals.metrics` exists precisely because tiers
produce ties.

**The label is a decision, not a rating.** "Would you interview this person for
this role" is a question a screener answers every day, so the answers are grounded
in something real. "How good is this candidate out of ten" is not.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from hirelens._compat import StrEnum


class Tier(StrEnum):
    """Five-point screening decision, worst to best."""

    STRONG_NO = "strong_no"
    NO = "no"
    MAYBE = "maybe"
    YES = "yes"
    STRONG_YES = "strong_yes"


#: Ordinal values. Evenly spaced because the tiers are ranks, not measurements,
#: and pretending to know that the gap between "yes" and "strong yes" is twice
#: the gap between "no" and "maybe" would be inventing precision.
TIER_VALUES: dict[Tier, float] = {
    Tier.STRONG_NO: 0.0,
    Tier.NO: 1.0,
    Tier.MAYBE: 2.0,
    Tier.YES: 3.0,
    Tier.STRONG_YES: 4.0,
}

TIER_ORDER: tuple[Tier, ...] = (
    Tier.STRONG_NO,
    Tier.NO,
    Tier.MAYBE,
    Tier.YES,
    Tier.STRONG_YES,
)

#: Shown during labelling. The definitions are the calibration: without them,
#: two people, or the same person on two days, mean different things by "maybe".
TIER_DEFINITIONS: dict[Tier, str] = {
    Tier.STRONG_YES: "Clear interview. Meets the bar with evidence to spare.",
    Tier.YES: "Would interview. Meets the requirements.",
    Tier.MAYBE: "Borderline. Would interview only if the pipeline were thin.",
    Tier.NO: "Would not interview. Misses requirements that matter.",
    Tier.STRONG_NO: "Clear reject. Not close to the role.",
}


class Label(BaseModel):
    """One human judgement of one candidate against one job."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    candidate_id: str
    tier: Tier
    rationale: str = Field(
        default="",
        description="Why. Short is fine, but empty makes the label unreviewable.",
    )
    labeller: str = "unknown"
    labelled_at: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.job_id, self.candidate_id)

    @property
    def value(self) -> float:
        return TIER_VALUES[self.tier]

    @classmethod
    def create(
        cls,
        job_id: str,
        candidate_id: str,
        tier: Tier,
        *,
        rationale: str = "",
        labeller: str = "unknown",
    ) -> Label:
        return cls(
            job_id=job_id,
            candidate_id=candidate_id,
            tier=tier,
            rationale=rationale,
            labeller=labeller,
            labelled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )


class LabelSet(BaseModel):
    """Every human label, keyed by (job, candidate)."""

    labels: list[Label] = Field(default_factory=list)

    def get(self, job_id: str, candidate_id: str) -> Label | None:
        return next((label for label in self.labels if label.key == (job_id, candidate_id)), None)

    def for_job(self, job_id: str) -> list[Label]:
        return [label for label in self.labels if label.job_id == job_id]

    def upsert(self, label: Label) -> None:
        """Replace any existing label for the same pair."""
        self.labels = [existing for existing in self.labels if existing.key != label.key]
        self.labels.append(label)
        self.labels.sort(key=lambda item: item.key)

    def missing(self, job_ids: list[str], candidate_ids: list[str]) -> list[tuple[str, str]]:
        """Pairs still needing a human decision."""
        have = {label.key for label in self.labels}
        return [
            (job_id, candidate_id)
            for job_id in job_ids
            for candidate_id in candidate_ids
            if (job_id, candidate_id) not in have
        ]

    @property
    def coverage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for label in self.labels:
            counts[label.job_id] = counts.get(label.job_id, 0) + 1
        return counts

    def tier_distribution(self, job_id: str | None = None) -> dict[str, int]:
        """How many labels landed in each tier.

        Worth checking before trusting any correlation: a set where everything is
        "yes" carries no ranking information, and a coefficient computed over it
        is meaningless however large it looks.
        """
        selected = self.for_job(job_id) if job_id else self.labels
        counts = {str(tier): 0 for tier in TIER_ORDER}
        for label in selected:
            counts[str(label.tier)] += 1
        return counts

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LabelSet:
        if not path.exists():
            return cls()
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


class LabelQualityWarning(BaseModel):
    """A reason to distrust the metrics computed from these labels."""

    model_config = ConfigDict(frozen=True)

    code: str
    message: str


def check_label_quality(labels: LabelSet, job_ids: list[str]) -> list[LabelQualityWarning]:
    """Sanity checks on the ground truth before it is used to judge anything.

    Reported at the top of the eval output rather than buried. A metric computed
    from degenerate labels is worse than no metric, because it looks like
    evidence.
    """
    warnings: list[LabelQualityWarning] = []

    for job_id in job_ids:
        job_labels = labels.for_job(job_id)

        if len(job_labels) < 5:
            warnings.append(
                LabelQualityWarning(
                    code="too_few_labels",
                    message=(
                        f"Job '{job_id}' has only {len(job_labels)} labels. Correlation on "
                        f"fewer than about 10 pairs is not interpretable."
                    ),
                )
            )
            continue

        distinct = {label.tier for label in job_labels}
        if len(distinct) < 2:
            warnings.append(
                LabelQualityWarning(
                    code="no_variance",
                    message=(
                        f"Job '{job_id}' has every candidate in the same tier. There is no "
                        f"ranking to reproduce, so correlation is undefined."
                    ),
                )
            )
        elif len(distinct) == 2:
            warnings.append(
                LabelQualityWarning(
                    code="low_variance",
                    message=(
                        f"Job '{job_id}' uses only {len(distinct)} of 5 tiers. Correlation "
                        f"will be dominated by a single split point."
                    ),
                )
            )

        without_rationale = sum(1 for label in job_labels if not label.rationale.strip())
        if without_rationale > len(job_labels) / 2:
            warnings.append(
                LabelQualityWarning(
                    code="missing_rationales",
                    message=(
                        f"Job '{job_id}' has {without_rationale} labels with no rationale. "
                        f"These cannot be reviewed or disputed later."
                    ),
                )
            )

    return warnings
