from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from hirelens._compat import StrEnum


class Tier(StrEnum):
    STRONG_NO = "strong_no"
    NO = "no"
    MAYBE = "maybe"
    YES = "yes"
    STRONG_YES = "strong_yes"


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

TIER_DEFINITIONS: dict[Tier, str] = {
    Tier.STRONG_YES: "Clear interview. Meets the bar with evidence to spare.",
    Tier.YES: "Would interview. Meets the requirements.",
    Tier.MAYBE: "Borderline. Would interview only if the pipeline were thin.",
    Tier.NO: "Would not interview. Misses requirements that matter.",
    Tier.STRONG_NO: "Clear reject. Not close to the role.",
}


class Label(BaseModel):
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
    labels: list[Label] = Field(default_factory=list)

    def get(self, job_id: str, candidate_id: str) -> Label | None:
        return next((label for label in self.labels if label.key == (job_id, candidate_id)), None)

    def for_job(self, job_id: str) -> list[Label]:
        return [label for label in self.labels if label.job_id == job_id]

    def upsert(self, label: Label) -> None:
        self.labels = [existing for existing in self.labels if existing.key != label.key]
        self.labels.append(label)
        self.labels.sort(key=lambda item: item.key)

    def missing(self, job_ids: list[str], candidate_ids: list[str]) -> list[tuple[str, str]]:
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
    model_config = ConfigDict(frozen=True)

    code: str
    message: str


def check_label_quality(labels: LabelSet, job_ids: list[str]) -> list[LabelQualityWarning]:
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
