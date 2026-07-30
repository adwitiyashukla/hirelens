"""The counterfactual fairness audit.

For each candidate profile, score it once per demographic variant, changing
nothing else, and measure what moved.

Three design decisions make this worth more than the usual "we care about bias"
paragraph.

**The null control sets the floor.** Two runs of an unmodified resume give the
system's own noise. A three-point swing when the name changes means nothing if the
score swings three points when nothing changes. Every axis is reported as drift
*relative to that floor*, and the gate only fires on drift that exceeds it. This
one control is the difference between a fairness measurement and fairness theatre.

**Blind and sighted are both measured.** With blind mode on, names and institutions
are masked before the model sees them, so drift should be near zero and any drift
that remains is a *redaction leak*: the mask failed somewhere. With blind mode off,
the same experiment measures the underlying model bias that blind mode exists to
suppress. Reporting both quantifies how much the mitigation is actually buying,
which is a number almost nobody publishes about their own system.

**Rank flips are reported alongside point drift.** A recruiter never sees the raw
score; they see who is at the top of the list. "Swapping the name moved this
candidate from 3rd to 7th" is the outcome that matters, and a small mean drift can
still reorder a shortlist.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from hirelens.assess.pipeline import ScreeningPipeline
from hirelens.audit.perturbations import DEFAULT_AXES, Axis, Variant, build_plan
from hirelens.config import Settings, get_settings
from hirelens.evals.golden import build_golden_set
from hirelens.evals.profiles import CandidateProfile, GoldenSet, JobSpec
from hirelens.evals.runner import _as_document
from hirelens.llm.client import LLMClient
from hirelens.retrieve.embeddings import Embedder

logger = logging.getLogger(__name__)


class Observation(BaseModel):
    """One scored (profile, variant) pair."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    job_id: str
    axis: str
    variant_label: str
    group: str
    blind: bool
    score: float
    rank: int = 0
    meets_must_haves: bool = True


class AxisResult(BaseModel):
    """Drift measured along one demographic axis, in one blind mode."""

    model_config = ConfigDict(frozen=True)

    axis: str
    blind: bool
    observations: int
    max_drift: float = Field(description="Largest score gap within a single profile, in points")
    mean_drift: float = Field(description="Mean within-profile score range across profiles")
    group_means: dict[str, float] = Field(default_factory=dict)
    group_gap: float = Field(
        default=0.0, description="Best-scoring group mean minus worst-scoring group mean"
    )
    rank_flips: int = Field(
        default=0, description="Profiles whose shortlist position changed under any variant"
    )
    must_have_flips: int = Field(
        default=0, description="Profiles where a variant changed must-have compliance"
    )

    @property
    def favoured_group(self) -> str:
        return max(self.group_means, key=lambda g: self.group_means[g]) if self.group_means else ""

    @property
    def disfavoured_group(self) -> str:
        return min(self.group_means, key=lambda g: self.group_means[g]) if self.group_means else ""


class AuditReport(BaseModel):
    """Everything one audit run produced."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    self_consistency_k: int
    profiles_tested: int
    variants_tested: int
    job_id: str

    noise_floor: float = Field(
        default=0.0,
        description="Mean drift from the null control: the system's own run-to-run movement",
    )
    noise_floor_max: float = 0.0

    axes: list[AxisResult] = Field(default_factory=list)
    threshold: float = 2.0
    elapsed_s: float = 0.0
    api_calls: int = 0
    warnings: list[str] = Field(default_factory=list)

    # -- interpretation ------------------------------------------------------

    def for_axis(self, axis: str, *, blind: bool) -> AxisResult | None:
        return next((a for a in self.axes if a.axis == axis and a.blind is blind), None)

    @property
    def blind_results(self) -> list[AxisResult]:
        return [a for a in self.axes if a.blind and a.axis != str(Axis.NULL)]

    @property
    def sighted_results(self) -> list[AxisResult]:
        return [a for a in self.axes if not a.blind and a.axis != str(Axis.NULL)]

    def excess_drift(self, result: AxisResult) -> float:
        """Drift above the system's own noise. This is the number that matters."""
        return max(0.0, result.max_drift - self.noise_floor_max)

    @property
    def worst_blind_axis(self) -> AxisResult | None:
        """The axis leaking most through the redaction, if any."""
        results = self.blind_results
        return max(results, key=lambda a: a.max_drift) if results else None

    @property
    def worst_sighted_axis(self) -> AxisResult | None:
        results = self.sighted_results
        return max(results, key=lambda a: a.max_drift) if results else None

    @property
    def blind_mode_benefit(self) -> float:
        """How many points of drift blind mode removes.

        The headline result of the whole audit: it quantifies the mitigation
        rather than asserting it.
        """
        blind = self.worst_blind_axis
        sighted = self.worst_sighted_axis
        if blind is None or sighted is None:
            return 0.0
        return round(sighted.max_drift - blind.max_drift, 2)

    @property
    def passes(self) -> bool:
        """True when no axis drifts beyond the threshold, above the noise floor.

        Only blind-mode results gate, because blind mode is the shipping
        configuration. The sighted numbers are diagnostic: they measure the
        problem, not the product.
        """
        return all(self.excess_drift(result) <= self.threshold for result in self.blind_results)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> AuditReport:
        import json

        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


@dataclass
class FairnessAudit:
    """Runs the counterfactual experiment."""

    golden: GoldenSet = field(default_factory=build_golden_set)
    settings: Settings = field(default_factory=get_settings)
    embedder: Embedder | None = None
    client: LLMClient | None = None

    async def run(
        self,
        *,
        job_id: str = "backend",
        profile_ids: list[str] | None = None,
        axes: tuple[Axis, ...] = DEFAULT_AXES,
        variants_per_axis: int | None = 4,
        both_modes: bool = True,
        threshold: float | None = None,
        top_k: int = 4,
        k: int | None = None,
    ) -> AuditReport:
        """Score every (profile, variant) combination and measure the drift.

        ``k`` overrides self-consistency sampling for this run only. Lowering it is
        the main cost lever, and the trade-off is explicit: fewer samples means a
        noisier score, which raises the null control's noise floor and makes the
        audit *less* able to detect small drift. It never makes the audit falsely
        claim bias, because everything is measured against that same floor.
        """
        started = time.perf_counter()
        warnings: list[str] = []

        job = self.golden.job(job_id)
        if job is None:
            raise ValueError(
                f"Unknown job '{job_id}'. Known: {[j.job_id for j in self.golden.jobs]}"
            )

        profiles = self._select_profiles(profile_ids)
        plan = build_plan(axes, variants_per_axis=variants_per_axis)

        # The cache must be off for the null control to mean anything: with it on,
        # two runs of an identical resume produce an identical prompt, hit the same
        # entry, and return the same score, so the measured noise floor would be
        # zero by construction rather than by measurement.
        audit_settings = self.settings.model_copy(
            update={
                "cache_enabled": False,
                "self_consistency_k": k or self.settings.self_consistency_k,
            }
        )
        client = self.client or LLMClient(settings=audit_settings)
        pipeline = ScreeningPipeline(client, settings=audit_settings, embedder=self.embedder)

        # Force the cache off on the client we were actually handed, not just in
        # our settings copy. An injected client carries its own cache built from
        # the caller's settings, so settings alone would leave it on.
        #
        # This is not a tidiness point. With caching on, two runs of an identical
        # resume produce an identical prompt, hit the same entry, and return the
        # same score, so the null control would report a noise floor of zero by
        # construction. Every drift number would then be compared against a floor
        # that was never measured, and the whole audit becomes theatre.
        cache_was_enabled = client.cache.enabled
        client.cache.enabled = False

        modes = [True, False] if both_modes else [self.settings.blind_mode]
        observations: list[Observation] = []

        try:
            # The rubric is compiled once and reused for every variant. Compiling
            # per variant would let rubric wording drift between conditions and
            # contaminate the comparison the experiment exists to make.
            rubric = await pipeline.compile_rubric(job.text)

            for blind in modes:
                for variant in plan:
                    for profile in profiles:
                        perturbed = profile.with_demographics(variant.apply(profile.demographics))
                        document = _as_document(
                            f"{profile.candidate_id}-{variant.label}", perturbed.render()
                        )
                        outcome = await pipeline.screen(
                            document, rubric, top_k=top_k, with_questions=False, blind=blind
                        )
                        observations.append(
                            Observation(
                                candidate_id=profile.candidate_id,
                                job_id=job_id,
                                axis=str(variant.axis),
                                variant_label=variant.label,
                                group=variant.group,
                                blind=blind,
                                score=outcome.assessment.score,
                                meets_must_haves=outcome.assessment.meets_all_must_haves,
                            )
                        )
        finally:
            client.cache.enabled = cache_was_enabled
            if self.client is None:
                await client.aclose()

        observations = _assign_ranks(observations)
        report = self._build_report(
            k=audit_settings.self_consistency_k,
            observations=observations,
            profiles=profiles,
            plan=plan,
            job=job,
            client=client,
            threshold=threshold if threshold is not None else self.settings.max_demographic_drift,
            warnings=warnings,
            elapsed=time.perf_counter() - started,
        )
        return report

    # -- helpers -------------------------------------------------------------

    def _select_profiles(self, profile_ids: list[str] | None) -> list[CandidateProfile]:
        """Which profiles to test.

        The default picks a spread across quality tiers rather than the first N.
        Bias is not uniform across the score range: a model may treat a borderline
        candidate very differently from an obviously strong one, and testing only
        strong profiles would miss exactly the cases where a swap changes a
        decision.
        """
        if profile_ids:
            chosen = [p for p in self.golden.profiles if p.candidate_id in profile_ids]
            if not chosen:
                raise ValueError(f"No profiles matching {profile_ids}")
            return chosen

        by_quality: dict[str, CandidateProfile] = {}
        for profile in self.golden.profiles:
            by_quality.setdefault(str(profile.quality), profile)
        return list(by_quality.values())

    def _build_report(
        self,
        *,
        k: int,
        observations: list[Observation],
        profiles: list[CandidateProfile],
        plan: list[Variant],
        job: JobSpec,
        client: LLMClient,
        threshold: float,
        warnings: list[str],
        elapsed: float,
    ) -> AuditReport:
        control_blind = _axis_result(observations, str(Axis.NULL), blind=True)
        control_sighted = _axis_result(observations, str(Axis.NULL), blind=False)

        controls = [c for c in (control_blind, control_sighted) if c is not None]
        noise_mean = statistics.fmean([c.mean_drift for c in controls]) if controls else 0.0
        noise_max = max((c.max_drift for c in controls), default=0.0)

        if noise_max > threshold:
            warnings.append(
                f"The system's own run-to-run noise ({noise_max:.1f} pts) already exceeds the "
                f"drift threshold ({threshold:.1f} pts). No demographic conclusion can be drawn "
                f"until self-consistency improves; raise k or lower the judge temperature."
            )

        axes_present = sorted({o.axis for o in observations})
        results = [
            result
            for axis in axes_present
            for blind in (True, False)
            if (result := _axis_result(observations, axis, blind=blind)) is not None
        ]

        usage = client.usage_summary()

        return AuditReport(
            provider=str(self.settings.llm_provider),
            model=self.settings.active_model,
            self_consistency_k=k,
            profiles_tested=len(profiles),
            variants_tested=len(plan),
            job_id=job.job_id,
            noise_floor=round(noise_mean, 2),
            noise_floor_max=round(noise_max, 2),
            axes=results,
            threshold=threshold,
            elapsed_s=round(elapsed, 1),
            api_calls=int(usage.get("api_calls", 0)),
            warnings=warnings,
        )


# ---------------------------------------------------------------------------


def _assign_ranks(observations: list[Observation]) -> list[Observation]:
    """Rank candidates within each (variant, blind) condition.

    Ranking within a condition is what makes rank flips meaningful: it answers
    "if every candidate had this demographic profile, where would each land",
    which isolates the effect of the attribute from the effect of the candidate.
    """
    ranked: list[Observation] = []
    conditions = {(o.variant_label, o.blind) for o in observations}

    for variant_label, blind in conditions:
        subset = [o for o in observations if o.variant_label == variant_label and o.blind is blind]
        ordered = sorted(subset, key=lambda o: -o.score)
        for position, observation in enumerate(ordered, start=1):
            ranked.append(observation.model_copy(update={"rank": position}))

    return ranked


def _axis_result(observations: list[Observation], axis: str, *, blind: bool) -> AxisResult | None:
    """Aggregate drift for one axis in one blind mode."""
    subset = [o for o in observations if o.axis == axis and o.blind is blind]
    if not subset:
        return None

    # Within-profile drift: the same candidate under different demographics.
    # Comparing across profiles would measure candidate quality, not bias.
    drifts: list[float] = []
    rank_flips = 0
    must_have_flips = 0

    for candidate_id in {o.candidate_id for o in subset}:
        rows = [o for o in subset if o.candidate_id == candidate_id]
        if len(rows) < 2:
            continue
        scores = [o.score for o in rows]
        drifts.append(max(scores) - min(scores))
        if len({o.rank for o in rows}) > 1:
            rank_flips += 1
        if len({o.meets_must_haves for o in rows}) > 1:
            must_have_flips += 1

    group_means = {
        group: round(statistics.fmean([o.score for o in subset if o.group == group]), 2)
        for group in sorted({o.group for o in subset})
    }
    values = list(group_means.values())

    return AxisResult(
        axis=axis,
        blind=blind,
        observations=len(subset),
        max_drift=round(max(drifts), 2) if drifts else 0.0,
        mean_drift=round(statistics.fmean(drifts), 2) if drifts else 0.0,
        group_means=group_means,
        group_gap=round(max(values) - min(values), 2) if len(values) > 1 else 0.0,
        rank_flips=rank_flips,
        must_have_flips=must_have_flips,
    )
