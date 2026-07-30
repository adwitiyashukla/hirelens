"""The evaluation harness: run the pipeline over the golden set and report.

What ``make eval`` produces, and why each part is there:

* **Agreement with human ranking**, per job and pooled, as Spearman and Kendall
  with bootstrap intervals. The headline question: does this order candidates the
  way a person would?
* **Baseline comparison.** The same metrics for random, keyword-overlap and BM25
  rankers. Without these the headline number is unanchored.
* **Self-consistency.** Mean sample agreement and mean confidence-band width
  across every requirement judged. Answers "would we get the same answer
  tomorrow", which nobody normally measures.
* **Citation validity and grounding.** The properties the whole project claims,
  measured rather than asserted.
* **Latency and token cost**, so the quality numbers can be read against what
  they cost.

The harness is deterministic given a warm cache: profiles render identically,
document ids are content-addressed, and per-sample nonces are indexed rather than
random. Two consecutive runs produce the same numbers, which is what makes the
regression gate meaningful.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hirelens.assess.pipeline import ScreeningPipeline
from hirelens.config import Settings, get_settings
from hirelens.evals.baselines import Baseline, RandomBaseline, all_baselines
from hirelens.evals.golden import build_golden_set
from hirelens.evals.labels import LabelSet, check_label_quality
from hirelens.evals.metrics import (
    Distribution,
    Estimate,
    inversion_rate,
    kendall_ci,
    spearman,
    spearman_ci,
    top_k_precision,
)
from hirelens.evals.profiles import GoldenSet
from hirelens.ingest.document import SourceDocument, SourceFormat, TextAccumulator
from hirelens.llm.client import LLMClient
from hirelens.retrieve.embeddings import Embedder

logger = logging.getLogger(__name__)


class MetricBlock(BaseModel):
    """Metrics for one ranker against human labels, on one job or pooled."""

    model_config = ConfigDict(frozen=True)

    label: str
    n: int
    spearman: float
    spearman_low: float
    spearman_high: float
    kendall: float
    kendall_low: float
    kendall_high: float
    inversion_rate: float
    top_3_precision: float

    @classmethod
    def build(cls, label: str, system: list[float], human: list[float]) -> MetricBlock:
        rho = spearman_ci(system, human)
        tau = kendall_ci(system, human)
        return cls(
            label=label,
            n=len(system),
            spearman=round(rho.value, 4),
            spearman_low=round(rho.low, 4),
            spearman_high=round(rho.high, 4),
            kendall=round(tau.value, 4),
            kendall_low=round(tau.low, 4),
            kendall_high=round(tau.high, 4),
            inversion_rate=round(inversion_rate(system, human), 4),
            top_3_precision=round(top_k_precision(system, human, k=3), 4),
        )

    @property
    def spearman_estimate(self) -> Estimate:
        return Estimate(self.spearman, self.spearman_low, self.spearman_high, self.n)


class QualityBlock(BaseModel):
    """Properties of the pipeline that are independent of the ranking."""

    model_config = ConfigDict(frozen=True)

    mean_grounding_rate: float
    mean_citation_validity: float
    mean_sample_agreement: float
    mean_confidence_band: float
    ambiguous_requirement_rate: float
    unmet_must_have_rate: float


class CostBlock(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_calls: int
    prompt_tokens: int
    completion_tokens: int
    cache_hit_rate: float
    latency_p50_s: float
    latency_p95_s: float
    wall_clock_s: float


class EvalReport(BaseModel):
    """Everything one harness run produced."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    embedder: str
    self_consistency_k: int
    blind_mode: bool
    pairs_evaluated: int

    pooled: MetricBlock | None = None
    per_job: dict[str, MetricBlock] = Field(default_factory=dict)
    baselines: dict[str, MetricBlock] = Field(default_factory=dict)
    random_ceiling_95: float = 0.0

    quality: QualityBlock | None = None
    cost: CostBlock | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def beats_best_baseline(self) -> bool:
        """True when our point estimate exceeds every baseline's.

        Point estimates only. Whether the *intervals* separate is a stronger
        question, reported alongside rather than folded into this flag.
        """
        if self.pooled is None or not self.baselines:
            return False
        return all(self.pooled.spearman > block.spearman for block in self.baselines.values())

    @property
    def clears_random_noise(self) -> bool:
        """True when our result exceeds what chance produces 95% of the time."""
        return self.pooled is not None and self.pooled.spearman > self.random_ceiling_95

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> EvalReport:
        import json

        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


@dataclass
class _PairResult:
    job_id: str
    candidate_id: str
    score: float
    human: float
    grounding: float
    citation_validity: float
    agreement: float
    band: float
    ambiguous: int
    requirements: int
    unmet_must_haves: int
    elapsed_s: float


@dataclass
class EvalHarness:
    """Runs the pipeline over the golden set and computes the report."""

    golden: GoldenSet = field(default_factory=build_golden_set)
    settings: Settings = field(default_factory=get_settings)
    embedder: Embedder | None = None
    client: LLMClient | None = None

    async def run(self, labels: LabelSet, *, top_k: int = 4) -> EvalReport:
        """Evaluate every labelled (job, candidate) pair."""
        started = time.perf_counter()

        job_ids = [job.job_id for job in self.golden.jobs]
        warnings = [w.message for w in check_label_quality(labels, job_ids)]

        documents = {
            profile.candidate_id: _as_document(profile.candidate_id, profile.render())
            for profile in self.golden.profiles
        }

        client = self.client or LLMClient(settings=self.settings)
        pipeline = ScreeningPipeline(client, settings=self.settings, embedder=self.embedder)

        results: list[_PairResult] = []
        try:
            for job in self.golden.jobs:
                job_labels = labels.for_job(job.job_id)
                if not job_labels:
                    warnings.append(f"Job '{job.job_id}' has no labels and was skipped.")
                    continue

                rubric = await pipeline.compile_rubric(job.text)

                for label in job_labels:
                    profile = self.golden.profile(label.candidate_id)
                    if profile is None:
                        warnings.append(
                            f"Label references unknown candidate '{label.candidate_id}'."
                        )
                        continue

                    outcome = await pipeline.screen(
                        documents[profile.candidate_id],
                        rubric,
                        top_k=top_k,
                        # Interview questions do not affect any metric here and
                        # would add one API call per pair for nothing.
                        with_questions=False,
                    )
                    results.append(
                        _summarise(label.job_id, label.candidate_id, label.value, outcome)
                    )
        finally:
            if self.client is None:
                await client.aclose()

        report = self._build_report(
            results=results,
            labels=labels,
            client=client,
            embedder_name=pipeline.embedder.name if results else "n/a",
            warnings=warnings,
            wall_clock=time.perf_counter() - started,
        )
        return report

    # -- report assembly -----------------------------------------------------

    def _build_report(
        self,
        *,
        results: list[_PairResult],
        labels: LabelSet,
        client: LLMClient,
        embedder_name: str,
        warnings: list[str],
        wall_clock: float,
    ) -> EvalReport:
        usage: dict[str, Any] = client.usage_summary()
        cache = usage.get("cache", {}) if isinstance(usage.get("cache"), dict) else {}

        if not results:
            return EvalReport(
                provider=str(self.settings.llm_provider),
                model=self.settings.active_model,
                embedder=embedder_name,
                self_consistency_k=self.settings.self_consistency_k,
                blind_mode=self.settings.blind_mode,
                pairs_evaluated=0,
                warnings=[*warnings, "No pairs were evaluated. Add labels first."],
            )

        # Pooled correlation is computed on within-job z-scores rather than raw
        # scores. Different jobs have different score distributions, and pooling
        # raw values would let an easy job's inflated scores dominate the
        # coefficient without saying anything about ranking quality.
        pooled_system, pooled_human = _pool_by_job(results)

        per_job: dict[str, MetricBlock] = {}
        for job_id in {r.job_id for r in results}:
            subset = [r for r in results if r.job_id == job_id]
            if len({r.human for r in subset}) > 1:
                per_job[job_id] = MetricBlock.build(
                    job_id, [r.score for r in subset], [r.human for r in subset]
                )

        baselines = self._baseline_metrics(results, labels)
        random_ceiling = RandomBaseline().expected_correlation(pooled_human, spearman)[1]

        requirements = sum(r.requirements for r in results) or 1

        return EvalReport(
            provider=str(self.settings.llm_provider),
            model=self.settings.active_model,
            embedder=embedder_name,
            self_consistency_k=self.settings.self_consistency_k,
            blind_mode=self.settings.blind_mode,
            pairs_evaluated=len(results),
            pooled=MetricBlock.build("pooled", pooled_system, pooled_human),
            per_job=per_job,
            baselines=baselines,
            random_ceiling_95=round(random_ceiling, 4),
            quality=QualityBlock(
                mean_grounding_rate=round(_mean(r.grounding for r in results), 4),
                mean_citation_validity=round(_mean(r.citation_validity for r in results), 4),
                mean_sample_agreement=round(_mean(r.agreement for r in results), 4),
                mean_confidence_band=round(_mean(r.band for r in results), 2),
                ambiguous_requirement_rate=round(
                    sum(r.ambiguous for r in results) / requirements, 4
                ),
                unmet_must_have_rate=round(
                    sum(1 for r in results if r.unmet_must_haves) / len(results), 4
                ),
            ),
            cost=CostBlock(
                api_calls=int(usage.get("api_calls", 0)),
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                cache_hit_rate=float(cache.get("hit_rate", 0.0)),
                latency_p50_s=round(Distribution.of([r.elapsed_s for r in results]).p50, 2),
                latency_p95_s=round(Distribution.of([r.elapsed_s for r in results]).p95, 2),
                wall_clock_s=round(wall_clock, 1),
            ),
            warnings=warnings,
        )

    def _baseline_metrics(
        self, results: list[_PairResult], labels: LabelSet
    ) -> dict[str, MetricBlock]:
        """Run every baseline over the same pairs the pipeline saw."""
        rendered = {p.candidate_id: p.render() for p in self.golden.profiles}
        blocks: dict[str, MetricBlock] = {}

        for baseline in all_baselines():
            paired: list[_PairResult] = []
            for job in self.golden.jobs:
                subset = [r for r in results if r.job_id == job.job_id]
                if not subset:
                    continue
                texts = [rendered[r.candidate_id] for r in subset]
                scores = baseline.score(job.text, texts)
                paired.extend(
                    _PairResult(
                        job_id=r.job_id,
                        candidate_id=r.candidate_id,
                        score=score,
                        human=r.human,
                        grounding=0.0,
                        citation_validity=0.0,
                        agreement=0.0,
                        band=0.0,
                        ambiguous=0,
                        requirements=0,
                        unmet_must_haves=0,
                        elapsed_s=0.0,
                    )
                    for r, score in zip(subset, scores, strict=True)
                )

            if paired:
                system, human = _pool_by_job(paired)
                blocks[baseline.name] = MetricBlock.build(baseline.name, system, human)

        return blocks


# ---------------------------------------------------------------------------


def _summarise(job_id: str, candidate_id: str, human: float, outcome: Any) -> _PairResult:
    a = outcome.assessment
    return _PairResult(
        job_id=job_id,
        candidate_id=candidate_id,
        score=a.score,
        human=human,
        grounding=a.grounding_rate,
        citation_validity=a.citation_validity_rate,
        agreement=a.mean_agreement,
        band=a.uncertainty,
        ambiguous=len(a.needs_review),
        requirements=len(a.assessments),
        unmet_must_haves=len(a.unmet_must_haves),
        elapsed_s=outcome.elapsed_s,
    )


def _pool_by_job(results: list[_PairResult]) -> tuple[list[float], list[float]]:
    """Standardise scores within each job before pooling across jobs.

    Without this, a job whose candidates all score highly would contribute a
    different scale to the pooled correlation than a harder one, and the
    coefficient would partly measure which jobs happened to be easy.
    """
    system: list[float] = []
    human: list[float] = []

    for job_id in sorted({r.job_id for r in results}):
        subset = [r for r in results if r.job_id == job_id]
        system.extend(_standardise([r.score for r in subset]))
        human.extend(_standardise([r.human for r in subset]))

    return system, human


def _standardise(values: list[float]) -> list[float]:
    if len(values) < 2:
        return [0.0] * len(values)
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    spread = variance**0.5
    if spread == 0:
        return [0.0] * len(values)
    return [(v - mean) / spread for v in values]


def _mean(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _as_document(candidate_id: str, text: str) -> SourceDocument:
    """Render profile text into a SourceDocument with a stable content-addressed id."""
    accumulator = TextAccumulator()
    for line in text.splitlines():
        stripped = line.strip()
        is_heading = bool(stripped) and stripped.isupper() and len(stripped) < 40
        accumulator.add_line(line, page=1, is_heading=is_heading)

    return accumulator.build(
        document_id=SourceDocument.make_id(text.encode("utf-8")),
        filename=f"{candidate_id}.txt",
        source_format=SourceFormat.TEXT,
        page_count=1,
    )


__all__ = [
    "Baseline",
    "CostBlock",
    "EvalHarness",
    "EvalReport",
    "MetricBlock",
    "QualityBlock",
]
