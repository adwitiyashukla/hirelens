"""The evaluation harness: golden set, metrics, baselines, and the regression gate."""

from hirelens.evals.baselines import (
    Baseline,
    BM25Baseline,
    KeywordOverlapBaseline,
    RandomBaseline,
    all_baselines,
)
from hirelens.evals.golden import build_golden_set
from hirelens.evals.labels import (
    TIER_DEFINITIONS,
    TIER_ORDER,
    TIER_VALUES,
    Label,
    LabelSet,
    Tier,
    check_label_quality,
)
from hirelens.evals.metrics import (
    Distribution,
    Estimate,
    bootstrap,
    inversion_rate,
    kendall_tau_b,
    rank_with_ties,
    spearman,
    spearman_ci,
    top_k_precision,
)
from hirelens.evals.profiles import (
    CandidateProfile,
    Demographics,
    GoldenSet,
    JobSpec,
    QualityTier,
)
from hirelens.evals.report import check_regression, to_console, to_markdown
from hirelens.evals.runner import EvalHarness, EvalReport, MetricBlock

__all__ = [
    "TIER_DEFINITIONS",
    "TIER_ORDER",
    "TIER_VALUES",
    "BM25Baseline",
    "Baseline",
    "CandidateProfile",
    "Demographics",
    "Distribution",
    "Estimate",
    "EvalHarness",
    "EvalReport",
    "GoldenSet",
    "JobSpec",
    "KeywordOverlapBaseline",
    "Label",
    "LabelSet",
    "MetricBlock",
    "QualityTier",
    "RandomBaseline",
    "Tier",
    "all_baselines",
    "bootstrap",
    "build_golden_set",
    "check_label_quality",
    "check_regression",
    "inversion_rate",
    "kendall_tau_b",
    "rank_with_ties",
    "spearman",
    "spearman_ci",
    "to_console",
    "to_markdown",
    "top_k_precision",
]
