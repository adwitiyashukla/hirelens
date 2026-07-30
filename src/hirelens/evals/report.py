"""Render an evaluation report, and gate CI on it.

Two outputs from the same data: a console view for working, and a markdown table
for the README. The README table is the artefact a recruiter reads, so it has to
be honest by construction rather than by intention. That means the baseline row
and the confidence intervals are printed whether or not they are flattering.

The regression gate is what makes prompt engineering into engineering. A prompt
change that raises the score on one hand-checked example and lowers agreement
across the golden set is a regression, and without a gate nobody would ever find
out.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hirelens.evals.runner import EvalReport

# How far a metric may drop before the gate fails. Small but not zero: bootstrap
# intervals and provider nondeterminism move the third decimal place, and a gate
# that fires on noise gets disabled within a week.
DEFAULT_TOLERANCE = 0.05


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def to_markdown(report: EvalReport) -> str:
    """The table that goes in the README."""
    lines: list[str] = []
    add = lines.append

    add("## Evaluation results")
    add("")
    add(
        f"Run against the {report.pairs_evaluated}-pair golden set "
        f"(`{report.provider}` / `{report.model}`, embeddings `{report.embedder}`, "
        f"self-consistency k={report.self_consistency_k}, "
        f"blind mode {'on' if report.blind_mode else 'off'})."
    )
    add("")

    if report.pooled is None:
        add("> No pairs evaluated. Add human labels with `hirelens label` first.")
        return "\n".join(lines)

    add("### Agreement with human ranking")
    add("")
    add(
        "| Ranker | Spearman rho (95% CI) | Kendall tau-b | Pairwise inversions | Top-3 precision |"
    )
    add("|---|---|---|---|---|")
    add(
        f"| **HireLens** | **{report.pooled.spearman:.3f}** "
        f"[{report.pooled.spearman_low:.2f}, {report.pooled.spearman_high:.2f}] "
        f"| {report.pooled.kendall:.3f} "
        f"| {report.pooled.inversion_rate:.1%} "
        f"| {report.pooled.top_3_precision:.0%} |"
    )
    for name, block in sorted(report.baselines.items()):
        add(
            f"| {name} | {block.spearman:.3f} "
            f"[{block.spearman_low:.2f}, {block.spearman_high:.2f}] "
            f"| {block.kendall:.3f} | {block.inversion_rate:.1%} "
            f"| {block.top_3_precision:.0%} |"
        )
    add("")
    add(
        f"Random ordering reaches rho = {report.random_ceiling_95:.2f} by chance 5% of the "
        f"time on a set this size, so that is the floor any result has to clear."
    )
    add("")

    if report.per_job:
        add("### Per job")
        add("")
        add("| Job | n | Spearman rho | Inversions |")
        add("|---|---|---|---|")
        for job_id, block in sorted(report.per_job.items()):
            add(f"| {job_id} | {block.n} | {block.spearman:.3f} | {block.inversion_rate:.1%} |")
        add("")

    if report.quality:
        q = report.quality
        add("### Pipeline properties")
        add("")
        add("| Property | Value | What it means |")
        add("|---|---|---|")
        add(
            f"| Citation validity | {q.mean_citation_validity:.1%} "
            f"| Share of cited spans that really contain the quoted text |"
        )
        add(
            f"| Grounding rate | {q.mean_grounding_rate:.1%} "
            f"| Share of extracted fields traceable to the document |"
        )
        add(
            f"| Self-consistency | {q.mean_sample_agreement:.1%} "
            f"| Share of repeated judgements that agreed |"
        )
        add(
            f"| Confidence band | {q.mean_confidence_band:.1f} pts "
            f"| Mean width of the score interval across samples |"
        )
        add(
            f"| Flagged for review | {q.ambiguous_requirement_rate:.1%} "
            f"| Requirements where repeated runs disagreed |"
        )
        add("")

    if report.cost:
        c = report.cost
        add("### Cost and latency")
        add("")
        add(
            "| API calls | Prompt tokens | Completion tokens | Cache hits "
            "| p50 latency | p95 latency |"
        )
        add("|---|---|---|---|---|---|")
        add(
            f"| {c.api_calls} | {c.prompt_tokens:,} | {c.completion_tokens:,} "
            f"| {c.cache_hit_rate:.0%} | {c.latency_p50_s:.1f}s | {c.latency_p95_s:.1f}s |"
        )
        add("")

    if report.warnings:
        add("### Caveats")
        add("")
        for warning in report.warnings:
            add(f"- {warning}")
        add("")

    add(
        "> Read the intervals, not just the point estimates. A golden set of this size "
        "cannot support a confident claim, and the width of the interval says so."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------


def to_console(report: EvalReport) -> str:
    """Compact plain-text view for the terminal."""
    out: list[str] = []
    add = out.append
    rule = "=" * 74

    add(rule)
    add(f"EVALUATION  {report.provider}/{report.model}  k={report.self_consistency_k}")
    add(f"            embeddings {report.embedder}, {report.pairs_evaluated} pairs")
    add(rule)

    if report.pooled is None:
        add("No pairs evaluated. Run `hirelens label` to create human ground truth.")
        for warning in report.warnings:
            add(f"  ! {warning}")
        return "\n".join(out)

    add("")
    add(f"{'ranker':<12} {'spearman':>22} {'kendall':>9} {'inversions':>12} {'top-3':>7}")
    add("-" * 74)
    add(
        f"{'HireLens':<12} {report.pooled.spearman_estimate.format():>22} "
        f"{report.pooled.kendall:>9.3f} {report.pooled.inversion_rate:>11.1%} "
        f"{report.pooled.top_3_precision:>7.0%}"
    )
    for name, block in sorted(report.baselines.items()):
        add(
            f"{name:<12} {block.spearman_estimate.format():>22} "
            f"{block.kendall:>9.3f} {block.inversion_rate:>11.1%} "
            f"{block.top_3_precision:>7.0%}"
        )
    add("-" * 74)
    add(f"{'random 95th':<12} {report.random_ceiling_95:>22.3f}   (chance ceiling)")

    add("")
    add(f"beats every baseline : {'yes' if report.beats_best_baseline else 'NO'}")
    add(f"clears chance        : {'yes' if report.clears_random_noise else 'NO'}")

    if report.per_job:
        add("")
        add("per job:")
        for job_id, block in sorted(report.per_job.items()):
            add(
                f"  {job_id:<12} n={block.n:<3} rho={block.spearman:>6.3f}  "
                f"inversions={block.inversion_rate:.0%}"
            )

    if report.quality:
        q = report.quality
        add("")
        add("pipeline properties:")
        add(f"  citation validity  {q.mean_citation_validity:>6.1%}")
        add(f"  grounding rate     {q.mean_grounding_rate:>6.1%}")
        add(f"  self-consistency   {q.mean_sample_agreement:>6.1%}")
        add(f"  confidence band    {q.mean_confidence_band:>6.1f} pts")
        add(f"  flagged for review {q.ambiguous_requirement_rate:>6.1%}")

    if report.cost:
        c = report.cost
        add("")
        add(
            f"cost: {c.api_calls} calls, {c.prompt_tokens:,} in / "
            f"{c.completion_tokens:,} out, cache {c.cache_hit_rate:.0%}, "
            f"p95 {c.latency_p95_s:.1f}s, wall {c.wall_clock_s:.0f}s"
        )

    if report.warnings:
        add("")
        add("warnings:")
        for warning in report.warnings:
            add(f"  ! {warning}")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Regression gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    failures: list[str]
    notes: list[str]

    def render(self) -> str:
        lines = ["PASS" if self.passed else "FAIL"]
        lines += [f"  x {failure}" for failure in self.failures]
        lines += [f"  - {note}" for note in self.notes]
        return "\n".join(lines)


def check_regression(
    current: EvalReport,
    baseline: EvalReport | None,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    require_beats_baselines: bool = True,
) -> GateResult:
    """Decide whether this run is acceptable. Used by CI.

    Absolute checks run whether or not a stored baseline exists; the comparison
    checks only run when there is something to compare against.
    """
    failures: list[str] = []
    notes: list[str] = []

    if current.pooled is None:
        return GateResult(False, ["No pairs were evaluated."], current.warnings)

    # -- absolute floors -----------------------------------------------------

    if require_beats_baselines and not current.beats_best_baseline:
        best = max(current.baselines.items(), key=lambda kv: kv[1].spearman, default=None)
        if best is not None:
            failures.append(
                f"Does not beat the '{best[0]}' baseline "
                f"({current.pooled.spearman:.3f} vs {best[1].spearman:.3f}). "
                f"The pipeline is not earning its complexity."
            )

    if not current.clears_random_noise:
        failures.append(
            f"Agreement ({current.pooled.spearman:.3f}) does not clear the chance ceiling "
            f"({current.random_ceiling_95:.3f}). This result is indistinguishable from noise."
        )

    if current.quality and current.quality.mean_citation_validity < 0.90:
        failures.append(
            f"Citation validity {current.quality.mean_citation_validity:.1%} is below 90%. "
            f"The grounding claim is the core of this project."
        )

    # -- comparison against the stored baseline ------------------------------

    if baseline is None or baseline.pooled is None:
        notes.append("No stored baseline. Recording this run as the new baseline.")
        return GateResult(not failures, failures, notes + current.warnings)

    drop = baseline.pooled.spearman - current.pooled.spearman
    if drop > tolerance:
        failures.append(
            f"Agreement dropped {drop:.3f} (from {baseline.pooled.spearman:.3f} to "
            f"{current.pooled.spearman:.3f}), beyond the {tolerance:.2f} tolerance."
        )
    elif drop > 0:
        notes.append(f"Agreement down {drop:.3f}, within tolerance.")
    else:
        notes.append(f"Agreement up {-drop:.3f}.")

    if current.quality and baseline.quality:
        consistency_drop = (
            baseline.quality.mean_sample_agreement - current.quality.mean_sample_agreement
        )
        if consistency_drop > tolerance:
            failures.append(
                f"Self-consistency dropped {consistency_drop:.3f}. The system became less "
                f"stable, even if agreement held."
            )

    return GateResult(not failures, failures, notes + current.warnings)


def write_markdown(report: EvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_markdown(report), encoding="utf-8")
