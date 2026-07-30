"""Audit commands.

``plan`` exists because the audit is the most expensive thing in the project and
a free-tier quota is easy to exhaust by accident. It prints the call estimate and
the exact experiment matrix without spending anything, so the cost is a decision
rather than a surprise.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hirelens.audit.perturbations import DEFAULT_AXES, Axis, build_plan, estimate_calls
from hirelens.audit.report import check_audit, to_console, to_markdown
from hirelens.audit.runner import AuditReport, FairnessAudit
from hirelens.evals.golden import build_golden_set

app = typer.Typer(name="audit", help="Counterfactual fairness audit.", no_args_is_help=True)
console = Console()

DEFAULT_REPORT = Path("evals/reports/fairness.json")
DEFAULT_MARKDOWN = Path("docs/BIAS_AUDIT.md")


@dataclass(frozen=True)
class Budget:
    """A preset sizing of the experiment, with its own sampling depth.

    The audit runs with the response cache off (see
    :func:`hirelens.audit.perturbations.estimate_calls` for why), so cost scales
    linearly with every dimension. ``k`` is the cheapest lever and the one with the
    clearest trade-off: fewer samples is cheaper but noisier, and a noisier system
    has a higher noise floor, which makes small drift undetectable. It never causes
    a false positive, only a missed one.
    """

    variants_per_axis: int | None
    profiles: int
    k: int
    both_modes: bool
    note: str


BUDGETS: dict[str, Budget] = {
    "tiny": Budget(
        variants_per_axis=2,
        profiles=2,
        k=2,
        both_modes=False,
        note="Blind mode only. Smoke-tests the audit; too small to conclude much.",
    ),
    "small": Budget(
        variants_per_axis=2,
        profiles=3,
        k=2,
        both_modes=True,
        note="Blind and sighted. Detects drift above roughly 3 points.",
    ),
    "medium": Budget(
        variants_per_axis=3,
        profiles=4,
        k=3,
        both_modes=True,
        note="Tighter noise floor, so smaller drift becomes visible.",
    ),
    "large": Budget(
        variants_per_axis=None,
        profiles=8,
        k=5,
        both_modes=True,
        note="Every variant. Realistically an overnight Ollama run.",
    ),
}


def _budget(name: str) -> Budget:
    if name not in BUDGETS:
        raise typer.BadParameter(f"Unknown budget. Choose from: {', '.join(BUDGETS)}")
    return BUDGETS[name]


def _resolve_axes(names: list[str] | None) -> tuple[Axis, ...]:
    if not names:
        return DEFAULT_AXES
    try:
        return tuple(Axis(name) for name in names)
    except ValueError as exc:
        valid = ", ".join(str(a) for a in Axis)
        raise typer.BadParameter(f"Unknown axis. Valid axes: {valid}") from exc


@app.command("plan")
def plan_command(
    budget: Annotated[str, typer.Option("--budget", help="tiny, small, medium or large")] = "small",
    job: Annotated[str, typer.Option("--job")] = "backend",
    axis: Annotated[list[str] | None, typer.Option("--axis", help="Repeatable")] = None,
) -> None:
    """Show the experiment matrix and the real cost. Spends nothing."""
    spec = _budget(budget)
    axes = _resolve_axes(axis)
    variants = build_plan(axes, variants_per_axis=spec.variants_per_axis)

    golden = build_golden_set()
    profile_count = min(spec.profiles, len(golden.profiles))
    modes = 2 if spec.both_modes else 1

    calls = estimate_calls(
        profiles=profile_count,
        variants=len(variants),
        modes=modes,
        self_consistency_k=spec.k,
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("axis")
    table.add_column("variants", justify="right")
    table.add_column("groups")
    for axis_value in (Axis.NULL, *axes):
        selected = [v for v in variants if v.axis is axis_value]
        if selected:
            table.add_row(
                str(axis_value),
                str(len(selected)),
                ", ".join(sorted({v.group for v in selected})),
            )
    console.print(table)

    console.print(
        Panel(
            f"profiles         {profile_count}\n"
            f"variants         {len(variants)}\n"
            f"modes            {'blind and sighted' if spec.both_modes else 'blind only'}\n"
            f"screenings       {profile_count * len(variants) * modes}\n"
            f"self-consistency k={spec.k}\n\n"
            f"[bold]API calls: ~{calls:,}[/bold]\n\n"
            f"[dim]{spec.note}\n\n"
            f"The cache is off for the audit, so this is the real number rather than an "
            f"upper bound. With caching on, two runs of an identical resume would return "
            f"the same score from the same cache entry, and the null control would measure "
            f"a noise floor of zero by construction.[/dim]",
            title=f"Audit plan ({budget}, job '{job}')",
            border_style="cyan",
        )
    )


@app.command("run")
def run_command(
    budget: Annotated[str, typer.Option("--budget", help="tiny, small, medium or large")] = "small",
    job: Annotated[str, typer.Option("--job")] = "backend",
    axis: Annotated[list[str] | None, typer.Option("--axis", help="Repeatable")] = None,
    threshold: Annotated[
        float | None, typer.Option("--threshold", help="Points of excess drift allowed")
    ] = None,
    blind_only: Annotated[
        bool, typer.Option("--blind-only", help="Skip the sighted diagnostic run, halving cost")
    ] = False,
    k: Annotated[
        int | None,
        typer.Option(
            "--k", help="Override self-consistency samples. Lower is cheaper but noisier."
        ),
    ] = None,
    report_path: Annotated[Path, typer.Option("--report")] = DEFAULT_REPORT,
    markdown_path: Annotated[Path, typer.Option("--markdown")] = DEFAULT_MARKDOWN,
    gate: Annotated[bool, typer.Option("--gate", help="Exit non-zero if the audit fails")] = False,
    fast_embeddings: Annotated[bool, typer.Option("--fast-embeddings")] = False,
) -> None:
    """Run the counterfactual bias audit."""
    spec = _budget(budget)
    axes = _resolve_axes(axis)

    golden = build_golden_set()
    profile_ids = [p.candidate_id for p in golden.profiles][: spec.profiles]

    embedder = None
    if fast_embeddings:
        from hirelens.retrieve.embeddings import HashingEmbedder

        embedder = HashingEmbedder()

    console.print(f"[dim]Running the {budget} audit on job '{job}'. This costs API calls.[/dim]\n")

    audit = FairnessAudit(embedder=embedder)
    report = asyncio.run(
        audit.run(
            job_id=job,
            profile_ids=profile_ids,
            axes=axes,
            variants_per_axis=spec.variants_per_axis,
            both_modes=spec.both_modes and not blind_only,
            threshold=threshold,
            # An explicit --k wins; otherwise the budget's own sampling depth
            # applies, since that is what its cost estimate was based on.
            k=k if k is not None else spec.k,
        )
    )

    console.print(to_console(report))

    report.save(report_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(to_markdown(report), encoding="utf-8")
    console.print(f"\nwrote {report_path} and {markdown_path}")

    result = check_audit(report)
    console.print("\n[bold]Audit gate[/bold]")
    console.print(result.render())

    if gate and not result.passed:
        raise typer.Exit(code=1)


@app.command("gate")
def gate_command(
    report_path: Annotated[Path, typer.Option("--report")] = DEFAULT_REPORT,
) -> None:
    """Check a stored audit report without re-running it."""
    if not report_path.exists():
        console.print(f"[red]No audit report at {report_path}.[/red]")
        raise typer.Exit(code=1)

    result = check_audit(AuditReport.load(report_path))
    console.print(result.render())
    if not result.passed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
