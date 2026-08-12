from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hirelens.evals.golden import build_golden_set
from hirelens.evals.labels import TIER_DEFINITIONS, TIER_ORDER, Label, LabelSet, Tier
from hirelens.evals.report import check_regression, to_console, to_markdown
from hirelens.evals.runner import EvalHarness, EvalReport

app = typer.Typer(name="evals", help="Evaluation harness.", no_args_is_help=True)
console = Console()

DEFAULT_LABELS = Path("evals/data/labels.json")
DEFAULT_BASELINE = Path("evals/reports/baseline.json")
DEFAULT_REPORT = Path("evals/reports/latest.json")
DEFAULT_MARKDOWN = Path("evals/reports/latest.md")


@app.command("generate")
def generate(
    out: Annotated[Path, typer.Option("--out", help="Where to write rendered resumes")] = Path(
        "evals/data/resumes"
    ),
) -> None:
    golden = build_golden_set()

    for profile in golden.profiles:
        profile.write(out)

    jobs_dir = out.parent / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    for job in golden.jobs:
        (jobs_dir / f"{job.job_id}.txt").write_text(job.text, encoding="utf-8")

    golden.save(out.parent / "golden_set.json")

    console.print(
        Panel(
            f"{len(golden.profiles)} resumes -> {out}\n"
            f"{len(golden.jobs)} jobs -> {jobs_dir}\n"
            f"{golden.pair_count} pairs to label\n\n"
            f"Next: [cyan]hirelens-evals label[/cyan]",
            title="Golden set generated",
            border_style="green",
        )
    )


@app.command("label")
def label(
    labels_path: Annotated[Path, typer.Option("--labels")] = DEFAULT_LABELS,
    job: Annotated[str | None, typer.Option("--job", help="Label one job only")] = None,
    labeller: Annotated[str, typer.Option("--as", help="Your name, recorded on each label")] = "",
    redo: Annotated[bool, typer.Option("--redo", help="Re-label pairs already done")] = False,
) -> None:
    golden = build_golden_set()
    labels = LabelSet.load(labels_path)

    jobs = [j for j in golden.jobs if job is None or j.job_id == job]
    if not jobs:
        console.print(f"[red]No job matching '{job}'.[/red]")
        raise typer.Exit(code=1)

    pending = [
        (j, p)
        for j in jobs
        for p in golden.profiles
        if redo or labels.get(j.job_id, p.candidate_id) is None
    ]

    if not pending:
        console.print("[green]Every pair is already labelled.[/green] Use --redo to revise.")
        _show_coverage(labels, golden)
        return

    console.print(
        Panel(
            f"{len(pending)} pair(s) to label. Saved after every answer, so you can stop "
            f"and resume at any point.\n\n"
            + "\n".join(
                f"  [cyan]{index}[/cyan]  {tier!s:<11} {TIER_DEFINITIONS[tier]}"
                for index, tier in enumerate(TIER_ORDER, start=1)
            )
            + "\n  [cyan]s[/cyan]  skip     [cyan]q[/cyan]  save and quit",
            title="Labelling",
            border_style="cyan",
        )
    )

    for position, (job_spec, profile) in enumerate(pending, start=1):
        console.print(f"\n[dim]{'=' * 74}[/dim]")
        console.print(f"[dim]{position} of {len(pending)}[/dim]")
        console.print(
            Panel(job_spec.text.strip(), title=f"JOB: {job_spec.title}", border_style="blue")
        )
        console.print(
            Panel(
                profile.render().strip(),
                title=f"CANDIDATE: {profile.candidate_id}",
                border_style="white",
            )
        )

        existing = labels.get(job_spec.job_id, profile.candidate_id)
        if existing:
            console.print(f"[dim]current: {existing.tier} ({existing.rationale})[/dim]")

        answer = (
            typer.prompt("Would you interview them for this role? (1-5, s, q)", default="s")
            .strip()
            .lower()
        )

        if answer == "q":
            break
        if answer == "s" or not answer:
            continue
        if not answer.isdigit() or not 1 <= int(answer) <= len(TIER_ORDER):
            console.print("[yellow]Not a valid tier, skipping.[/yellow]")
            continue

        tier: Tier = TIER_ORDER[int(answer) - 1]
        rationale = typer.prompt("Why? (one line)", default="").strip()

        labels.upsert(
            Label.create(
                job_spec.job_id,
                profile.candidate_id,
                tier,
                rationale=rationale,
                labeller=labeller or "unknown",
            )
        )
        labels.save(labels_path)
        console.print(f"[green]saved[/green] {profile.candidate_id} -> {tier}")

    labels.save(labels_path)
    console.print(f"\nWrote {labels_path}")
    _show_coverage(labels, golden)


def _show_coverage(labels: LabelSet, golden) -> None:
    table = Table(title="Label coverage", header_style="bold")
    table.add_column("job")
    table.add_column("labelled", justify="right")
    table.add_column("tier spread")

    for job in golden.jobs:
        distribution = labels.tier_distribution(job.job_id)
        spread = " ".join(
            f"{name.split('_')[-1][:2]}:{count}" for name, count in distribution.items()
        )
        table.add_row(
            job.job_id, f"{len(labels.for_job(job.job_id))}/{len(golden.profiles)}", spread
        )
    console.print(table)


@app.command("run")
def run(
    labels_path: Annotated[Path, typer.Option("--labels")] = DEFAULT_LABELS,
    report_path: Annotated[Path, typer.Option("--report")] = DEFAULT_REPORT,
    markdown_path: Annotated[Path, typer.Option("--markdown")] = DEFAULT_MARKDOWN,
    baseline_path: Annotated[Path, typer.Option("--baseline")] = DEFAULT_BASELINE,
    gate: Annotated[bool, typer.Option("--gate", help="Exit non-zero on regression")] = False,
    update_baseline: Annotated[
        bool, typer.Option("--update-baseline", help="Record this run as the new baseline")
    ] = False,
    fast_embeddings: Annotated[
        bool, typer.Option("--fast-embeddings", help="Use the hashing embedder")
    ] = False,
    top_k: Annotated[int, typer.Option("--top-k")] = 4,
) -> None:
    labels = LabelSet.load(labels_path)
    if not labels.labels:
        console.print(
            Panel(
                f"No labels found at {labels_path}.\n\n"
                f"Human ground truth is the one thing that cannot be generated. Run "
                f"[cyan]hirelens-evals label[/cyan] first.",
                title="[yellow]Nothing to evaluate",
                border_style="yellow",
            )
        )
        raise typer.Exit(code=1)

    embedder = None
    if fast_embeddings:
        from hirelens.retrieve.embeddings import HashingEmbedder

        embedder = HashingEmbedder()

    harness = EvalHarness(embedder=embedder)
    report = asyncio.run(harness.run(labels, top_k=top_k))

    console.print(to_console(report))

    report.save(report_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(to_markdown(report), encoding="utf-8")
    console.print(f"\nwrote {report_path} and {markdown_path}")

    stored = EvalReport.load(baseline_path) if baseline_path.exists() else None
    result = check_regression(report, stored)
    console.print("\n[bold]Regression gate[/bold]")
    console.print(result.render())

    if update_baseline or (stored is None and result.passed):
        report.save(baseline_path)
        console.print(f"recorded baseline at {baseline_path}")

    if gate and not result.passed:
        raise typer.Exit(code=1)


@app.command("gate")
def gate_only(
    report_path: Annotated[Path, typer.Option("--report")] = DEFAULT_REPORT,
    baseline_path: Annotated[Path, typer.Option("--baseline")] = DEFAULT_BASELINE,
) -> None:
    if not report_path.exists():
        console.print(f"[red]No report at {report_path}.[/red]")
        raise typer.Exit(code=1)

    current = EvalReport.load(report_path)
    stored = EvalReport.load(baseline_path) if baseline_path.exists() else None
    result = check_regression(current, stored)

    console.print(result.render())
    if not result.passed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
