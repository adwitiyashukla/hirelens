from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hirelens import __version__
from hirelens.config import Provider, get_settings
from hirelens.ingest import read_document

app = typer.Typer(
    name="hirelens",
    help="Evidence-grounded candidate screening.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else getattr(logging, get_settings().log_level, logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)-7s %(name)-28s %(message)s")


@app.command()
def version() -> None:
    console.print(f"hirelens {__version__}")


@app.command()
def doctor() -> None:
    try:
        settings = get_settings()
    except Exception as exc:
        console.print(Panel(str(exc), title="[red]Configuration error", border_style="red"))
        raise typer.Exit(code=1) from exc

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("provider", f"[cyan]{settings.llm_provider}[/cyan]")
    table.add_row("model", settings.active_model)
    table.add_row("embeddings", f"{settings.embedding_model} [dim](local, no API)[/dim]")
    table.add_row("blind mode", "on" if settings.blind_mode else "[yellow]off[/yellow]")
    table.add_row("cache", str(settings.cache_dir) if settings.cache_enabled else "disabled")
    table.add_row("self-consistency k", str(settings.self_consistency_k))
    console.print(Panel(table, title="Configuration", border_style="cyan"))

    if not settings.has_credentials:
        console.print(
            Panel(
                "No API key configured for this provider.\n\n"
                "Ingestion and parsing still work without one. To enable scoring, add a key "
                "to your .env file, or set HIRELENS_LLM_PROVIDER=ollama to run locally.",
                title="[yellow]No credentials",
                border_style="yellow",
            )
        )
        raise typer.Exit(code=1)

    console.print("\nPinging provider...")
    try:
        reply = asyncio.run(_ping())
    except Exception as exc:
        console.print(Panel(str(exc), title="[red]Provider unreachable", border_style="red"))
        if settings.llm_provider is Provider.OLLAMA:
            console.print(
                "\n[dim]Ollama checklist: `ollama serve` running, and "
                f"`ollama pull {settings.ollama_model}` completed.[/dim]"
            )
        raise typer.Exit(code=1) from exc

    console.print(
        Panel(reply.strip() or "(empty)", title="[green]Provider OK", border_style="green")
    )


async def _ping() -> str:
    from hirelens.llm import LLMClient

    async with LLMClient() as client:
        return await client.chat(
            system="You are a health check. Answer in exactly three words.",
            user="Are you reachable?",
            max_tokens=32,
        )


@app.command()
def models() -> None:
    settings = get_settings()

    if settings.llm_provider is Provider.OLLAMA:
        console.print(
            "[yellow]Ollama serves whatever you have pulled locally. "
            "Run `ollama list` to see it.[/yellow]"
        )
        raise typer.Exit(code=1)

    if not settings.has_credentials:
        console.print("[red]No API key configured. Add one to your .env file.[/red]")
        raise typer.Exit(code=1)

    is_gemini = settings.llm_provider is Provider.GEMINI
    if is_gemini:
        available = asyncio.run(_list_gemini_models(settings.gemini_api_key))
        env_var, highlight_token = "HIRELENS_GEMINI_MODEL", "flash"
    else:
        available = asyncio.run(_list_groq_models(settings.groq_api_key))
        env_var, highlight_token = "HIRELENS_GROQ_MODEL", "llama"

    if not available:
        console.print("[red]No usable models returned. Check the key is valid.[/red]")
        raise typer.Exit(code=1)

    table = Table(title="Models available to your key", header_style="bold")
    table.add_column("model name")
    table.add_column("description", style="dim")
    for name, description in available:
        highlight = "[green]" if highlight_token in name else ""
        table.add_row(f"{highlight}{name}", description[:60])
    console.print(table)

    names = [name for name, _ in available]
    suggested = _suggest_model(names) if is_gemini else _suggest_groq_model(names)
    if suggested:
        console.print(
            Panel(
                f"Put this line in your .env file:\n\n  [cyan]{env_var}={suggested}[/cyan]",
                title="Recommended",
                border_style="green",
            )
        )


async def _list_gemini_models(api_key: str) -> list[tuple[str, str]]:
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={"x-goog-api-key": api_key},
        )
        response.raise_for_status()
        payload = response.json()

    return [
        (
            entry["name"].removeprefix("models/"),
            entry.get("description", ""),
        )
        for entry in payload.get("models", [])
        if "generateContent" in entry.get("supportedGenerationMethods", [])
    ]


async def _list_groq_models(api_key: str) -> list[tuple[str, str]]:
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        payload = response.json()

    models: list[tuple[str, str]] = []
    for entry in payload.get("data", []):
        name = entry.get("id", "")
        if not name:
            continue
        if any(token in name.lower() for token in ("whisper", "guard", "tts")):
            continue
        window = entry.get("context_window")
        owner = entry.get("owned_by", "")
        detail = f"{owner}, {window:,} token context" if window else owner
        models.append((name, detail))

    return sorted(models)


def _suggest_groq_model(names: list[str]) -> str | None:
    preferred = [
        name
        for name in names
        if "llama" in name.lower() and any(token in name for token in ("70b", "versatile"))
    ]
    if preferred:
        return preferred[0]

    instruct = [name for name in names if "instruct" in name.lower() or "llama" in name.lower()]
    return instruct[0] if instruct else (names[0] if names else None)


def _suggest_model(names: list[str]) -> str | None:
    excluded = ("lite", "image", "tts", "thinking", "preview", "robotics")
    candidates = [
        name for name in names if "flash" in name and not any(token in name for token in excluded)
    ]
    if not candidates:
        return names[0] if names else None

    aliases = [name for name in candidates if name.endswith("latest")]
    if aliases:
        return aliases[0]

    return sorted(candidates, reverse=True)[0]


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="Resume file: .pdf, .docx, .txt or .md")],
    show_text: Annotated[bool, typer.Option("--text", help="Print the extracted text")] = False,
    show_blocks: Annotated[
        int, typer.Option("--blocks", help="Print the first N block offsets")
    ] = 0,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    _configure_logging(verbose)

    doc = read_document(path)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("document id", doc.document_id)
    table.add_row("format", str(doc.source_format))
    table.add_row("pages", str(doc.page_count))
    table.add_row("characters", f"{doc.char_count:,}")
    table.add_row("indexed lines", str(len(doc.blocks)))
    table.add_row("headings found", str(sum(1 for b in doc.blocks if b.is_heading)))
    if doc.is_probably_scanned:
        table.add_row("warning", "[yellow]looks like a scan, needs OCR[/yellow]")
    console.print(Panel(table, title=doc.filename, border_style="cyan"))

    if show_blocks:
        block_table = Table(title=f"First {show_blocks} blocks")
        block_table.add_column("span", style="dim")
        block_table.add_column("pg", justify="right")
        block_table.add_column("H", justify="center")
        block_table.add_column("text")
        for block in doc.blocks[:show_blocks]:
            block_table.add_row(
                f"{block.span.start}:{block.span.end}",
                str(block.page),
                "*" if block.is_heading else "",
                doc.slice(block.span)[:72],
            )
        console.print(block_table)

    if show_text:
        console.print(Panel(doc.text, title="Extracted text", border_style="dim"))


@app.command()
def redact_preview(
    path: Annotated[Path, typer.Argument(help="Resume file")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    _configure_logging(verbose)

    from hirelens.extract import redact

    doc = read_document(path)
    report = redact(doc.text)

    table = Table(show_header=True, header_style="bold")
    table.add_column("category")
    table.add_column("found", justify="right")
    table.add_column("examples")
    for category, count in sorted(report.counts.items()):
        examples = [s.text for s in report.spans if str(s.category) == category][:3]
        table.add_row(category, str(count), ", ".join(e[:28] for e in examples))
    console.print(table)

    same_length = len(report.redacted_text) == len(doc.text)
    console.print(
        f"\noffsets preserved: "
        f"{'[green]yes[/green]' if same_length else '[red]NO[/red]'} "
        f"({len(doc.text)} chars in, {len(report.redacted_text)} out)"
    )
    console.print(Panel(report.redacted_text[:900], title="Masked view", border_style="dim"))


@app.command()
def parse(
    path: Annotated[Path, typer.Argument(help="Resume file")],
    blind: Annotated[
        bool | None, typer.Option("--blind/--no-blind", help="Override blind mode")
    ] = None,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write the parsed resume to a file")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    _configure_logging(verbose)

    doc = read_document(path)
    result = asyncio.run(_run_parse(doc, blind))
    resume = result.resume

    console.print(
        Panel(
            f"sections     {', '.join(result.sections_found) or 'none'}\n"
            f"redacted     {result.redaction.summary()}\n"
            f"grounding    {resume.grounding.summary()}"
            + (
                f"\n[yellow]failed[/yellow]       {', '.join(resume.failed_sections)}"
                if resume.failed_sections
                else ""
            ),
            title=f"{doc.filename}",
            border_style="cyan",
        )
    )

    fields = Table(show_header=True, header_style="bold", title="Extracted fields")
    fields.add_column("field")
    fields.add_column("value")
    fields.add_column("cited", justify="center")
    fields.add_column("evidence in document", style="dim")

    def row(label: str, cited: object) -> None:
        from hirelens.schemas.evidence import Cited

        if not isinstance(cited, Cited):
            return
        evidence = ""
        if cited.citations:
            evidence = cited.citations[0].quote.replace("\n", " ")[:52]
        fields.add_row(
            label,
            str(cited.value)[:44],
            "[green]y[/green]" if cited.is_grounded else "[yellow]n[/yellow]",
            evidence,
        )

    b = resume.basics
    for label, value in (
        ("name", b.name),
        ("email", b.email),
        ("location", b.location),
        ("headline", b.headline),
    ):
        row(label, value)
    for profile in b.profiles:
        row(f"profile/{profile.network.value[:12]}", profile.url)
    for job in resume.work:
        row("work/company", job.company)
        row("work/position", job.position)
        for highlight in job.highlights[:2]:
            row("work/highlight", highlight)
    for edu in resume.education:
        row("edu/institution", edu.institution)
        row("edu/degree", edu.degree)
    for project in resume.projects:
        row("project/name", project.name)
        row("project/url", project.url)
    for skill in resume.skills[:12]:
        row("skill", skill.name)

    console.print(fields)

    if resume.grounding.unlocatable_quotes:
        console.print(
            Panel(
                "\n".join(f"- {q}" for q in resume.grounding.unlocatable_quotes[:8]),
                title="[yellow]Quotes the model produced that are not in the document",
                border_style="yellow",
            )
        )

    if json_out:
        json_out.write_text(resume.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"\nwrote {json_out}")


async def _run_parse(doc: object, blind: bool | None) -> object:
    from hirelens.extract import ResumeExtractor
    from hirelens.ingest.document import SourceDocument

    assert isinstance(doc, SourceDocument)
    extractor = ResumeExtractor()
    try:
        return await extractor.extract(doc, blind=blind)
    finally:
        await extractor.client.aclose()


@app.command()
def match(
    resume_path: Annotated[Path, typer.Argument(help="Resume file")],
    jd_path: Annotated[Path, typer.Argument(help="Job description as a .txt or .md file")],
    top_k: Annotated[int, typer.Option("--top-k", help="Evidence units per requirement")] = 3,
    fast_embeddings: Annotated[
        bool,
        typer.Option(
            "--fast-embeddings",
            help="Use the dependency-free hashing embedder instead of the real model",
        ),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    _configure_logging(verbose)

    jd_text = jd_path.read_text(encoding="utf-8")
    doc = read_document(resume_path)
    rubric, hits_by_requirement = asyncio.run(_run_match(doc, jd_text, top_k, fast_embeddings))

    console.print(
        Panel(
            f"role         {rubric.role_title or 'unspecified'}\n"
            f"seniority    {rubric.seniority or 'unspecified'}\n"
            f"requirements {rubric.summary()}",
            title=jd_path.name,
            border_style="cyan",
        )
    )

    for requirement in rubric.requirements:
        hits = hits_by_requirement.get(requirement.requirement_id, [])
        marker = "[red]MUST[/red]" if requirement.is_blocking else "[dim]nice[/dim]"
        console.print(
            f"\n{marker} [bold]{requirement.text}[/bold]  [dim]({requirement.weight:.1f} pts)[/dim]"
        )

        if not hits:
            console.print("     [yellow]no supporting evidence found[/yellow]")
            continue

        for hit in hits:
            quote = hit.unit.text.replace("\n", " ")
            console.print(
                f"     [dim]{hit.found_by:16}[/dim] p{hit.unit.page or '?'} "
                f"[dim]{hit.unit.span.start}:{hit.unit.span.end}[/dim]  {quote[:88]}"
            )


async def _run_match(
    doc: object, jd_text: str, top_k: int, fast_embeddings: bool
) -> tuple[object, dict]:
    from hirelens.assess import RubricCompiler
    from hirelens.extract import ResumeExtractor
    from hirelens.ingest.document import SourceDocument
    from hirelens.retrieve import HashingEmbedder, HybridRetriever, chunk_resume, get_embedder

    assert isinstance(doc, SourceDocument)
    settings = get_settings()

    extractor = ResumeExtractor()
    try:
        compiler = RubricCompiler(extractor.client)
        rubric, extraction = await asyncio.gather(compiler.compile(jd_text), extractor.extract(doc))
    finally:
        await extractor.client.aclose()

    embedder = HashingEmbedder() if fast_embeddings else get_embedder(settings.embedding_model)
    retriever = HybridRetriever(units=chunk_resume(extraction.resume), embedder=embedder)

    hits = retriever.search_many(
        {r.requirement_id: r.query for r in rubric.requirements}, top_k=top_k
    )
    return rubric, hits


@app.command()
def score(
    resume_paths: Annotated[list[Path], typer.Argument(help="One or more resume files")],
    jd: Annotated[Path, typer.Option("--jd", help="Job description file")],
    top_k: Annotated[int, typer.Option("--top-k", help="Evidence units per requirement")] = 4,
    no_questions: Annotated[
        bool, typer.Option("--no-questions", help="Skip interview question generation")
    ] = False,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write full assessments to a file")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    _configure_logging(verbose)

    jd_text = jd.read_text(encoding="utf-8")
    documents = [read_document(path) for path in resume_paths]
    rubric, results = asyncio.run(_run_score(documents, jd_text, top_k, not no_questions))

    console.print(
        Panel(
            f"role         {rubric.role_title or 'unspecified'}\n"
            f"rubric       {rubric.summary()}\n"
            f"candidates   {len(results)} of {len(documents)} screened",
            title=jd.name,
            border_style="cyan",
        )
    )

    if len(results) > 1:
        _print_shortlist(results)

    for result in results:
        _print_assessment(result)

    if json_out:
        payload = {
            "rubric": rubric.model_dump(mode="json"),
            "assessments": [r.assessment.model_dump(mode="json") for r in results],
        }
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"\nwrote {json_out}")


def _print_shortlist(results: list) -> None:
    table = Table(title="Shortlist", header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("candidate")
    table.add_column("score", justify="right")
    table.add_column("band")
    table.add_column("must-haves")
    table.add_column("agree", justify="right")

    for position, result in enumerate(results, start=1):
        a = result.assessment
        low, high = a.score_range
        table.add_row(
            str(position),
            a.candidate_label,
            f"{a.score:.0f} [dim]({low:.0f}-{high:.0f})[/dim]",
            a.band,
            "[green]all met[/green]"
            if a.meets_all_must_haves
            else f"[red]{len(a.unmet_must_haves)} unmet[/red]",
            f"{a.mean_agreement:.0%}",
        )
    console.print(table)


def _print_assessment(result: object) -> None:
    from hirelens.schemas.assessment import Verdict

    a = result.assessment  # type: ignore[attr-defined]
    colours = {
        Verdict.STRONG: "green",
        Verdict.CLEAR: "green",
        Verdict.PARTIAL: "yellow",
        Verdict.WEAK: "red",
        Verdict.NONE: "red",
    }

    console.print(
        Panel(
            f"{a.summary()}\n"
            f"[dim]grounding {a.grounding_rate:.0%}, citations valid "
            f"{a.citation_validity_rate:.0%}, "
            f"{result.evidence_unit_count} evidence units, "  # type: ignore[attr-defined]
            f"{result.elapsed_s:.1f}s[/dim]",  # type: ignore[attr-defined]
            title=f"{a.candidate_label}",
            border_style="green" if a.meets_all_must_haves else "yellow",
        )
    )

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("")
    table.add_column("requirement")
    table.add_column("verdict")
    table.add_column("pts", justify="right")
    table.add_column("evidence", style="dim")

    for item in a.sorted_assessments():
        colour = colours[item.verdict]
        evidence = item.citations[0].quote.replace("\n", " ")[:46] if item.citations else ""
        flag = "[red]MUST[/red]" if item.kind.value == "must_have" else "[dim]nice[/dim]"
        review = " [yellow]?[/yellow]" if item.is_ambiguous else ""
        table.add_row(
            flag,
            item.requirement_text[:46],
            f"[{colour}]{item.verdict}[/{colour}]{review}",
            f"{item.points:.1f}/{item.max_points:.1f}",
            evidence,
        )
    console.print(table)

    if a.risks:
        console.print("\n[bold]Risk flags[/bold]")
        for risk in a.risks:
            colour = {"high": "red", "medium": "yellow", "low": "dim"}[risk.level.value]
            console.print(f"  [{colour}]{risk.level.value:6}[/{colour}] {risk.message}")

    if a.questions:
        console.print("\n[bold]Interview questions[/bold]")
        for index, question in enumerate(a.questions, start=1):
            console.print(f"  {index}. {question.question}")
            console.print(f"     [dim]{question.rationale}[/dim]")
    console.print()


async def _run_score(
    documents: list, jd_text: str, top_k: int, with_questions: bool
) -> tuple[object, list]:
    from hirelens.assess import ScreeningPipeline

    async with ScreeningPipeline() as pipeline:
        return await pipeline.screen_batch(
            documents, jd_text, top_k=top_k, with_questions=with_questions
        )


if __name__ == "__main__":
    app()
