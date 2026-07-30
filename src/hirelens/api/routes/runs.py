"""Screening runs, live progress, and results."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from hirelens.api.db.repository import (
    AssessmentRepository,
    DocumentRepository,
    JobRepository,
    RunRepository,
)
from hirelens.api.deps import RunnerDep, SessionDep
from hirelens.api.runner import ScreeningRunner
from hirelens.api.schemas import (
    AssessmentDetail,
    CitationOut,
    DocumentOut,
    HighlightBox,
    QuestionOut,
    RequirementResultOut,
    RiskOut,
    RunCreate,
    RunOut,
    ShortlistEntry,
    ShortlistOut,
)
from hirelens.ingest.document import BoundingBox, TextBlock
from hirelens.schemas.assessment import VERDICT_VALUES, Verdict
from hirelens.schemas.evidence import Citation, Span

logger = logging.getLogger(__name__)

router = APIRouter(tags=["runs"])

#: Heartbeat interval for the event stream. Proxies and load balancers close idle
#: connections after 30 to 60 seconds, and a screening stage can easily be quiet
#: for longer than that, so a comment frame keeps the connection alive.
_HEARTBEAT_S = 15.0


@router.post("/runs", response_model=RunOut, status_code=status.HTTP_202_ACCEPTED)
async def create_run(
    payload: RunCreate,
    session: AsyncSession = SessionDep,
    runner: ScreeningRunner = RunnerDep,
) -> RunOut:
    """Start screening a batch of candidates against a job.

    Returns **202 Accepted** immediately. Screening takes tens of seconds per
    candidate, so it happens in the background; subscribe to
    ``GET /api/runs/{id}/events`` for live progress or poll ``GET /api/runs/{id}``.
    """
    job = await JobRepository(session).get(payload.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {payload.job_id} not found")

    documents = await DocumentRepository(session).get_many(payload.document_ids)
    found = {document.id for document in documents}
    missing = [i for i in payload.document_ids if i not in found]
    if missing:
        raise HTTPException(
            status_code=404, detail=f"Unknown document ids: {', '.join(missing[:5])}"
        )

    run = await RunRepository(session).create(
        job_id=job.id, total=len(documents), blind_mode=payload.blind_mode
    )
    # Commit before handing off, or the background task opens its own session and
    # cannot see a run that is still sitting in this transaction.
    await session.commit()

    runner.submit(
        run.id,
        [document.id for document in documents],
        top_k=payload.top_k,
        with_questions=payload.with_questions,
    )
    return RunOut.model_validate(run)


@router.get("/runs/{run_id}", response_model=RunOut)
async def get_run(run_id: str, session: AsyncSession = SessionDep) -> RunOut:
    run = await RunRepository(session).get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return RunOut.model_validate(run)


@router.get("/runs/{run_id}/events")
async def stream_run_events(
    run_id: str,
    session: AsyncSession = SessionDep,
    runner: ScreeningRunner = RunnerDep,
) -> StreamingResponse:
    """Stream run progress as server-sent events.

    SSE rather than WebSockets: the traffic is one-directional, it survives proxies
    that mangle WebSocket upgrades, and browsers reconnect automatically. There is
    nothing here a bidirectional channel would buy.

    A client that subscribes after the run finished still receives one terminal
    event and a clean close, because the channel retains its last state.
    """
    run = await RunRepository(session).get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    channel = runner.channel(run_id)

    # A run that finished before anyone subscribed leaves an empty channel. Seed
    # it from the database so the stream reports the truth instead of hanging.
    if channel.latest is None:
        from hirelens.api.schemas import RunProgress

        channel.publish(
            RunProgress(
                run_id=run.id,
                status=run.status,
                stage=run.stage,
                total=run.total,
                completed=run.completed,
                failed=run.failed,
                message="current state",
            )
        )

    async def event_stream() -> AsyncIterator[str]:
        queue: asyncio.Queue[str] = asyncio.Queue()

        async def pump() -> None:
            async for event in channel.subscribe():
                await queue.put(f"event: progress\ndata: {event.model_dump_json()}\n\n")
            await queue.put("event: done\ndata: {}\n\n")

        task = asyncio.create_task(pump())
        try:
            while True:
                try:
                    frame = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_S)
                except TimeoutError:
                    # A comment frame. Ignored by EventSource, but it keeps
                    # intermediaries from closing a quiet connection.
                    yield ": keep-alive\n\n"
                    continue

                yield frame
                if frame.startswith("event: done"):
                    return
        finally:
            task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Nginx buffers responses by default, which defeats streaming entirely.
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/runs/{run_id}/shortlist", response_model=ShortlistOut)
async def get_shortlist(run_id: str, session: AsyncSession = SessionDep) -> ShortlistOut:
    """The ranked candidate list.

    Ordered by must-have compliance first, then score, matching
    :func:`hirelens.assess.pipeline.rank`. A 75 that misses a hard requirement is
    not ahead of a 62 that meets them all.
    """
    run = await RunRepository(session).get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    rows = await AssessmentRepository(session).shortlist(run_id)
    return ShortlistOut(
        run=RunOut.model_validate(run),
        entries=[ShortlistEntry.model_validate(row) for row in rows],
    )


@router.get("/assessments/{assessment_id}", response_model=AssessmentDetail)
async def get_assessment(
    assessment_id: str, session: AsyncSession = SessionDep
) -> AssessmentDetail:
    """One candidate in full, with citations resolved against the stored document.

    Citations are re-verified here rather than trusted from the saved payload, and
    each one carries the highlight rectangles for the original PDF. That means the
    grounding claim is checkable at read time, not only at the moment of scoring.
    """
    row = await AssessmentRepository(session).get(assessment_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Assessment {assessment_id} not found")

    document = await DocumentRepository(session).get(row.document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Source document is missing")

    payload = row.assessment_json
    blocks = _restore_blocks(document.blocks_json)

    requirements = [
        _requirement_out(item, document.text, blocks) for item in payload.get("assessments", [])
    ]

    return AssessmentDetail(
        id=row.id,
        run_id=row.run_id,
        document=DocumentOut.model_validate(document),
        candidate_label=row.candidate_label,
        score=row.score,
        score_low=row.score_low,
        score_high=row.score_high,
        band=row.band,
        meets_must_haves=row.meets_must_haves,
        mean_agreement=row.mean_agreement,
        grounding_rate=row.grounding_rate,
        citation_validity_rate=row.citation_validity_rate,
        requirements=requirements,
        risks=[
            RiskOut(**{k: risk[k] for k in ("code", "level", "message")})
            for risk in payload.get("risks", [])
        ],
        questions=[
            QuestionOut(
                question=question.get("question", ""),
                rationale=question.get("rationale", ""),
                targets=question.get("targets", ""),
            )
            for question in payload.get("questions", [])
        ],
    )


def _requirement_out(
    item: dict, document_text: str, blocks: list[TextBlock]
) -> RequirementResultOut:
    verdict = item.get("verdict", str(Verdict.NONE))
    weight = float(item.get("weight", 0.0))
    coefficient = VERDICT_VALUES.get(Verdict(verdict), 0.0)

    samples = item.get("samples", [])
    agreement = sum(1 for sample in samples if sample == verdict) / len(samples) if samples else 1.0
    values = [VERDICT_VALUES.get(Verdict(sample), 0.0) for sample in samples]
    spread = (max(values) - min(values)) if values else 0.0

    return RequirementResultOut(
        requirement_id=item.get("requirement_id", ""),
        requirement_text=item.get("requirement_text", ""),
        kind=item.get("kind", ""),
        weight=weight,
        verdict=verdict,
        points=round(weight * coefficient, 2),
        max_points=weight,
        agreement=round(agreement, 3),
        is_ambiguous=spread > 0.3,
        reasoning=item.get("reasoning", ""),
        citations=[
            _citation_out(citation, document_text, blocks) for citation in item.get("citations", [])
        ],
    )


def _citation_out(payload: dict, document_text: str, blocks: list[TextBlock]) -> CitationOut:
    span = Span(start=payload["span"]["start"], end=payload["span"]["end"])
    citation = Citation(
        document_id=payload.get("document_id", ""),
        span=span,
        quote=payload.get("quote", ""),
        page=payload.get("page"),
    )

    boxes = [
        HighlightBox(
            page=block.page,
            x0=block.bbox.x0,
            y0=block.bbox.y0,
            x1=block.bbox.x1,
            y1=block.bbox.y1,
        )
        for block in blocks
        if block.bbox is not None and block.span.overlaps(span)
    ]

    return CitationOut(
        start=span.start,
        end=span.end,
        page=citation.page,
        # Read the quote back from the document rather than echoing the stored
        # one, so what the UI highlights is what the document actually says.
        quote=document_text[span.start : span.end] if span.end <= len(document_text) else "",
        verified=citation.verify(document_text),
        boxes=boxes,
    )


def _restore_blocks(blocks_json: dict | None) -> list[TextBlock]:
    entries = (blocks_json or {}).get("blocks", [])
    restored: list[TextBlock] = []
    for entry in entries:
        bbox = entry.get("bbox")
        restored.append(
            TextBlock(
                span=Span(start=entry["span"]["start"], end=entry["span"]["end"]),
                page=entry.get("page", 1),
                bbox=BoundingBox(**bbox) if bbox else None,
                is_heading=entry.get("is_heading", False),
                font_size=entry.get("font_size"),
            )
        )
    return restored
