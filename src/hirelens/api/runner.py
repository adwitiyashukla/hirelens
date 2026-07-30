"""Background screening execution with live progress.

Screening a batch takes tens of seconds per candidate, so it cannot happen on the
request path. The API accepts the run, returns immediately with an id, and the work
proceeds in the background while the client subscribes to progress over
server-sent events.

**Why an in-process runner rather than Celery or arq.** Those need a broker, a
worker process, and a deployment story, and this workload does not justify any of
it: one screening is a handful of network calls with almost no CPU, the concurrency
limit that matters is the LLM provider's rate limit (already enforced by the
client's semaphore), and a free-tier demo cannot afford to run Redis. The trade-off
is real and stated rather than hidden: **a restart loses in-flight runs.** The run
row is left in ``running`` and is recoverable by re-submitting. The
:class:`ScreeningRunner` interface is deliberately narrow so swapping in a real
queue later touches one class.

**Progress is published, not polled.** Each run gets an asyncio broadcast channel;
subscribers receive every stage change. A late subscriber immediately receives the
current state, so a client that connects after the run finished still gets a
terminal event and closes cleanly rather than hanging forever.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hirelens.api.db.models import RunStatus, ScreeningRun
from hirelens.api.db.repository import (
    AssessmentRepository,
    DocumentRepository,
    JobRepository,
    RunRepository,
)
from hirelens.api.db.session import session_scope
from hirelens.api.schemas import RunProgress
from hirelens.assess.pipeline import ScreeningPipeline
from hirelens.config import Settings, get_settings
from hirelens.ingest.document import (
    BoundingBox,
    SourceDocument,
    SourceFormat,
    TextBlock,
)
from hirelens.schemas.evidence import Span
from hirelens.schemas.job import Rubric

logger = logging.getLogger(__name__)

#: How long a finished run's progress channel is kept so late subscribers still
#: get a terminal event instead of an empty stream.
_CHANNEL_TTL_S = 300


class ProgressChannel:
    """Fan-out of progress events for one run.

    Keeps the latest event so a subscriber joining mid-run, or after the end,
    immediately learns where things stand rather than waiting for the next change
    that may never come.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._subscribers: set[asyncio.Queue[RunProgress]] = set()
        self._latest: RunProgress | None = None
        self._closed = False

    @property
    def latest(self) -> RunProgress | None:
        return self._latest

    @property
    def closed(self) -> bool:
        return self._closed

    def publish(self, event: RunProgress) -> None:
        self._latest = event
        for queue in list(self._subscribers):
            # Never block the pipeline on a slow reader. A full queue means the
            # client cannot keep up, and dropping an intermediate event is
            # harmless because the next one carries the complete state anyway.
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(event)
        if event.is_terminal:
            self._closed = True

    async def subscribe(self) -> AsyncIterator[RunProgress]:
        queue: asyncio.Queue[RunProgress] = asyncio.Queue(maxsize=64)
        self._subscribers.add(queue)
        try:
            if self._latest is not None:
                yield self._latest
                if self._latest.is_terminal:
                    return
            while True:
                event = await queue.get()
                yield event
                if event.is_terminal:
                    return
        finally:
            self._subscribers.discard(queue)


class ScreeningRunner:
    """Executes screening runs off the request path."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        settings: Settings | None = None,
        pipeline_factory=None,
    ) -> None:
        self.session_factory = session_factory
        self.settings = settings or get_settings()
        # Injectable so tests can supply a pipeline backed by a fake provider.
        self._pipeline_factory = pipeline_factory or (
            lambda: ScreeningPipeline(settings=self.settings)
        )
        self._channels: dict[str, ProgressChannel] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    # -- channels ------------------------------------------------------------

    def channel(self, run_id: str) -> ProgressChannel:
        channel = self._channels.get(run_id)
        if channel is None:
            channel = ProgressChannel(run_id)
            self._channels[run_id] = channel
        return channel

    def _publish(self, run: ScreeningRun, message: str = "") -> None:
        self.channel(run.id).publish(
            RunProgress(
                run_id=run.id,
                status=run.status,
                stage=run.stage,
                total=run.total,
                completed=run.completed,
                failed=run.failed,
                message=message,
            )
        )

    # -- submission ----------------------------------------------------------

    def submit(self, run_id: str, document_ids: list[str], **options) -> None:
        """Queue a run. Returns immediately."""
        if run_id in self._tasks and not self._tasks[run_id].done():
            logger.warning("run %s is already executing", run_id)
            return

        task = asyncio.create_task(self._execute(run_id, document_ids, **options))
        self._tasks[run_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(run_id, None))

    async def wait(self, run_id: str, *, timeout: float = 120.0) -> None:
        """Block until a run finishes. For tests and for CLI-style callers."""
        task = self._tasks.get(run_id)
        if task is not None:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)

    async def shutdown(self) -> None:
        """Cancel in-flight runs on server shutdown.

        Their rows stay in ``running``, which is the honest record of what
        happened. Marking them failed would be wrong: they were interrupted, not
        rejected, and resubmitting is cheap because extraction is cached.
        """
        for task in list(self._tasks.values()):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()

    # -- execution -----------------------------------------------------------

    async def _execute(
        self,
        run_id: str,
        document_ids: list[str],
        *,
        top_k: int = 4,
        with_questions: bool = True,
    ) -> None:
        pipeline = self._pipeline_factory()

        try:
            async with session_scope(self.session_factory) as session:
                run = await RunRepository(session).get(run_id)
                if run is None:
                    logger.error("run %s vanished before execution", run_id)
                    return
                await RunRepository(session).mark_running(run)
                self._publish(run)
                blind_mode = run.blind_mode
                job_id = run.job_id

            rubric = await self._resolve_rubric(job_id, pipeline, run_id)

            for position, document_id in enumerate(document_ids, start=1):
                await self._screen_one(
                    run_id=run_id,
                    document_id=document_id,
                    rubric=rubric,
                    pipeline=pipeline,
                    position=position,
                    total=len(document_ids),
                    blind_mode=blind_mode,
                    top_k=top_k,
                    with_questions=with_questions,
                )

            async with session_scope(self.session_factory) as session:
                run = await RunRepository(session).get(run_id)
                if run is not None:
                    await RunRepository(session).finish(run)
                    self._publish(run, "run complete")

        except asyncio.CancelledError:
            logger.info("run %s cancelled", run_id)
            raise
        except Exception as exc:
            logger.exception("run %s failed", run_id)
            async with session_scope(self.session_factory) as session:
                run = await RunRepository(session).get(run_id)
                if run is not None:
                    await RunRepository(session).finish(run, error=str(exc)[:500])
                    self._publish(run, f"run failed: {exc}")
        finally:
            await pipeline.aclose()

    async def _resolve_rubric(
        self, job_id: str, pipeline: ScreeningPipeline, run_id: str
    ) -> Rubric:
        """Reuse the stored rubric, or compile and store it once.

        Reuse is a correctness requirement, not a saving. Two candidates screened
        against "the same job" must face literally the same requirements, and
        recompiling would let the wording drift between them.
        """
        async with session_scope(self.session_factory) as session:
            job = await JobRepository(session).get(job_id)
            if job is None:
                raise ValueError(f"Job {job_id} not found")

            if job.rubric_json is not None:
                return Rubric.model_validate(job.rubric_json)

            run = await RunRepository(session).get(run_id)
            if run is not None:
                await RunRepository(session).set_stage(run, "compiling rubric")
                self._publish(run, "compiling the job description into a rubric")

            rubric = await pipeline.compile_rubric(job.description)
            await JobRepository(session).attach_rubric(job, rubric)
            return rubric

    async def _screen_one(
        self,
        *,
        run_id: str,
        document_id: str,
        rubric: Rubric,
        pipeline: ScreeningPipeline,
        position: int,
        total: int,
        blind_mode: bool,
        top_k: int,
        with_questions: bool,
    ) -> None:
        """Screen one candidate. A failure costs that candidate, not the batch."""
        async with session_scope(self.session_factory) as session:
            run = await RunRepository(session).get(run_id)
            document = await DocumentRepository(session).get(document_id)
            if run is None or document is None:
                logger.warning("skipping missing document %s in run %s", document_id, run_id)
                return

            await RunRepository(session).set_stage(run, f"screening {position} of {total}")
            self._publish(run, f"screening candidate {position} of {total}")
            source = _to_source_document(document)

        try:
            result = await pipeline.screen(
                source,
                rubric,
                top_k=top_k,
                with_questions=with_questions,
                blind=blind_mode,
            )
        except Exception as exc:
            logger.exception("screening failed for %s", document_id)
            async with session_scope(self.session_factory) as session:
                run = await RunRepository(session).get(run_id)
                if run is not None:
                    await RunRepository(session).record_progress(run, failed=1)
                    self._publish(run, f"candidate {position} failed: {exc}")
            return

        async with session_scope(self.session_factory) as session:
            await AssessmentRepository(session).create(
                run_id=run_id,
                document_id=document_id,
                assessment=result.assessment,
                resume=result.resume,
                elapsed_s=result.elapsed_s,
            )
            run = await RunRepository(session).get(run_id)
            if run is not None:
                await RunRepository(session).record_progress(run, completed=1)
                self._publish(run, f"candidate {position} scored {result.assessment.score:.0f}")


def _to_source_document(row) -> SourceDocument:
    """Rebuild the in-memory document from its stored row.

    The offset map is restored too. Without it, citations resolved during this run
    would have no bounding boxes and the frontend could not draw a highlight over
    the original PDF.
    """
    payload = (row.blocks_json or {}).get("blocks", [])
    blocks: list[TextBlock] = []

    for entry in payload:
        bbox = entry.get("bbox")
        blocks.append(
            TextBlock(
                span=Span(start=entry["span"]["start"], end=entry["span"]["end"]),
                page=entry.get("page", 1),
                bbox=BoundingBox(**bbox) if bbox else None,
                is_heading=entry.get("is_heading", False),
                font_size=entry.get("font_size"),
            )
        )

    return SourceDocument(
        document_id=row.id,
        filename=row.filename,
        source_format=SourceFormat(row.source_format),
        text=row.text,
        blocks=blocks,
        page_count=row.page_count,
    )


__all__ = ["ProgressChannel", "RunStatus", "ScreeningRunner"]
