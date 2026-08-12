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

_CHANNEL_TTL_S = 300


class ProgressChannel:
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
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        settings: Settings | None = None,
        pipeline_factory=None,
    ) -> None:
        self.session_factory = session_factory
        self.settings = settings or get_settings()
        self._pipeline_factory = pipeline_factory or (
            lambda: ScreeningPipeline(settings=self.settings)
        )
        self._channels: dict[str, ProgressChannel] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

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

    def submit(self, run_id: str, document_ids: list[str], **options) -> None:
        if run_id in self._tasks and not self._tasks[run_id].done():
            logger.warning("run %s is already executing", run_id)
            return

        task = asyncio.create_task(self._execute(run_id, document_ids, **options))
        self._tasks[run_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(run_id, None))

    async def wait(self, run_id: str, *, timeout: float = 120.0) -> None:
        task = self._tasks.get(run_id)
        if task is not None:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)

    async def shutdown(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()

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
