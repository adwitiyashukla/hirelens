"""Data access, kept out of the route handlers.

Routes translate HTTP to intent; repositories translate intent to SQL. Keeping
them apart is what lets the background runner reuse the same persistence logic
without importing anything from FastAPI, and it means the screening pipeline can
be tested without a web server anywhere in the picture.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from hirelens.api.db.models import Assessment, Document, JobPosting, RunStatus, ScreeningRun
from hirelens.ingest.document import SourceDocument


def new_id() -> str:
    return uuid.uuid4().hex


def job_id_for(description: str) -> str:
    """Content-addressed job id.

    Posting the same description twice reuses the row, which means the compiled
    rubric is reused too. That is the same idempotence property documents have,
    and it matters more here: recompiling would produce a subtly different rubric
    and silently make two runs incomparable.
    """
    return hashlib.sha256(description.strip().encode("utf-8")).hexdigest()[:32]


class JobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, job_id: str) -> JobPosting | None:
        return await self.session.get(JobPosting, job_id)

    async def upsert(self, description: str, *, title: str = "") -> JobPosting:
        """Create the posting, or return the existing one for identical text."""
        job_id = job_id_for(description)
        existing = await self.get(job_id)
        if existing is not None:
            return existing

        posting = JobPosting(id=job_id, title=title, description=description.strip())
        self.session.add(posting)
        await self.session.flush()
        return posting

    async def attach_rubric(self, job: JobPosting, rubric: Any) -> JobPosting:
        job.rubric_id = rubric.rubric_id
        job.rubric_json = rubric.model_dump(mode="json")
        await self.session.flush()
        return job

    async def list(self, *, limit: int = 50, offset: int = 0) -> list[JobPosting]:
        result = await self.session.execute(
            select(JobPosting).order_by(JobPosting.created_at.desc()).limit(limit).offset(offset)
        )
        return list(result.scalars())

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(JobPosting))
        return int(result.scalar_one())


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, document_id: str) -> Document | None:
        return await self.session.get(Document, document_id)

    async def upsert(
        self,
        source: SourceDocument,
        *,
        raw_bytes: bytes | None = None,
        content_type: str = "application/pdf",
    ) -> tuple[Document, bool]:
        """Store an ingested document. Returns (row, created).

        Idempotent because the id is the hash of the file bytes: re-uploading the
        same resume returns the existing row and its cached extraction rather than
        paying for it again.
        """
        existing = await self.get(source.document_id)
        if existing is not None:
            return existing, False

        document = Document(
            id=source.document_id,
            filename=source.filename,
            content_type=content_type,
            source_format=str(source.source_format),
            text=source.text,
            raw_bytes=raw_bytes,
            page_count=source.page_count,
            char_count=source.char_count,
            # The offset map, so highlights can be drawn long after the run.
            blocks_json={"blocks": [block.model_dump(mode="json") for block in source.blocks]},
        )
        self.session.add(document)
        await self.session.flush()
        return document, True

    async def list(self, *, limit: int = 50, offset: int = 0) -> list[Document]:
        result = await self.session.execute(
            select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
        )
        return list(result.scalars())

    async def get_many(self, document_ids: list[str]) -> list[Document]:
        if not document_ids:
            return []
        result = await self.session.execute(select(Document).where(Document.id.in_(document_ids)))
        found = {d.id: d for d in result.scalars()}
        # Preserve the caller's order so results line up with the request.
        return [found[i] for i in document_ids if i in found]


class RunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, run_id: str) -> ScreeningRun | None:
        return await self.session.get(ScreeningRun, run_id)

    async def create(self, *, job_id: str, total: int, blind_mode: bool = True) -> ScreeningRun:
        run = ScreeningRun(
            id=new_id(),
            job_id=job_id,
            total=total,
            status=str(RunStatus.PENDING),
            stage="queued",
            blind_mode=blind_mode,
        )
        self.session.add(run)
        await self.session.flush()
        return run

    async def mark_running(self, run: ScreeningRun, stage: str = "compiling rubric") -> None:
        run.status = str(RunStatus.RUNNING)
        run.stage = stage
        run.started_at = datetime.now(timezone.utc)
        await self.session.flush()

    async def set_stage(self, run: ScreeningRun, stage: str) -> None:
        run.stage = stage
        await self.session.flush()

    async def record_progress(
        self, run: ScreeningRun, *, completed: int = 0, failed: int = 0
    ) -> None:
        run.completed += completed
        run.failed += failed
        await self.session.flush()

    async def finish(self, run: ScreeningRun, *, error: str | None = None) -> None:
        run.status = str(RunStatus.FAILED if error else RunStatus.COMPLETED)
        run.stage = "failed" if error else "done"
        run.error = error
        run.finished_at = datetime.now(timezone.utc)
        await self.session.flush()

    async def list_for_job(self, job_id: str, *, limit: int = 20) -> list[ScreeningRun]:
        result = await self.session.execute(
            select(ScreeningRun)
            .where(ScreeningRun.job_id == job_id)
            .order_by(ScreeningRun.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars())


class AssessmentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, assessment_id: str) -> Assessment | None:
        return await self.session.get(Assessment, assessment_id)

    async def create(
        self,
        *,
        run_id: str,
        document_id: str,
        assessment: Any,
        resume: Any | None = None,
        elapsed_s: float = 0.0,
    ) -> Assessment:
        low, high = assessment.score_range
        row = Assessment(
            id=new_id(),
            run_id=run_id,
            document_id=document_id,
            candidate_label=assessment.candidate_label,
            score=assessment.score,
            score_low=low,
            score_high=high,
            band=assessment.band,
            meets_must_haves=assessment.meets_all_must_haves,
            mean_agreement=assessment.mean_agreement,
            grounding_rate=assessment.grounding_rate,
            citation_validity_rate=assessment.citation_validity_rate,
            elapsed_s=elapsed_s,
            assessment_json=assessment.model_dump(mode="json"),
            resume_json=resume.model_dump(mode="json") if resume is not None else None,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def shortlist(self, run_id: str) -> list[Assessment]:
        """Every candidate in a run, in shortlist order.

        Must-have compliance sorts ahead of score, matching
        :func:`hirelens.assess.pipeline.rank`. A 75 that misses a hard requirement
        is not better than a 62 that meets them all, and the ordering has to say so
        in SQL as well as in Python or the API and the CLI would disagree.
        """
        result = await self.session.execute(
            select(Assessment)
            .where(Assessment.run_id == run_id)
            .order_by(Assessment.meets_must_haves.desc(), Assessment.score.desc())
        )
        return list(result.scalars())

    async def for_document(self, document_id: str, *, limit: int = 20) -> list[Assessment]:
        result = await self.session.execute(
            select(Assessment)
            .where(Assessment.document_id == document_id)
            .order_by(Assessment.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars())
