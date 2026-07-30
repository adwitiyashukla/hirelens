"""Database schema.

Four tables, and the shape of them follows from decisions made much earlier in the
pipeline rather than from ORM convention.

**Documents are content-addressed.** The primary key is the SHA-256 of the file
bytes, which the ingestion layer already computes. Uploading the same resume twice
is therefore idempotent for free: the second upload resolves to the existing row,
reuses its cached extraction, and costs nothing. Recruiters really do upload the
same CV twice.

**The full text is stored, not just the path.** Every citation in the system is a
character offset into that exact string, so if the text is not stored the
highlights cannot be rendered later. Storing the original bytes as well means the
frontend can display the real PDF with boxes drawn over it.

**Assessments store their JSON payload whole.** The relational columns are the ones
worth querying (score, rubric, whether must-haves were met); the nested structure
of per-requirement verdicts, samples and citations is stored as JSON. Normalising
that into four more tables would buy nothing, because nothing queries across
requirements, and it would couple the database schema to a Pydantic model that is
still moving.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, ClassVar

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from hirelens._compat import StrEnum


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base with a JSON type that works on SQLite and Postgres alike."""

    # Maps the `dict[str, Any]` annotation to a JSON column that works on both
    # SQLite and Postgres without a backend-specific type.
    type_annotation_map: ClassVar[dict] = {dict[str, Any]: JSON}


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobPosting(Base):
    """A job description and the rubric compiled from it.

    The rubric is cached on the row rather than recompiled per run. Two candidates
    screened against "the same job" must be scored against literally the same
    requirements or their scores are not comparable, and recompiling would let the
    wording drift between them.
    """

    __tablename__ = "job_postings"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(String(255), default="")
    description: Mapped[str] = mapped_column(Text)
    rubric_id: Mapped[str | None] = mapped_column(String(32), default=None)
    rubric_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    runs: Mapped[list[ScreeningRun]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )

    @property
    def is_compiled(self) -> bool:
        return self.rubric_json is not None


class Document(Base):
    """An uploaded resume, keyed by the hash of its bytes."""

    __tablename__ = "documents"

    # Content-addressed, so re-uploading the same file is a no-op.
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(64), default="application/pdf")
    source_format: Mapped[str] = mapped_column(String(16), default="pdf")

    # The canonical text every span indexes into. Without it, citations resolved
    # at screening time could never be rendered again.
    text: Mapped[str] = mapped_column(Text)
    raw_bytes: Mapped[bytes | None] = mapped_column(LargeBinary, default=None)

    page_count: Mapped[int] = mapped_column(Integer, default=1)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    # Offset map: page and bounding box per line, for PDF highlight overlays.
    blocks_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    assessments: Mapped[list[Assessment]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class ScreeningRun(Base):
    """One batch of candidates screened against one job."""

    __tablename__ = "screening_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("job_postings.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(16), default=str(RunStatus.PENDING))

    total: Mapped[int] = mapped_column(Integer, default=0)
    completed: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)

    stage: Mapped[str] = mapped_column(String(64), default="queued")
    error: Mapped[str | None] = mapped_column(Text, default=None)
    blind_mode: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    job: Mapped[JobPosting] = relationship(back_populates="runs")
    assessments: Mapped[list[Assessment]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )

    @property
    def is_terminal(self) -> bool:
        return self.status in (str(RunStatus.COMPLETED), str(RunStatus.FAILED))

    @property
    def progress(self) -> float:
        return (self.completed + self.failed) / self.total if self.total else 0.0


class Assessment(Base):
    """One candidate's result within one run."""

    __tablename__ = "assessments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("screening_runs.id", ondelete="CASCADE"))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))

    # Queryable columns: everything a shortlist view sorts or filters on.
    candidate_label: Mapped[str] = mapped_column(String(255), default="")
    score: Mapped[float] = mapped_column(Float, default=0.0)
    score_low: Mapped[float] = mapped_column(Float, default=0.0)
    score_high: Mapped[float] = mapped_column(Float, default=0.0)
    band: Mapped[str] = mapped_column(String(32), default="")
    meets_must_haves: Mapped[bool] = mapped_column(Boolean, default=True)
    mean_agreement: Mapped[float] = mapped_column(Float, default=1.0)
    grounding_rate: Mapped[float] = mapped_column(Float, default=1.0)
    citation_validity_rate: Mapped[float] = mapped_column(Float, default=1.0)
    elapsed_s: Mapped[float] = mapped_column(Float, default=0.0)

    # The nested detail. Nothing queries across requirements, so normalising this
    # would add tables and joins for no benefit.
    assessment_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    resume_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run: Mapped[ScreeningRun] = relationship(back_populates="assessments")
    document: Mapped[Document] = relationship(back_populates="assessments")


# The shortlist query is the hot path: every candidate in a run, ordered by
# must-have compliance then score. This index matches it exactly.
Index(
    "ix_assessments_shortlist",
    Assessment.run_id,
    Assessment.meets_must_haves.desc(),
    Assessment.score.desc(),
)
Index("ix_runs_job_created", ScreeningRun.job_id, ScreeningRun.created_at.desc())
