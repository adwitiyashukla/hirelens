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
    type_annotation_map: ClassVar[dict] = {dict[str, Any]: JSON}


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobPosting(Base):
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
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(64), default="application/pdf")
    source_format: Mapped[str] = mapped_column(String(16), default="pdf")

    text: Mapped[str] = mapped_column(Text)
    raw_bytes: Mapped[bytes | None] = mapped_column(LargeBinary, default=None)

    page_count: Mapped[int] = mapped_column(Integer, default=1)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    blocks_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    assessments: Mapped[list[Assessment]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class ScreeningRun(Base):
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
    __tablename__ = "assessments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("screening_runs.id", ondelete="CASCADE"))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))

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

    assessment_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    resume_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run: Mapped[ScreeningRun] = relationship(back_populates="assessments")
    document: Mapped[Document] = relationship(back_populates="assessments")


Index(
    "ix_assessments_shortlist",
    Assessment.run_id,
    Assessment.meets_must_haves.desc(),
    Assessment.score.desc(),
)
Index("ix_runs_job_created", ScreeningRun.job_id, ScreeningRun.created_at.desc())
