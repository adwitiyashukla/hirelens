"""Request and response models for the HTTP API.

Deliberately separate from the domain models in ``hirelens.schemas``. A response
model is a contract with a frontend and changes for presentation reasons; a domain
model changes for pipeline reasons. Returning ``CitedResume`` straight out of a
route would weld the two together, so that renaming an internal field becomes a
breaking API change.

The separation also lets responses carry things the domain has no opinion about,
like highlight rectangles resolved from the stored offset map.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


class JobCreate(BaseModel):
    description: str = Field(
        min_length=80,
        description="The full job posting. Short descriptions compile into poor rubrics.",
    )
    title: str = Field(default="", max_length=255)


class RequirementOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    requirement_id: str
    text: str
    kind: str
    category: str
    weight: float
    evidence_hint: str


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    description: str
    rubric_id: str | None = None
    created_at: datetime
    requirements: list[RequirementOut] = Field(default_factory=list)

    @property
    def is_compiled(self) -> bool:
        return bool(self.requirements)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    source_format: str
    page_count: int
    char_count: int
    created_at: datetime


class UploadResult(BaseModel):
    """One uploaded file.

    ``created`` is false when the exact bytes were already stored. Surfaced rather
    than hidden so a recruiter who uploads a duplicate is told, instead of quietly
    getting a second copy of the same candidate in the shortlist.
    """

    document: DocumentOut
    created: bool


class UploadResponse(BaseModel):
    uploaded: list[UploadResult]
    rejected: list[RejectedUpload] = Field(default_factory=list)

    @property
    def accepted_ids(self) -> list[str]:
        return [item.document.id for item in self.uploaded]


class RejectedUpload(BaseModel):
    model_config = ConfigDict(frozen=True)

    filename: str
    reason: str


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


class RunCreate(BaseModel):
    job_id: str
    document_ids: list[str] = Field(min_length=1)
    blind_mode: bool = True
    top_k: int = Field(default=4, ge=1, le=10)
    with_questions: bool = True


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    job_id: str
    status: str
    stage: str
    total: int
    completed: int
    failed: int
    blind_mode: bool
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def progress(self) -> float:
        return (self.completed + self.failed) / self.total if self.total else 0.0


class RunProgress(BaseModel):
    """One server-sent event during a run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    status: str
    stage: str
    total: int
    completed: int
    failed: int
    message: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.status in ("completed", "failed")


# ---------------------------------------------------------------------------
# Assessments
# ---------------------------------------------------------------------------


class ShortlistEntry(BaseModel):
    """A row in the ranked candidate table. Deliberately thin.

    The shortlist view renders dozens of these, so it carries only what the table
    shows. Per-requirement verdicts and citations arrive when a candidate is
    opened, from the detail endpoint.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    candidate_label: str
    score: float
    score_low: float
    score_high: float
    band: str
    meets_must_haves: bool
    mean_agreement: float
    grounding_rate: float
    citation_validity_rate: float
    elapsed_s: float


class ShortlistOut(BaseModel):
    run: RunOut
    entries: list[ShortlistEntry]


class HighlightBox(BaseModel):
    """A rectangle to draw over the rendered PDF."""

    model_config = ConfigDict(frozen=True)

    page: int
    x0: float
    y0: float
    x1: float
    y1: float


class CitationOut(BaseModel):
    """A citation, resolved against the stored document.

    ``quote`` is re-read from the stored text rather than trusted from the saved
    payload, and ``verified`` records whether it still checks out. That makes the
    grounding claim re-checkable at read time, not only at write time.
    """

    model_config = ConfigDict(frozen=True)

    start: int
    end: int
    page: int | None = None
    quote: str
    verified: bool = True
    boxes: list[HighlightBox] = Field(default_factory=list)


class RequirementResultOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    requirement_id: str
    requirement_text: str
    kind: str
    weight: float
    verdict: str
    points: float
    max_points: float
    agreement: float
    is_ambiguous: bool
    reasoning: str
    citations: list[CitationOut] = Field(default_factory=list)


class RiskOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    level: str
    message: str


class QuestionOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    rationale: str
    targets: str = ""


class AssessmentDetail(BaseModel):
    """Everything the candidate detail view needs, in one request."""

    id: str
    run_id: str
    document: DocumentOut
    candidate_label: str
    score: float
    score_low: float
    score_high: float
    band: str
    meets_must_haves: bool
    mean_agreement: float
    grounding_rate: float
    citation_validity_rate: float
    requirements: list[RequirementResultOut]
    risks: list[RiskOut] = Field(default_factory=list)
    questions: list[QuestionOut] = Field(default_factory=list)


class DocumentText(BaseModel):
    """The document text plus its offset map, for rendering highlights."""

    document_id: str
    filename: str
    page_count: int
    text: str
    blocks: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthOut(BaseModel):
    status: str
    version: str
    database: bool
    provider: str
    model: str
    provider_configured: bool
    blind_mode: bool


class ErrorOut(BaseModel):
    detail: str
    code: str = "error"
