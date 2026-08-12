from __future__ import annotations

import re
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T")

_FUZZY_MATCH_THRESHOLD = 0.85


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _similarity(a: str, b: str) -> float:
    ta, tb = set(_normalise(a).split()), set(_normalise(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class Span(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: int = Field(ge=0, description="Inclusive start offset in the document text")
    end: int = Field(gt=0, description="Exclusive end offset in the document text")

    @model_validator(mode="after")
    def _check_order(self) -> Span:
        if self.end <= self.start:
            raise ValueError(f"span end ({self.end}) must be greater than start ({self.start})")
        return self

    def __len__(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end

    def slice_of(self, text: str) -> str:
        return text[self.start : self.end]


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str = Field(description="Which source document this points into")
    span: Span
    quote: str = Field(
        min_length=1,
        description="The text the model claims lives at this span. Verified, not trusted.",
    )
    page: int | None = Field(
        default=None, description="1-indexed PDF page, when the source was paginated"
    )

    def verify(self, document_text: str, *, threshold: float = _FUZZY_MATCH_THRESHOLD) -> bool:
        if self.span.end > len(document_text):
            return False
        actual = self.span.slice_of(document_text)
        if _normalise(actual) == _normalise(self.quote):
            return True
        return _similarity(actual, self.quote) >= threshold

    def resolved_quote(self, document_text: str) -> str:
        return self.span.slice_of(document_text)


class Cited(BaseModel, Generic[T]):
    value: T
    citations: list[Citation] = Field(
        default_factory=list,
        description="Supporting spans. Empty means the value was inferred, not read.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Model-reported confidence. Advisory only, never used for gating.",
    )

    @property
    def is_grounded(self) -> bool:
        return len(self.citations) > 0

    def verify(self, document_text: str) -> VerificationResult:
        if not self.citations:
            return VerificationResult(total=0, valid=0, invalid_quotes=[])
        invalid = [c.quote for c in self.citations if not c.verify(document_text)]
        return VerificationResult(
            total=len(self.citations),
            valid=len(self.citations) - len(invalid),
            invalid_quotes=invalid,
        )

    @classmethod
    def inferred(cls, value: T, *, confidence: float = 0.5) -> Cited[T]:
        return cls(value=value, citations=[], confidence=confidence)


class VerificationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    total: int = Field(ge=0)
    valid: int = Field(ge=0)
    invalid_quotes: list[str] = Field(default_factory=list)

    @property
    def rate(self) -> float:
        return 1.0 if self.total == 0 else self.valid / self.total

    @property
    def ok(self) -> bool:
        return not self.invalid_quotes

    def __add__(self, other: VerificationResult) -> VerificationResult:
        return VerificationResult(
            total=self.total + other.total,
            valid=self.valid + other.valid,
            invalid_quotes=[*self.invalid_quotes, *other.invalid_quotes],
        )


class EvidenceUnit(BaseModel):
    model_config = ConfigDict(frozen=True)

    unit_id: str
    document_id: str
    text: str = Field(min_length=1, description="Searchable text: the claim plus its context")
    span: Span
    quote: str = Field(
        default="",
        description="Exact text at `span`. Falls back to `text` when no context was added.",
    )
    section: str = Field(
        default="unknown",
        description="Which resume section this came from: work, projects, skills, ...",
    )
    page: int | None = None

    @property
    def claim(self) -> str:
        return self.quote or self.text

    def as_citation(self) -> Citation:
        return Citation(
            document_id=self.document_id,
            span=self.span,
            quote=self.claim,
            page=self.page,
        )
