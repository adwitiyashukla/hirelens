"""Evidence primitives: spans, citations, and the ``Cited[T]`` wrapper.

This module is the load-bearing idea of the whole project.

A generic resume screener asks a model for a score and prints the number. There is
no way to check the number, because there is no link between the output and the
input. Here, every value the model produces is wrapped in :class:`Cited`, which
carries the character range in the source document that the value came from. A
value with no citation does not type-check, and a citation whose offsets do not
actually contain the quoted text fails :meth:`Citation.verify`.

The practical effect is that hallucination stops being a silent correctness
problem and becomes a loud validation error we can count, report, and regress
against in CI. The "citation validity rate" metric in the README is computed
directly from this machinery.
"""

from __future__ import annotations

import re
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T")

# How much slack we allow between the quoted text and the text actually found at
# the offsets. Models routinely normalise whitespace or drop a trailing period,
# and failing those would be pedantry rather than hallucination detection.
_FUZZY_MATCH_THRESHOLD = 0.85


def _normalise(text: str) -> str:
    """Collapse whitespace and case so that cosmetic differences do not fail a match."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity. Cheap, dependency-free, good enough here.

    We are not trying to measure semantic similarity, only to tolerate the
    whitespace and punctuation noise that models introduce when they echo a
    quote back. Anything genuinely invented will score far below the threshold.
    """
    ta, tb = set(_normalise(a).split()), set(_normalise(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class Span(BaseModel):
    """A half-open character range ``[start, end)`` in a source document."""

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
    """A pointer from a produced value back to the text that justifies it."""

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
        """Return True if the span really does contain something like :attr:`quote`.

        This is the hallucination check. A model that invents an achievement will
        either give offsets that point at unrelated text (low similarity) or
        offsets outside the document (caught by the bounds check).
        """
        if self.span.end > len(document_text):
            return False
        actual = self.span.slice_of(document_text)
        if _normalise(actual) == _normalise(self.quote):
            return True
        return _similarity(actual, self.quote) >= threshold

    def resolved_quote(self, document_text: str) -> str:
        """The authoritative text, taken from the document rather than the model."""
        return self.span.slice_of(document_text)


class Cited(BaseModel, Generic[T]):
    """A value together with the evidence that supports it.

    Usage in a schema looks like::

        class WorkExperience(BaseModel):
            company: Cited[str]
            start_date: Cited[str] | None
            highlights: list[Cited[str]]

    which makes it structurally impossible to record a company name without also
    recording where in the resume that name appeared.
    """

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
        """True when at least one citation backs this value."""
        return len(self.citations) > 0

    def verify(self, document_text: str) -> VerificationResult:
        """Check every citation against the source document."""
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
        """Construct an explicitly ungrounded value.

        Sometimes a field genuinely is an inference rather than a quote, for
        example normalising "Jan 2023 - present" into an end date of ``None``.
        Making that construction explicit keeps the grounding statistics honest:
        we can tell "no evidence was found" apart from "nobody bothered to look".
        """
        return cls(value=value, citations=[], confidence=confidence)


class VerificationResult(BaseModel):
    """Outcome of checking a set of citations against the source."""

    model_config = ConfigDict(frozen=True)

    total: int = Field(ge=0)
    valid: int = Field(ge=0)
    invalid_quotes: list[str] = Field(default_factory=list)

    @property
    def rate(self) -> float:
        """Fraction of citations that survived verification. 1.0 when there are none."""
        return 1.0 if self.total == 0 else self.valid / self.total

    @property
    def ok(self) -> bool:
        return not self.invalid_quotes

    def __add__(self, other: VerificationResult) -> VerificationResult:
        """Combine results so callers can fold over a whole document."""
        return VerificationResult(
            total=self.total + other.total,
            valid=self.valid + other.valid,
            invalid_quotes=[*self.invalid_quotes, *other.invalid_quotes],
        )


class EvidenceUnit(BaseModel):
    """One retrievable chunk of a resume.

    The retrieval layer works over these rather than over raw text, because a
    requirement like "has production Kubernetes experience" should match a single
    bullet point, not a whole page. Keeping the span means a retrieval hit can be
    turned straight back into a highlight in the UI.

    ``text`` and ``quote`` are deliberately different things and conflating them is
    a real bug that this docstring exists to prevent recurring.

    * ``text`` is what the retriever searches. It includes parent context, so a
      bullet reading "Reduced p99 latency to 180ms" is indexed as "Backend
      Engineer at Fintech Co. Reduced p99 latency to 180ms" and can be found by a
      query about backend performance work.
    * ``quote`` is the exact text living at ``span``, and nothing else. It is what
      a citation asserts and what gets verified against the document.

    Using ``text`` as the quote would attach a citation whose span does not
    contain it, which fails verification and would quietly drop the evidence for a
    score that was, in fact, correctly reasoned.
    """

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
        """The text this unit actually points at, without retrieval context."""
        return self.quote or self.text

    def as_citation(self) -> Citation:
        """Turn a retrieved unit into a citation the judge can attach to a score."""
        return Citation(
            document_id=self.document_id,
            span=self.span,
            quote=self.claim,
            page=self.page,
        )
