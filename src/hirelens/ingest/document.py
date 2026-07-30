"""The :class:`SourceDocument` type and the offset bookkeeping behind it.

Everything downstream cites into a document by character offset, so the one job
this module has is to produce a single canonical text string and never lose track
of where each piece of it came from. Concretely, we keep a table of
:class:`TextBlock` entries mapping character ranges back to (page, bounding box),
which is what lets the frontend draw a highlight rectangle over the original PDF
rather than showing a plain-text approximation of it.

If you change how text is joined here, citations everywhere shift. Treat the
offsets as an API.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from hirelens._compat import StrEnum
from hirelens.schemas.evidence import Span


class SourceFormat(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    TEXT = "text"


class BoundingBox(BaseModel):
    """A rectangle on a page, in PDF points, origin at the top left."""

    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def union(self, other: BoundingBox) -> BoundingBox:
        return BoundingBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )


class TextBlock(BaseModel):
    """A run of text whose position on the page we know.

    One of these per visual line. Lines are the right granularity: word-level
    would bloat the map for no benefit, paragraph-level would make highlights
    too coarse to be convincing in the UI.
    """

    model_config = ConfigDict(frozen=True)

    span: Span
    page: int = Field(ge=1, description="1-indexed page number")
    bbox: BoundingBox | None = None
    is_heading: bool = False
    font_size: float | None = None


class SourceDocument(BaseModel):
    """A resume (or any input document) plus its offset map.

    ``text`` is the canonical string that every :class:`~hirelens.schemas.evidence.Span`
    in the system indexes into. It is built once, at ingestion, and never
    rewritten. Redaction produces spans to *mask at render time* rather than
    editing this string, precisely so that offsets stay stable.
    """

    model_config = ConfigDict(frozen=True)

    document_id: str
    filename: str
    source_format: SourceFormat
    text: str
    blocks: list[TextBlock] = Field(default_factory=list)
    page_count: int = Field(default=1, ge=1)
    metadata: dict[str, str] = Field(default_factory=dict)

    # -- construction --------------------------------------------------------

    @staticmethod
    def make_id(payload: bytes) -> str:
        """Content-addressed id.

        Hashing the bytes rather than using a uuid means re-uploading the same
        resume reuses the same id, which in turn means the LLM response cache
        hits and the eval harness is reproducible across runs.
        """
        return hashlib.sha256(payload).hexdigest()[:16]

    # -- lookups -------------------------------------------------------------

    def slice(self, span: Span) -> str:
        return span.slice_of(self.text)

    def page_of(self, span: Span) -> int | None:
        """Which page a span starts on, or None if it falls outside every block."""
        for block in self.blocks:
            if block.span.start <= span.start < block.span.end:
                return block.page
        return None

    def blocks_for(self, span: Span) -> list[TextBlock]:
        """Every block the span touches, in document order."""
        return [b for b in self.blocks if b.span.overlaps(span)]

    def highlight_boxes(self, span: Span) -> list[tuple[int, BoundingBox]]:
        """(page, box) pairs to draw for a citation.

        A span that crosses a line break produces several boxes, which is why the
        return type is a list. The frontend draws each one.
        """
        return [(b.page, b.bbox) for b in self.blocks_for(span) if b.bbox is not None]

    def line_at(self, offset: int) -> str:
        """The full visual line containing ``offset``. Handy for debugging citations."""
        start = self.text.rfind("\n", 0, offset) + 1
        end = self.text.find("\n", offset)
        return self.text[start : end if end != -1 else len(self.text)]

    # -- convenience ---------------------------------------------------------

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def is_probably_scanned(self) -> bool:
        """Heuristic for an image-only PDF that needs OCR.

        A page of real text is normally well over a thousand characters. Under a
        hundred per page almost always means the text layer is missing.
        """
        return self.char_count / max(self.page_count, 1) < 100


class TextAccumulator:
    """Builds a document's canonical text while recording where each line landed.

    Downstream code should never concatenate document text by hand. Push lines
    through here so the offset map stays truthful.
    """

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._blocks: list[TextBlock] = []
        self._offset = 0

    def add_line(
        self,
        text: str,
        *,
        page: int,
        bbox: BoundingBox | None = None,
        is_heading: bool = False,
        font_size: float | None = None,
    ) -> Span | None:
        """Append one visual line. Returns the span it occupies, or None if blank."""
        stripped = text.rstrip()
        if not stripped:
            self._parts.append("\n")
            self._offset += 1
            return None

        span = Span(start=self._offset, end=self._offset + len(stripped))
        self._parts.append(stripped + "\n")
        self._offset += len(stripped) + 1
        self._blocks.append(
            TextBlock(
                span=span,
                page=page,
                bbox=bbox,
                is_heading=is_heading,
                font_size=font_size,
            )
        )
        return span

    def add_page_break(self) -> None:
        self._parts.append("\n")
        self._offset += 1

    def build(
        self,
        *,
        document_id: str,
        filename: str,
        source_format: SourceFormat,
        page_count: int,
        metadata: dict[str, str] | None = None,
    ) -> SourceDocument:
        return SourceDocument(
            document_id=document_id,
            filename=filename,
            source_format=source_format,
            text="".join(self._parts),
            blocks=self._blocks,
            page_count=max(page_count, 1),
            metadata=metadata or {},
        )


class IngestionError(RuntimeError):
    """Raised when a document cannot be read at all."""


def detect_format(path: Path) -> SourceFormat:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return SourceFormat.PDF
    if suffix in {".docx", ".doc"}:
        return SourceFormat.DOCX
    if suffix in {".txt", ".md", ".markdown"}:
        return SourceFormat.TEXT
    raise IngestionError(
        f"Unsupported file type '{suffix}'. HireLens reads .pdf, .docx, .txt and .md."
    )
