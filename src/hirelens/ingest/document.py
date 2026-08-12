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
    model_config = ConfigDict(frozen=True)

    span: Span
    page: int = Field(ge=1, description="1-indexed page number")
    bbox: BoundingBox | None = None
    is_heading: bool = False
    font_size: float | None = None


class SourceDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    filename: str
    source_format: SourceFormat
    text: str
    blocks: list[TextBlock] = Field(default_factory=list)
    page_count: int = Field(default=1, ge=1)
    metadata: dict[str, str] = Field(default_factory=dict)

    @staticmethod
    def make_id(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()[:16]

    def slice(self, span: Span) -> str:
        return span.slice_of(self.text)

    def page_of(self, span: Span) -> int | None:
        for block in self.blocks:
            if block.span.start <= span.start < block.span.end:
                return block.page
        return None

    def blocks_for(self, span: Span) -> list[TextBlock]:
        return [b for b in self.blocks if b.span.overlaps(span)]

    def highlight_boxes(self, span: Span) -> list[tuple[int, BoundingBox]]:
        return [(b.page, b.bbox) for b in self.blocks_for(span) if b.bbox is not None]

    def line_at(self, offset: int) -> str:
        start = self.text.rfind("\n", 0, offset) + 1
        end = self.text.find("\n", offset)
        return self.text[start : end if end != -1 else len(self.text)]

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def is_probably_scanned(self) -> bool:
        return self.char_count / max(self.page_count, 1) < 100


class TextAccumulator:
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
    pass


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
