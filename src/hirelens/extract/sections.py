from __future__ import annotations

import logging
import re

from pydantic import BaseModel, ConfigDict

from hirelens._compat import StrEnum
from hirelens.ingest.document import SourceDocument
from hirelens.schemas.evidence import Span

logger = logging.getLogger(__name__)


class SectionKind(StrEnum):
    BASICS = "basics"
    WORK = "work"
    EDUCATION = "education"
    PROJECTS = "projects"
    SKILLS = "skills"
    AWARDS = "awards"
    OTHER = "other"


_HEADING_KEYWORDS: dict[SectionKind, tuple[str, ...]] = {
    SectionKind.WORK: (
        "work experience",
        "professional experience",
        "employment history",
        "work history",
        "experience",
        "employment",
        "internships",
        "internship",
    ),
    SectionKind.EDUCATION: ("education", "academic background", "qualifications", "academics"),
    SectionKind.PROJECTS: (
        "projects",
        "personal projects",
        "side projects",
        "selected projects",
        "open source",
        "portfolio",
    ),
    SectionKind.SKILLS: (
        "skills",
        "technical skills",
        "technologies",
        "tech stack",
        "competencies",
        "languages and tools",
    ),
    SectionKind.AWARDS: (
        "awards",
        "achievements",
        "honors",
        "honours",
        "certifications",
        "certificates",
        "publications",
    ),
}

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


class Section(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: SectionKind
    span: Span
    heading: str = ""

    def text_from(self, document: SourceDocument) -> str:
        return document.slice(self.span)


class SectionMap(BaseModel):
    sections: list[Section]

    def of(self, kind: SectionKind) -> list[Section]:
        return [s for s in self.sections if s.kind is kind]

    def span_for(self, kind: SectionKind) -> Span | None:
        matches = self.of(kind)
        if not matches:
            return None
        return Span(start=min(s.span.start for s in matches), end=max(s.span.end for s in matches))

    def text_for(self, kind: SectionKind, document: SourceDocument) -> str:
        return "\n".join(s.text_from(document) for s in self.of(kind))

    @property
    def kinds_found(self) -> set[SectionKind]:
        return {s.kind for s in self.sections}


def segment(document: SourceDocument) -> SectionMap:
    boundaries = _heading_boundaries(document)

    if not boundaries:
        logger.debug("no headings detected in %s, falling back to keyword scan", document.filename)
        boundaries = _keyword_boundaries(document)

    if not boundaries:
        logger.warning(
            "%s has no recognisable section structure; treating the whole document as one "
            "section. Extraction will still run but with less precise grounding.",
            document.filename,
        )
        return SectionMap(
            sections=[
                Section(kind=SectionKind.OTHER, span=Span(start=0, end=max(len(document.text), 1)))
            ]
        )

    sections: list[Section] = []

    first_start = boundaries[0][0]
    if first_start > 0:
        sections.append(Section(kind=SectionKind.BASICS, span=Span(start=0, end=first_start)))

    for index, (_start, heading_end, kind, heading) in enumerate(boundaries):
        end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(document.text)
        body_start = min(heading_end + 1, end)
        if body_start >= end:
            continue
        sections.append(Section(kind=kind, span=Span(start=body_start, end=end), heading=heading))

    return SectionMap(sections=sections)


def _heading_boundaries(document: SourceDocument) -> list[tuple[int, int, SectionKind, str]]:
    boundaries: list[tuple[int, int, SectionKind, str]] = []
    for block in document.blocks:
        if not block.is_heading:
            continue
        heading = document.slice(block.span).strip()
        kind = classify_heading(heading)
        if kind is not None:
            boundaries.append((block.span.start, block.span.end, kind, heading))
    return boundaries


def _keyword_boundaries(document: SourceDocument) -> list[tuple[int, int, SectionKind, str]]:
    boundaries: list[tuple[int, int, SectionKind, str]] = []
    offset = 0
    for line in document.text.split("\n"):
        stripped = line.strip()
        if stripped and len(stripped) <= 40:
            kind = classify_heading(stripped)
            if kind is not None:
                start = offset + line.index(stripped)
                boundaries.append((start, start + len(stripped), kind, stripped))
        offset += len(line) + 1
    return boundaries


def classify_heading(heading: str) -> SectionKind | None:
    cleaned = _NON_ALNUM.sub(" ", heading.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned or len(cleaned) > 40:
        return None

    candidates = [
        (keyword, kind) for kind, keywords in _HEADING_KEYWORDS.items() for keyword in keywords
    ]
    for keyword, kind in sorted(candidates, key=lambda pair: -len(pair[0])):
        if (
            cleaned == keyword
            or cleaned.startswith(f"{keyword} ")
            or cleaned.endswith(f" {keyword}")
        ):
            return kind
    return None
