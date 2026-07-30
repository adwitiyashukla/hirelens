"""Turn a source document into a :class:`CitedResume`.

The pipeline for one resume:

1. **Segment** into sections using detected headings.
2. **Redact** identifying spans, length-preservingly, when blind mode is on.
3. **Extract** each section concurrently, one focused LLM call per section,
   returning raw values plus verbatim quotes.
4. **Locate** every quote in the source ourselves and turn it into a span.
5. **Verify** each resolved citation actually contains the quoted text.
6. **Report** grounding statistics so degradation is visible rather than silent.

Steps 4 and 5 are the ones that make this different from a normal resume parser.
The model is never asked where something is, only what it says, and the claim is
checked against the document before it is allowed into the output.

Section failures are isolated. If the model returns garbage for the projects
section, that section is recorded in ``failed_sections`` and the other five still
produce a usable resume. One bad section should not cost a whole candidate.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TypeVar

from pydantic import BaseModel

from hirelens.config import Settings, get_settings
from hirelens.extract.locator import SpanLocator
from hirelens.extract.pii import PIICategory, RedactionReport, redact
from hirelens.extract.prompts import SYSTEM_MESSAGE, build_extraction_prompt, supported_kinds
from hirelens.extract.sections import SectionKind, SectionMap, segment
from hirelens.ingest.document import SourceDocument
from hirelens.llm.base import LLMError
from hirelens.llm.client import LLMClient
from hirelens.schemas.evidence import Citation, Cited, Span
from hirelens.schemas.resume import (
    Award,
    Basics,
    CitedResume,
    Education,
    GroundingStats,
    Profile,
    Project,
    RawAwardsSection,
    RawBasicsSection,
    RawEducationSection,
    RawField,
    RawProjectsSection,
    RawSkillsSection,
    RawWorkSection,
    Skill,
    WorkExperience,
)

logger = logging.getLogger(__name__)

TSection = TypeVar("TSection", bound=BaseModel)

_SECTION_SCHEMAS: dict[SectionKind, type[BaseModel]] = {
    SectionKind.BASICS: RawBasicsSection,
    SectionKind.WORK: RawWorkSection,
    SectionKind.EDUCATION: RawEducationSection,
    SectionKind.PROJECTS: RawProjectsSection,
    SectionKind.SKILLS: RawSkillsSection,
    SectionKind.AWARDS: RawAwardsSection,
}

# Below this, a section is almost certainly a stray heading with no body, and
# calling the model on it wastes a request from a limited free-tier quota.
_MIN_SECTION_CHARS = 15


class ExtractionResult(BaseModel):
    """Everything one parse produced, including the diagnostics."""

    resume: CitedResume
    redaction: RedactionReport
    sections_found: list[str]
    llm_usage: dict[str, object] = {}


class ResumeExtractor:
    """Extracts a cited resume from a document."""

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or LLMClient(settings=self.settings)

    async def extract(
        self,
        document: SourceDocument,
        *,
        blind: bool | None = None,
        categories: frozenset[PIICategory] | None = None,
    ) -> ExtractionResult:
        """Parse ``document`` into a :class:`CitedResume`."""
        blind = self.settings.blind_mode if blind is None else blind

        section_map = segment(document)
        logger.info(
            "%s: found sections %s",
            document.filename,
            sorted(str(k) for k in section_map.kinds_found),
        )

        report = redact(document.text, categories=categories)
        # The model reads the masked view; the locator searches the same view, so
        # spans resolved here are valid against the original text too. That only
        # holds because masking preserves length.
        model_text = report.redacted_text if blind else document.text
        locator = SpanLocator(model_text)

        raw_sections = await self._extract_all_sections(document, section_map, model_text)

        resume = self._assemble(
            document=document,
            raw_sections=raw_sections,
            section_map=section_map,
            locator=locator,
            source_text=model_text,
        )

        return ExtractionResult(
            resume=resume,
            redaction=report,
            sections_found=sorted(str(k) for k in section_map.kinds_found),
            llm_usage=self.client.usage_summary(),
        )

    # -- LLM calls -----------------------------------------------------------

    async def _extract_all_sections(
        self,
        document: SourceDocument,
        section_map: SectionMap,
        model_text: str,
    ) -> dict[SectionKind, BaseModel | None]:
        """One concurrent call per section. Failures isolated per section."""
        kinds = [k for k in supported_kinds() if self._section_text(k, section_map, model_text)]
        results = await asyncio.gather(
            *(self._extract_section(kind, section_map, model_text) for kind in kinds),
            return_exceptions=True,
        )

        out: dict[SectionKind, BaseModel | None] = {}
        for kind, result in zip(kinds, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning("%s section failed: %s", kind, result)
                out[kind] = None
            else:
                out[kind] = result
        return out

    async def _extract_section(
        self, kind: SectionKind, section_map: SectionMap, model_text: str
    ) -> BaseModel | None:
        text = self._section_text(kind, section_map, model_text)
        if not text:
            return None

        schema = _SECTION_SCHEMAS[kind]
        try:
            return await self.client.structured(
                schema,  # type: ignore[type-var]
                system=SYSTEM_MESSAGE,
                user=build_extraction_prompt(kind, text),
                temperature=self.settings.extraction_temperature,
            )
        except LLMError as exc:
            logger.warning("%s extraction failed after repairs: %s", kind, exc)
            raise

    def _section_text(self, kind: SectionKind, section_map: SectionMap, model_text: str) -> str:
        """The slice of text for one section, with sensible fallbacks."""
        span = section_map.span_for(kind)

        if span is None and kind is SectionKind.BASICS:
            # No explicit contact block: use the header, where it always lives.
            span = Span(start=0, end=min(500, len(model_text)))

        if span is None:
            # Unstructured resume: fall back to the whole document rather than
            # skipping the section entirely.
            other = section_map.span_for(SectionKind.OTHER)
            if other is None:
                return ""
            span = other

        text = model_text[span.start : span.end].strip()
        return text if len(text) >= _MIN_SECTION_CHARS else ""

    # -- assembly ------------------------------------------------------------

    def _assemble(
        self,
        *,
        document: SourceDocument,
        raw_sections: dict[SectionKind, BaseModel | None],
        section_map: SectionMap,
        locator: SpanLocator,
        source_text: str,
    ) -> CitedResume:
        document_id = document.document_id
        tracker = _GroundingTracker(locator=locator, source_text=source_text, document=document)

        def window(kind: SectionKind) -> Span | None:
            return section_map.span_for(kind)

        basics = Basics()
        raw_basics = raw_sections.get(SectionKind.BASICS)
        if isinstance(raw_basics, RawBasicsSection):
            scope = window(SectionKind.BASICS)
            rb = raw_basics.basics
            basics = Basics(
                name=tracker.cite(rb.name, scope),
                email=tracker.cite(rb.email, scope),
                phone=tracker.cite(rb.phone, scope),
                location=tracker.cite(rb.location, scope),
                headline=tracker.cite(rb.headline, scope),
                profiles=[
                    Profile(
                        network=tracker.cite_value(p.network, p.quote, scope),
                        url=tracker.cite_value(p.url, p.quote or p.url, scope),
                    )
                    for p in rb.profiles
                ],
            )

        work: list[WorkExperience] = []
        raw_work = raw_sections.get(SectionKind.WORK)
        if isinstance(raw_work, RawWorkSection):
            scope = window(SectionKind.WORK)
            for job in raw_work.work:
                company = tracker.cite(job.company, scope)
                if company is None:
                    continue
                work.append(
                    WorkExperience(
                        company=company,
                        position=tracker.cite(job.position, scope),
                        start_date=tracker.cite(job.start_date, scope),
                        end_date=tracker.cite(job.end_date, scope),
                        is_current=job.is_current,
                        highlights=tracker.cite_many(job.highlights, scope),
                    )
                )

        education: list[Education] = []
        raw_education = raw_sections.get(SectionKind.EDUCATION)
        if isinstance(raw_education, RawEducationSection):
            scope = window(SectionKind.EDUCATION)
            for entry in raw_education.education:
                institution = tracker.cite(entry.institution, scope)
                if institution is None:
                    continue
                education.append(
                    Education(
                        institution=institution,
                        degree=tracker.cite(entry.degree, scope),
                        field_of_study=tracker.cite(entry.field_of_study, scope),
                        start_date=tracker.cite(entry.start_date, scope),
                        end_date=tracker.cite(entry.end_date, scope),
                        score=tracker.cite(entry.score, scope),
                    )
                )

        projects: list[Project] = []
        raw_projects = raw_sections.get(SectionKind.PROJECTS)
        if isinstance(raw_projects, RawProjectsSection):
            scope = window(SectionKind.PROJECTS)
            for entry in raw_projects.projects:
                name = tracker.cite(entry.name, scope)
                if name is None:
                    continue
                projects.append(
                    Project(
                        name=name,
                        description=tracker.cite(entry.description, scope),
                        url=tracker.cite(entry.url, scope),
                        technologies=tracker.cite_many(entry.technologies, scope),
                        highlights=tracker.cite_many(entry.highlights, scope),
                    )
                )

        skills: list[Skill] = []
        raw_skills = raw_sections.get(SectionKind.SKILLS)
        if isinstance(raw_skills, RawSkillsSection):
            scope = window(SectionKind.SKILLS)
            for entry in raw_skills.skills:
                name = tracker.cite(entry.name, scope)
                if name is not None:
                    skills.append(Skill(name=name, category=entry.category))

        awards: list[Award] = []
        raw_awards = raw_sections.get(SectionKind.AWARDS)
        if isinstance(raw_awards, RawAwardsSection):
            scope = window(SectionKind.AWARDS)
            for entry in raw_awards.awards:
                title = tracker.cite(entry.title, scope)
                if title is not None:
                    awards.append(
                        Award(
                            title=title,
                            awarder=tracker.cite(entry.awarder, scope),
                            date=tracker.cite(entry.date, scope),
                        )
                    )

        failed = [str(kind) for kind, value in raw_sections.items() if value is None]

        return CitedResume(
            document_id=document_id,
            basics=basics,
            work=work,
            education=education,
            projects=projects,
            skills=skills,
            awards=awards,
            grounding=tracker.stats(),
            failed_sections=failed,
        )


class _GroundingTracker:
    """Converts raw quoted values into cited ones, counting as it goes."""

    def __init__(self, *, locator: SpanLocator, source_text: str, document: SourceDocument) -> None:
        self.locator = locator
        self.source_text = source_text
        self.document = document
        self.document_id = document.document_id
        self.total_fields = 0
        self.grounded_fields = 0
        self.total_citations = 0
        self.valid_citations = 0
        self.unlocatable: list[str] = []

    def cite(self, field: RawField | None, scope: Span | None) -> Cited[str] | None:
        if field is None or not str(field.value).strip():
            return None
        return self.cite_value(field.value, field.quote, scope)

    def cite_value(self, value: str, quote: str, scope: Span | None) -> Cited[str]:
        """Resolve one value's quote into a verified citation.

        Falls back to searching for the value itself when the model gave no quote.
        A model that returns ``value="Acme Corp"`` with an empty quote has still
        told us something locatable, and recovering it costs one string search.
        """
        self.total_fields += 1
        value = value.strip()

        # Try the model's quote first, then the value, then a widened search
        # without the section restriction.
        for candidate, within in ((quote, scope), (value, scope), (quote, None), (value, None)):
            if not candidate or not candidate.strip():
                continue
            located = self.locator.locate(candidate.strip(), within=within)
            if located is None:
                continue

            citation = Citation(
                document_id=self.document_id,
                span=located.span,
                quote=self.source_text[located.span.start : located.span.end],
                # Resolve the page now so the frontend can jump straight to it
                # without needing the document's block map at render time.
                page=self.document.page_of(located.span),
            )
            self.total_citations += 1
            if citation.verify(self.source_text):
                self.valid_citations += 1
            self.grounded_fields += 1
            return Cited(value=value, citations=[citation], confidence=located.score)

        if quote.strip():
            self.unlocatable.append(quote.strip()[:120])
        return Cited.inferred(value, confidence=0.3)

    def cite_many(self, fields: list[RawField], scope: Span | None) -> list[Cited[str]]:
        out: list[Cited[str]] = []
        for field in fields:
            cited = self.cite(field, scope)
            if cited is not None:
                out.append(cited)
        return out

    def stats(self) -> GroundingStats:
        return GroundingStats(
            total_fields=self.total_fields,
            grounded_fields=self.grounded_fields,
            total_citations=self.total_citations,
            valid_citations=self.valid_citations,
            unlocatable_quotes=self.unlocatable[:20],
        )
