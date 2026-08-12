from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest

from hirelens.config import Provider, Settings
from hirelens.extract.locator import SpanLocator
from hirelens.extract.pii import (
    PIICategory,
    detect_pii,
    redact,
)
from hirelens.extract.sections import SectionKind, classify_heading, segment
from hirelens.ingest import read_document
from hirelens.llm.base import CompletionRequest, CompletionResponse, LLMProvider, Usage
from hirelens.llm.client import LLMClient

RESUME = """PRIYA NARAYANAN
priya.n@example.com | +91 98765 43210 | Bengaluru
github.com/pnarayanan

EXPERIENCE
Backend Engineer, Acme Corp (2023 - present)
Cut p99 checkout latency from 1.2s to 180ms by removing an ORM N+1.
Owned reconciliation handling 40k transactions daily.

PROJECTS
kvstore - Raft-based distributed key-value store in Go.

EDUCATION
B.Tech Computer Science, NIT Trichy (2019 - 2023). CGPA 8.7

SKILLS
Go, Python, PostgreSQL, Kafka, Kubernetes
"""


@pytest.fixture
def resume_doc(tmp_path: Path):
    path = tmp_path / "priya.txt"
    path.write_text(RESUME, encoding="utf-8")
    return read_document(path)


class TestSpanLocator:
    def test_exact_match(self) -> None:
        locator = SpanLocator(RESUME)
        found = locator.locate("Backend Engineer, Acme Corp")
        assert found is not None
        assert found.strategy == "exact"
        assert RESUME[found.span.start : found.span.end] == "Backend Engineer, Acme Corp"

    def test_whitespace_normalised_match(self) -> None:
        locator = SpanLocator(RESUME)
        found = locator.locate("Backend   Engineer,    Acme Corp")
        assert found is not None
        assert found.strategy == "normalised"
        assert "Acme Corp" in RESUME[found.span.start : found.span.end]

    def test_case_insensitive_match(self) -> None:
        locator = SpanLocator(RESUME)
        found = locator.locate("BACKEND ENGINEER, ACME CORP")
        assert found is not None
        assert RESUME[found.span.start : found.span.end].lower().startswith("backend engineer")

    def test_fuzzy_match_on_a_paraphrased_tail(self) -> None:
        locator = SpanLocator(RESUME)
        found = locator.locate("Cut p99 checkout latency from 1.2s to 180ms")
        assert found is not None
        assert "p99 checkout latency" in RESUME[found.span.start : found.span.end]

    def test_invented_quote_is_not_located(self) -> None:
        locator = SpanLocator(RESUME)
        assert locator.locate("led a team of fifteen engineers at Google") is None

    def test_ambiguous_short_quote_is_refused(self) -> None:
        locator = SpanLocator(RESUME)
        assert locator.locate("Go") is None

    def test_unique_short_quote_is_accepted(self) -> None:
        locator = SpanLocator("SKILLS\nC, Rust, Elixir\n")
        found = locator.locate("C")
        assert found is not None
        assert found.strategy == "unique-short"
        assert found.span.start == 7

    def test_short_quote_becomes_unique_once_scoped(self) -> None:
        from hirelens.schemas.evidence import Span

        locator = SpanLocator(RESUME)
        skills_start = RESUME.index("SKILLS")
        found = locator.locate("Go", within=Span(start=skills_start, end=len(RESUME)))
        assert found is not None
        assert found.span.start >= skills_start

    def test_short_match_respects_word_boundaries(self) -> None:
        locator = SpanLocator("Worked at Google using Django and Go daily")
        found = locator.locate("Go")
        assert found is not None
        assert found.span.start == "Worked at Google using Django and ".__len__()

    def test_within_restricts_the_search(self) -> None:
        locator = SpanLocator(RESUME)
        skills_start = RESUME.index("SKILLS")
        unrestricted = locator.locate("Python")
        restricted = locator.locate(
            "Python",
            within=__import__("hirelens.schemas.evidence", fromlist=["Span"]).Span(
                start=skills_start, end=len(RESUME)
            ),
        )
        assert unrestricted is not None
        assert restricted is not None
        assert restricted.span.start >= skills_start

    def test_returned_spans_always_slice_back_to_real_text(self) -> None:
        locator = SpanLocator(RESUME)
        for quote in ["NIT Trichy", "kvstore", "PostgreSQL", "reconciliation"]:
            found = locator.locate(quote)
            assert found is not None, quote
            sliced = RESUME[found.span.start : found.span.end]
            assert quote.lower() in sliced.lower()

    def test_locate_all_skips_the_unfindable(self) -> None:
        locator = SpanLocator(RESUME)
        results = locator.locate_all(["kvstore", "invented achievement here", "Kafka"])
        assert set(results) == {"kvstore", "Kafka"}


class TestPIIDetection:
    def test_finds_the_obvious_identifiers(self) -> None:
        found = {s.category for s in detect_pii(RESUME)}
        assert PIICategory.NAME in found
        assert PIICategory.EMAIL in found
        assert PIICategory.LOCATION in found

    def test_name_is_the_header_line(self) -> None:
        names = [s for s in detect_pii(RESUME) if s.category is PIICategory.NAME]
        assert names and names[0].text == "PRIYA NARAYANAN"

    def test_institution_detected(self) -> None:
        found = [s.text for s in detect_pii(RESUME) if s.category is PIICategory.INSTITUTION]
        assert any("NIT" in text for text in found)

    def test_year_ranges_are_not_mistaken_for_phone_numbers(self) -> None:
        phones = [s.text for s in detect_pii("Worked 2019 - 2023 at Acme")]
        assert not phones

    def test_metrics_are_not_mistaken_for_phone_numbers(self) -> None:
        spans = detect_pii("Handled 40k transactions and cut latency to 180ms")
        assert not [s for s in spans if s.category is PIICategory.PHONE]

    def test_detected_spans_do_not_overlap(self) -> None:
        spans = detect_pii(RESUME)
        for earlier, later in pairwise(spans):
            assert not earlier.span.overlaps(later.span)

    def test_spans_slice_back_to_their_own_text(self) -> None:
        for item in detect_pii(RESUME):
            assert RESUME[item.span.start : item.span.end] == item.text


class TestRedaction:
    def test_masking_preserves_length(self) -> None:
        report = redact(RESUME)
        assert len(report.redacted_text) == len(RESUME)

    def test_name_and_email_are_gone(self) -> None:
        report = redact(RESUME)
        assert "PRIYA NARAYANAN" not in report.redacted_text
        assert "priya.n@example.com" not in report.redacted_text

    def test_technical_content_survives(self) -> None:
        report = redact(RESUME)
        for keeper in ["Backend Engineer", "p99 checkout latency", "kvstore", "PostgreSQL"]:
            assert keeper in report.redacted_text

    def test_github_url_is_kept_by_default(self) -> None:
        report = redact(RESUME)
        assert "github.com/pnarayanan" in report.redacted_text

    def test_url_masking_is_opt_in(self) -> None:
        report = redact(RESUME, categories=[PIICategory.URL])
        assert "github.com/pnarayanan" not in report.redacted_text

    def test_spans_valid_in_the_original_are_valid_in_the_masked_view(self) -> None:
        report = redact(RESUME)
        start = RESUME.index("p99 checkout latency")
        end = start + len("p99 checkout latency")
        assert report.redacted_text[start:end] == "p99 checkout latency"

    def test_empty_selection_changes_nothing(self) -> None:
        report = redact(RESUME, categories=[])
        assert report.redacted_text == RESUME

    def test_summary_is_human_readable(self) -> None:
        assert "name" in redact(RESUME).summary()


class TestSectionClassification:
    @pytest.mark.parametrize(
        ("heading", "expected"),
        [
            ("EXPERIENCE", SectionKind.WORK),
            ("Work Experience", SectionKind.WORK),
            ("PROFESSIONAL EXPERIENCE", SectionKind.WORK),
            ("Education", SectionKind.EDUCATION),
            ("PROJECTS", SectionKind.PROJECTS),
            ("Technical Skills", SectionKind.SKILLS),
            ("Certifications", SectionKind.AWARDS),
        ],
    )
    def test_known_headings(self, heading: str, expected: SectionKind) -> None:
        assert classify_heading(heading) is expected

    def test_work_experience_beats_experience(self) -> None:
        assert classify_heading("Work Experience") is SectionKind.WORK

    def test_unknown_heading_returns_none(self) -> None:
        assert classify_heading("References available on request") is None

    def test_a_long_sentence_is_not_a_heading(self) -> None:
        assert classify_heading("Improved the education portal search ranking") is None


class TestSegmentation:
    def test_finds_every_section(self, resume_doc) -> None:
        found = segment(resume_doc).kinds_found
        assert {
            SectionKind.WORK,
            SectionKind.PROJECTS,
            SectionKind.EDUCATION,
            SectionKind.SKILLS,
        } <= found

    def test_section_text_contains_only_its_own_content(self, resume_doc) -> None:
        section_map = segment(resume_doc)
        work = section_map.text_for(SectionKind.WORK, resume_doc)
        assert "Acme Corp" in work
        assert "kvstore" not in work

    def test_heading_line_is_excluded_from_the_body(self, resume_doc) -> None:
        section_map = segment(resume_doc)
        assert "SKILLS" not in section_map.text_for(SectionKind.SKILLS, resume_doc)

    def test_basics_covers_the_header(self, resume_doc) -> None:
        section_map = segment(resume_doc)
        assert "priya.n@example.com" in section_map.text_for(SectionKind.BASICS, resume_doc)

    def test_unstructured_document_still_produces_one_section(self, tmp_path: Path) -> None:
        path = tmp_path / "flat.txt"
        path.write_text("just a paragraph with no headings at all in it", encoding="utf-8")
        section_map = segment(read_document(path))
        assert len(section_map.sections) == 1


class ScriptedProvider(LLMProvider):
    name = "scripted"
    model = "scripted-model"

    def __init__(self, by_keyword: dict[str, dict]) -> None:
        self.by_keyword = by_keyword
        self.prompts: list[str] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        conversation = "\n".join(m.content for m in request.messages)
        self.prompts.append(request.messages[-1].content)
        for keyword, payload in self.by_keyword.items():
            if keyword in conversation:
                return CompletionResponse(
                    content=json.dumps(payload), model=self.model, usage=Usage(1, 1)
                )
        return CompletionResponse(content="{}", model=self.model, usage=Usage(1, 1))

    async def aclose(self) -> None:
        return None


def build_extractor(tmp_path: Path, provider: LLMProvider, *, blind: bool = False):
    from hirelens.extract import ResumeExtractor

    settings = Settings(
        llm_provider=Provider.OLLAMA,
        cache_dir=tmp_path / "cache",
        cache_enabled=False,
        blind_mode=blind,
        requests_per_minute=0,
    )
    return ResumeExtractor(LLMClient(provider, settings=settings), settings=settings)


SCRIPT = {
    "Extract the candidate's contact": {
        "basics": {
            "name": {"value": "Priya Narayanan", "quote": "PRIYA NARAYANAN"},
            "email": {"value": "priya.n@example.com", "quote": "priya.n@example.com"},
            "profiles": [
                {
                    "network": "GitHub",
                    "url": "github.com/pnarayanan",
                    "quote": "github.com/pnarayanan",
                }
            ],
        }
    },
    "Extract paid professional experience": {
        "work": [
            {
                "company": {"value": "Acme Corp", "quote": "Acme Corp"},
                "position": {"value": "Backend Engineer", "quote": "Backend Engineer"},
                "is_current": True,
                "highlights": [
                    {
                        "value": "Cut p99 latency to 180ms",
                        "quote": "Cut p99 checkout latency from 1.2s to 180ms",
                    },
                    {
                        "value": "Managed a team of 12",
                        "quote": "Managed a team of 12 engineers across three offices",
                    },
                ],
            }
        ]
    },
    "Extract personal, academic": {
        "projects": [
            {
                "name": {"value": "kvstore", "quote": "kvstore"},
                "description": {
                    "value": "Raft-based distributed key-value store",
                    "quote": "Raft-based distributed key-value store in Go",
                },
                "technologies": [{"value": "Go", "quote": "in Go"}],
            }
        ]
    },
    "Extract individual skills": {
        "skills": [
            {"name": {"value": "PostgreSQL", "quote": "PostgreSQL"}, "category": ""},
            {"name": {"value": "Kubernetes", "quote": "Kubernetes"}, "category": ""},
        ]
    },
    "Extract formal education": {
        "education": [
            {
                "institution": {"value": "NIT Trichy", "quote": "NIT Trichy"},
                "degree": {"value": "B.Tech", "quote": "B.Tech Computer Science"},
                "score": {"value": "8.7", "quote": "CGPA 8.7"},
            }
        ]
    },
}


class TestResumeExtractor:
    async def test_extracts_all_sections(self, resume_doc, tmp_path: Path) -> None:
        extractor = build_extractor(tmp_path, ScriptedProvider(SCRIPT))
        result = await extractor.extract(resume_doc, blind=False)
        resume = result.resume

        assert resume.basics.name is not None
        assert resume.basics.name.value == "Priya Narayanan"
        assert resume.work[0].company.value == "Acme Corp"
        assert resume.education[0].institution.value == "NIT Trichy"
        assert resume.projects[0].name.value == "kvstore"
        assert "PostgreSQL" in resume.skill_names

    async def test_real_quotes_are_grounded_and_verify(self, resume_doc, tmp_path: Path) -> None:
        extractor = build_extractor(tmp_path, ScriptedProvider(SCRIPT))
        resume = (await extractor.extract(resume_doc, blind=False)).resume

        company = resume.work[0].company
        assert company.is_grounded
        assert company.verify(resume_doc.text).ok
        assert company.citations[0].resolved_quote(resume_doc.text) == "Acme Corp"

    async def test_hallucinated_quote_is_reported_not_cited(
        self, resume_doc, tmp_path: Path
    ) -> None:
        extractor = build_extractor(tmp_path, ScriptedProvider(SCRIPT))
        resume = (await extractor.extract(resume_doc, blind=False)).resume

        invented = next(h for h in resume.work[0].highlights if "team of 12" in h.value)
        assert not invented.is_grounded
        assert any("team of 12" in quote for quote in resume.grounding.unlocatable_quotes)

    async def test_every_citation_in_the_output_verifies(self, resume_doc, tmp_path: Path) -> None:
        extractor = build_extractor(tmp_path, ScriptedProvider(SCRIPT))
        resume = (await extractor.extract(resume_doc, blind=False)).resume
        assert resume.verify(resume_doc.text).ok

    async def test_grounding_stats_are_recorded(self, resume_doc, tmp_path: Path) -> None:
        extractor = build_extractor(tmp_path, ScriptedProvider(SCRIPT))
        resume = (await extractor.extract(resume_doc, blind=False)).resume

        assert resume.grounding.total_fields > 0
        assert 0.0 < resume.grounding.grounding_rate < 1.0
        assert resume.grounding.citation_validity_rate == 1.0

    async def test_citations_carry_a_page_number(self, resume_doc, tmp_path: Path) -> None:
        extractor = build_extractor(tmp_path, ScriptedProvider(SCRIPT))
        resume = (await extractor.extract(resume_doc, blind=False)).resume

        citation = resume.work[0].company.citations[0]
        assert citation.page == 1

    async def test_github_profile_is_discoverable(self, resume_doc, tmp_path: Path) -> None:
        extractor = build_extractor(tmp_path, ScriptedProvider(SCRIPT))
        resume = (await extractor.extract(resume_doc, blind=False)).resume
        assert resume.basics.github_url() == "github.com/pnarayanan"

    async def test_sections_are_extracted_from_their_own_text(
        self, resume_doc, tmp_path: Path
    ) -> None:
        provider = ScriptedProvider(SCRIPT)
        extractor = build_extractor(tmp_path, provider)
        await extractor.extract(resume_doc, blind=False)

        work_prompt = next(p for p in provider.prompts if "paid professional experience" in p)
        assert "Acme Corp" in work_prompt
        assert "kvstore" not in work_prompt

    async def test_a_failing_section_does_not_sink_the_parse(
        self, resume_doc, tmp_path: Path
    ) -> None:
        broken = dict(SCRIPT)
        broken["Extract personal, academic"] = {"projects": "this is not a list"}
        provider = ScriptedProvider(broken)
        extractor = build_extractor(tmp_path, provider)
        result = await extractor.extract(resume_doc, blind=False)

        assert "projects" in result.resume.failed_sections
        assert not result.resume.projects
        assert result.resume.work
        assert result.resume.education
        assert result.resume.skills
        assert result.resume.basics.name is not None

    async def test_a_failing_section_is_retried_before_being_given_up_on(
        self, resume_doc, tmp_path: Path
    ) -> None:
        broken = dict(SCRIPT)
        broken["Extract personal, academic"] = {"projects": "this is not a list"}
        provider = ScriptedProvider(broken)
        extractor = build_extractor(tmp_path, provider)
        await extractor.extract(resume_doc, blind=False)

        repair_turns = [p for p in provider.prompts if "did not validate" in p]
        assert len(repair_turns) == 2

    async def test_blind_mode_hides_identity_from_the_prompt(
        self, resume_doc, tmp_path: Path
    ) -> None:
        provider = ScriptedProvider(SCRIPT)
        extractor = build_extractor(tmp_path, provider, blind=True)
        await extractor.extract(resume_doc, blind=True)

        assert provider.prompts, "expected at least one prompt"
        assert all("PRIYA NARAYANAN" not in prompt for prompt in provider.prompts)
        assert any("Backend Engineer" in prompt for prompt in provider.prompts)

    async def test_blind_mode_still_grounds_technical_claims(
        self, resume_doc, tmp_path: Path
    ) -> None:
        extractor = build_extractor(tmp_path, ScriptedProvider(SCRIPT), blind=True)
        resume = (await extractor.extract(resume_doc, blind=True)).resume
        assert resume.work[0].company.is_grounded
