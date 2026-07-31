"""Phase 4 tests: verdict aggregation, judging, self-consistency, risks, pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hirelens.assess.judge import RequirementJudge, build_prompt
from hirelens.assess.pipeline import (
    ScreeningPipeline,
    _reject_degenerate_extraction,
    rank,
)
from hirelens.assess.questions import collect_gaps
from hirelens.assess.risks import detect_risks, parse_month_index
from hirelens.assess.rubric import RubricCompiler
from hirelens.config import Provider, Settings
from hirelens.ingest.document import SourceDocument
from hirelens.llm.base import CompletionRequest, CompletionResponse, LLMProvider, Usage
from hirelens.llm.client import LLMClient
from hirelens.retrieve.embeddings import HashingEmbedder
from hirelens.retrieve.hybrid import RetrievalHit
from hirelens.schemas.assessment import (
    VERDICT_VALUES,
    CandidateAssessment,
    RequirementAssessment,
    Verdict,
    aggregate_verdicts,
)
from hirelens.schemas.evidence import Citation, Cited, EvidenceUnit, Span
from hirelens.schemas.job import (
    RawRequirement,
    RawRubric,
    Requirement,
    RequirementCategory,
    RequirementKind,
    Rubric,
)
from hirelens.schemas.resume import CitedResume, Project, WorkExperience

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

SOURCE = """Backend Engineer, Fintech Co. 2021 - present
Deployed and operated the payments service on Kubernetes.
Built a Kafka consumer group processing 2M settlement events per day.
Was responsible for helping with the internal admin tools.
kvstore - Raft-based distributed key-value store in Go.
"""


def cite(fragment: str) -> Cited[str]:
    start = SOURCE.index(fragment)
    return Cited(
        value=fragment,
        citations=[
            Citation(
                document_id="doc-1",
                span=Span(start=start, end=start + len(fragment)),
                quote=fragment,
                page=1,
            )
        ],
    )


def unit(unit_id: str, fragment: str, *, context: str = "") -> EvidenceUnit:
    """An evidence unit whose span really covers ``fragment`` in SOURCE.

    ``context`` mirrors what chunking does in production: it widens the searchable
    text without widening the span, so these fixtures exercise the same
    text-versus-quote split the real pipeline has.
    """
    start = SOURCE.index(fragment)
    return EvidenceUnit(
        unit_id=unit_id,
        document_id="doc-1",
        text=f"{context} {fragment}".strip(),
        quote=fragment,
        span=Span(start=start, end=start + len(fragment)),
        section="work",
        page=1,
    )


def hit(unit_id: str, fragment: str, *, context: str = "") -> RetrievalHit:
    return RetrievalHit(unit=unit(unit_id, fragment, context=context), score=0.5, bm25_rank=1)


def requirement(text: str, kind: RequirementKind = RequirementKind.MUST_HAVE):
    return Rubric.from_raw(
        RawRubric(
            requirements=[
                RawRequirement(
                    text=text,
                    kind=kind,
                    category=RequirementCategory.EXPERIENCE,
                    evidence_hint=text,
                )
            ]
        ),
        source_text=text,
    ).requirements[0]


def settings_for(tmp_path: Path, **overrides):
    base = {
        "llm_provider": Provider.OLLAMA,
        "cache_enabled": False,
        "cache_dir": tmp_path,
        "self_consistency_k": 5,
        "requests_per_minute": 0,
    }
    return Settings(**{**base, **overrides})


class SequenceProvider(LLMProvider):
    """Cycles through scripted verdicts, so sample spread can be controlled."""

    name = "seq"
    model = "seq-model"

    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = payloads
        self.calls: list[str] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        payload = self.payloads[len(self.calls) % len(self.payloads)]
        self.calls.append(request.messages[-1].content)
        return CompletionResponse(content=json.dumps(payload), model=self.model, usage=Usage(1, 1))

    async def aclose(self) -> None:
        return None


def judgement(verdict: str, units: list[str] | None = None) -> dict:
    return {
        "verdict": verdict,
        "reasoning": f"Evidence supports this at the {verdict} level.",
        "evidence_unit_ids": units or ["u1"],
    }


# ---------------------------------------------------------------------------
# Verdict scale
# ---------------------------------------------------------------------------


class TestVerdictAggregation:
    def test_unanimous_samples(self) -> None:
        assert aggregate_verdicts([Verdict.CLEAR] * 5) is Verdict.CLEAR

    def test_median_not_mode(self) -> None:
        """Mode would pick WEAK here; the median is the more defensible answer."""
        samples = [Verdict.WEAK, Verdict.WEAK, Verdict.CLEAR, Verdict.STRONG, Verdict.STRONG]
        assert aggregate_verdicts(samples) is Verdict.CLEAR

    def test_single_outlier_does_not_move_the_result(self) -> None:
        samples = [Verdict.CLEAR, Verdict.CLEAR, Verdict.CLEAR, Verdict.CLEAR, Verdict.NONE]
        assert aggregate_verdicts(samples) is Verdict.CLEAR

    def test_even_split_resolves_downward(self) -> None:
        """Overstating fit is the costlier error, so ties go to the lower verdict."""
        assert aggregate_verdicts([Verdict.PARTIAL, Verdict.CLEAR]) is Verdict.PARTIAL

    def test_empty_samples(self) -> None:
        assert aggregate_verdicts([]) is Verdict.NONE

    def test_scale_is_monotonic(self) -> None:
        values = [
            VERDICT_VALUES[v]
            for v in (Verdict.NONE, Verdict.WEAK, Verdict.PARTIAL, Verdict.CLEAR, Verdict.STRONG)
        ]
        assert values == sorted(values)


class TestRequirementAssessment:
    def build(self, verdict: Verdict, samples: list[Verdict], weight: float = 20.0):
        return RequirementAssessment(
            requirement_id="r1",
            requirement_text="Has run Kubernetes in production",
            kind=RequirementKind.MUST_HAVE,
            weight=weight,
            verdict=verdict,
            samples=samples,
        )

    def test_points_are_weight_times_coefficient(self) -> None:
        item = self.build(Verdict.CLEAR, [Verdict.CLEAR] * 5, weight=20.0)
        assert item.points == pytest.approx(16.0)  # 20 * 0.8

    def test_consistent_samples_are_not_ambiguous(self) -> None:
        item = self.build(Verdict.CLEAR, [Verdict.CLEAR] * 5)
        assert not item.is_ambiguous
        assert item.spread == 0.0
        assert item.agreement == 1.0

    def test_split_samples_are_flagged_for_review(self) -> None:
        item = self.build(
            Verdict.PARTIAL, [Verdict.NONE, Verdict.PARTIAL, Verdict.PARTIAL, Verdict.STRONG]
        )
        assert item.is_ambiguous
        assert item.spread == pytest.approx(1.0)

    def test_met_and_unmet_are_mutually_exclusive(self) -> None:
        assert self.build(Verdict.CLEAR, []).is_met
        assert self.build(Verdict.NONE, []).is_unmet
        assert not self.build(Verdict.PARTIAL, []).is_met
        assert not self.build(Verdict.PARTIAL, []).is_unmet


class TestCandidateAssessment:
    def build(self, items: list[RequirementAssessment]) -> CandidateAssessment:
        return CandidateAssessment(document_id="doc-1", rubric_id="rb-1", assessments=items)

    def test_score_is_the_weighted_sum(self) -> None:
        a = self.build(
            [
                RequirementAssessment(
                    requirement_id="r1",
                    requirement_text="a",
                    kind=RequirementKind.MUST_HAVE,
                    weight=60.0,
                    verdict=Verdict.CLEAR,
                    samples=[Verdict.CLEAR],
                ),
                RequirementAssessment(
                    requirement_id="r2",
                    requirement_text="b",
                    kind=RequirementKind.NICE_TO_HAVE,
                    weight=40.0,
                    verdict=Verdict.PARTIAL,
                    samples=[Verdict.PARTIAL],
                ),
            ]
        )
        assert a.score == pytest.approx(68.0)  # 60*0.8 + 40*0.5

    def test_confidence_band_reflects_sample_spread(self) -> None:
        a = self.build(
            [
                RequirementAssessment(
                    requirement_id="r1",
                    requirement_text="a",
                    kind=RequirementKind.MUST_HAVE,
                    weight=100.0,
                    verdict=Verdict.PARTIAL,
                    samples=[Verdict.WEAK, Verdict.PARTIAL, Verdict.CLEAR],
                )
            ]
        )
        low, high = a.score_range
        assert low == pytest.approx(20.0)  # WEAK
        assert high == pytest.approx(80.0)  # CLEAR
        assert a.uncertainty == pytest.approx(60.0)

    def test_unanimous_samples_give_a_zero_width_band(self) -> None:
        a = self.build(
            [
                RequirementAssessment(
                    requirement_id="r1",
                    requirement_text="a",
                    kind=RequirementKind.MUST_HAVE,
                    weight=100.0,
                    verdict=Verdict.CLEAR,
                    samples=[Verdict.CLEAR] * 5,
                )
            ]
        )
        assert a.uncertainty == 0.0

    def test_unmet_must_have_overrides_a_good_score(self) -> None:
        """A 75 that is missing a hard requirement is not a strong fit."""
        a = self.build(
            [
                RequirementAssessment(
                    requirement_id="r1",
                    requirement_text="nice thing",
                    kind=RequirementKind.NICE_TO_HAVE,
                    weight=80.0,
                    verdict=Verdict.STRONG,
                    samples=[Verdict.STRONG],
                ),
                RequirementAssessment(
                    requirement_id="r2",
                    requirement_text="hard requirement",
                    kind=RequirementKind.MUST_HAVE,
                    weight=20.0,
                    verdict=Verdict.NONE,
                    samples=[Verdict.NONE],
                ),
            ]
        )
        assert a.score == pytest.approx(80.0)
        assert not a.meets_all_must_haves
        assert a.band == "missing a must-have"

    def test_sorted_assessments_put_must_haves_first(self) -> None:
        a = self.build(
            [
                RequirementAssessment(
                    requirement_id="r1",
                    requirement_text="nice",
                    kind=RequirementKind.NICE_TO_HAVE,
                    weight=50.0,
                    verdict=Verdict.NONE,
                    samples=[Verdict.NONE],
                ),
                RequirementAssessment(
                    requirement_id="r2",
                    requirement_text="must",
                    kind=RequirementKind.MUST_HAVE,
                    weight=50.0,
                    verdict=Verdict.STRONG,
                    samples=[Verdict.STRONG],
                ),
            ]
        )
        assert a.sorted_assessments()[0].requirement_text == "must"


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


class TestJudgePrompt:
    def test_prompt_contains_only_the_retrieved_evidence(self) -> None:
        """The judge must not be able to reach for unrelated resume content."""
        prompt = build_prompt(
            requirement("Has run Kubernetes in production"),
            [
                hit(
                    "u1",
                    "Deployed and operated the payments service on Kubernetes.",
                    context="Backend Engineer at Fintech Co.",
                )
            ],
        )
        assert "Kubernetes" in prompt
        assert "Kafka" not in prompt
        assert "kvstore" not in prompt

    def test_prompt_labels_evidence_with_ids(self) -> None:
        prompt = build_prompt(
            requirement("Has run Kubernetes in production"),
            [hit("u1", "Deployed and operated the payments service on Kubernetes.")],
        )
        assert "[u1]" in prompt


class TestRequirementJudge:
    async def test_unanimous_evidence_gives_a_confident_verdict(self, tmp_path: Path) -> None:
        provider = SequenceProvider([judgement("clear")])
        judge = RequirementJudge(
            LLMClient(provider, settings=settings_for(tmp_path)), settings=settings_for(tmp_path)
        )
        result = await judge.judge(
            requirement("Has run Kubernetes in production"),
            [hit("u1", "Deployed and operated the payments service on Kubernetes.")],
        )
        assert result.verdict is Verdict.CLEAR
        assert len(result.samples) == 5
        assert not result.is_ambiguous

    async def test_split_samples_produce_an_ambiguous_result(self, tmp_path: Path) -> None:
        provider = SequenceProvider([judgement("none"), judgement("partial"), judgement("strong")])
        judge = RequirementJudge(
            LLMClient(provider, settings=settings_for(tmp_path)), settings=settings_for(tmp_path)
        )
        result = await judge.judge(
            requirement("Has run Kubernetes in production"),
            [hit("u1", "Deployed and operated the payments service on Kubernetes.")],
        )
        assert result.is_ambiguous
        assert len(set(result.samples)) > 1

    async def test_samples_are_not_collapsed_by_the_cache(self, tmp_path: Path) -> None:
        """The classic way to fake self-consistency: k identical cached requests."""
        provider = SequenceProvider([judgement("clear")])
        settings = settings_for(tmp_path, cache_enabled=True)
        judge = RequirementJudge(LLMClient(provider, settings=settings), settings=settings)
        await judge.judge(
            requirement("Has run Kubernetes"),
            [hit("u1", "Deployed and operated the payments service on Kubernetes.")],
        )
        # Five distinct prompts, so five real calls rather than one plus four hits.
        assert len(provider.calls) == 5
        assert len(set(provider.calls)) == 5

    async def test_no_evidence_short_circuits_without_calling_the_model(
        self, tmp_path: Path
    ) -> None:
        provider = SequenceProvider([judgement("clear")])
        judge = RequirementJudge(
            LLMClient(provider, settings=settings_for(tmp_path)), settings=settings_for(tmp_path)
        )
        result = await judge.judge(requirement("Has run Kubernetes"), [])
        assert result.verdict is Verdict.NONE
        assert provider.calls == []

    async def test_citations_come_from_the_units_the_model_named(self, tmp_path: Path) -> None:
        provider = SequenceProvider([judgement("clear", ["u2"])])
        judge = RequirementJudge(
            LLMClient(provider, settings=settings_for(tmp_path)), settings=settings_for(tmp_path)
        )
        result = await judge.judge(
            requirement("Has used Kafka"),
            [
                hit("u1", "Deployed and operated the payments service on Kubernetes."),
                hit("u2", "Built a Kafka consumer group processing 2M settlement events per day."),
            ],
        )
        assert result.citations
        assert "Kafka" in result.citations[0].quote

    async def test_invented_unit_ids_are_dropped(self, tmp_path: Path) -> None:
        provider = SequenceProvider([judgement("clear", ["u99", "does-not-exist"])])
        judge = RequirementJudge(
            LLMClient(provider, settings=settings_for(tmp_path)), settings=settings_for(tmp_path)
        )
        result = await judge.judge(
            requirement("Has used Kafka"),
            [hit("u1", "Built a Kafka consumer group processing 2M settlement events per day.")],
        )
        # Falls back to the top hit rather than emitting a citation to nothing.
        assert len(result.citations) == 1
        assert result.citations[0].verify(SOURCE)

    async def test_a_none_verdict_carries_no_citations(self, tmp_path: Path) -> None:
        """'Nothing supports this' and 'here is the evidence' cannot both be true."""
        provider = SequenceProvider([judgement("none", ["u1"])])
        judge = RequirementJudge(
            LLMClient(provider, settings=settings_for(tmp_path)), settings=settings_for(tmp_path)
        )
        result = await judge.judge(
            requirement("Has front-end design experience"),
            [hit("u1", "Built a Kafka consumer group processing 2M settlement events per day.")],
        )
        assert result.verdict is Verdict.NONE
        assert result.citations == []

    async def test_every_citation_verifies_against_the_source(self, tmp_path: Path) -> None:
        provider = SequenceProvider([judgement("strong", ["u1"])])
        judge = RequirementJudge(
            LLMClient(provider, settings=settings_for(tmp_path)), settings=settings_for(tmp_path)
        )
        result = await judge.judge(
            requirement("Has used Kafka"),
            [hit("u1", "Built a Kafka consumer group processing 2M settlement events per day.")],
        )
        assert all(c.verify(SOURCE) for c in result.citations)


# ---------------------------------------------------------------------------
# Risks
# ---------------------------------------------------------------------------


class TestDateParsing:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("Jan 2023", 2023 * 12 + 1), ("March 2021", 2021 * 12 + 3), ("03/2022", 2022 * 12 + 3)],
    )
    def test_parses_common_formats(self, value: str, expected: int) -> None:
        assert parse_month_index(value) == expected

    def test_year_only_lands_mid_year(self) -> None:
        assert parse_month_index("2022") == 2022 * 12 + 6

    def test_present_sorts_last(self) -> None:
        assert parse_month_index("present") > parse_month_index("Dec 2030")

    def test_unparseable_returns_none(self) -> None:
        assert parse_month_index("sometime after graduation") is None
        assert parse_month_index("") is None


class TestRiskFlags:
    def test_employment_gap_is_reported_without_penalty(self) -> None:
        resume = CitedResume(
            document_id="doc-1",
            work=[
                WorkExperience(
                    company=cite("Fintech Co."),
                    start_date=Cited.inferred("Jan 2018"),
                    end_date=Cited.inferred("Dec 2019"),
                ),
                WorkExperience(
                    company=cite("Fintech Co."),
                    start_date=Cited.inferred("Jan 2023"),
                    is_current=True,
                ),
            ],
        )
        flags = detect_risks(resume)
        gap = next(f for f in flags if f.code == "employment_gap")
        assert "does not affect the score" in gap.message

    def test_no_flag_for_a_short_gap(self) -> None:
        resume = CitedResume(
            document_id="doc-1",
            work=[
                WorkExperience(
                    company=cite("Fintech Co."),
                    start_date=Cited.inferred("Jan 2022"),
                    end_date=Cited.inferred("Mar 2023"),
                ),
                WorkExperience(company=cite("Fintech Co."), start_date=Cited.inferred("Jun 2023")),
            ],
        )
        assert not [f for f in detect_risks(resume) if f.code == "employment_gap"]

    def test_projects_without_links_are_flagged(self) -> None:
        resume = CitedResume(
            document_id="doc-1",
            projects=[Project(name=cite("kvstore")), Project(name=cite("kvstore"))],
        )
        assert any(f.code == "projects_without_links" for f in detect_risks(resume))

    def test_vague_claims_are_flagged(self) -> None:
        resume = CitedResume(
            document_id="doc-1",
            work=[
                WorkExperience(
                    company=cite("Fintech Co."),
                    highlights=[
                        Cited.inferred("Was responsible for the admin tools"),
                        Cited.inferred("Helped with the migration"),
                        Cited.inferred("Involved in planning"),
                    ],
                )
            ],
        )
        assert any(f.code == "vague_claims" for f in detect_risks(resume))

    def test_low_grounding_is_a_high_risk(self) -> None:
        from hirelens.schemas.resume import GroundingStats

        resume = CitedResume(
            document_id="doc-1",
            grounding=GroundingStats(total_fields=10, grounded_fields=3),
        )
        flag = next(f for f in detect_risks(resume) if f.code == "low_grounding")
        assert flag.level.value == "high"

    def test_unmet_must_have_is_flagged_from_the_assessment(self) -> None:
        assessment = CandidateAssessment(
            document_id="doc-1",
            rubric_id="rb-1",
            assessments=[
                RequirementAssessment(
                    requirement_id="r1",
                    requirement_text="Has run Kubernetes",
                    kind=RequirementKind.MUST_HAVE,
                    weight=100.0,
                    verdict=Verdict.NONE,
                    samples=[Verdict.NONE],
                )
            ],
        )
        flags = detect_risks(CitedResume(document_id="doc-1"), assessment)
        assert any(f.code == "unmet_must_have" for f in flags)


# ---------------------------------------------------------------------------
# Interview questions
# ---------------------------------------------------------------------------


class TestGapCollection:
    def test_unmet_must_haves_come_first(self) -> None:
        assessment = CandidateAssessment(
            document_id="doc-1",
            rubric_id="rb-1",
            assessments=[
                RequirementAssessment(
                    requirement_id="r1",
                    requirement_text="Partial thing",
                    kind=RequirementKind.NICE_TO_HAVE,
                    weight=50.0,
                    verdict=Verdict.PARTIAL,
                    samples=[Verdict.PARTIAL],
                ),
                RequirementAssessment(
                    requirement_id="r2",
                    requirement_text="Critical thing",
                    kind=RequirementKind.MUST_HAVE,
                    weight=50.0,
                    verdict=Verdict.NONE,
                    samples=[Verdict.NONE],
                ),
            ],
        )
        gaps = collect_gaps(assessment, CitedResume(document_id="doc-1"))
        assert "Critical thing" in gaps[0]

    def test_no_gaps_when_everything_is_clear(self) -> None:
        assessment = CandidateAssessment(
            document_id="doc-1",
            rubric_id="rb-1",
            assessments=[
                RequirementAssessment(
                    requirement_id="r1",
                    requirement_text="All good",
                    kind=RequirementKind.MUST_HAVE,
                    weight=100.0,
                    verdict=Verdict.STRONG,
                    samples=[Verdict.STRONG] * 5,
                )
            ],
        )
        assert collect_gaps(assessment, CitedResume(document_id="doc-1")) == []


# ---------------------------------------------------------------------------
# Ranking and pipeline
# ---------------------------------------------------------------------------


def make_result(score_weight: float, verdict: Verdict, must_verdict: Verdict):
    from hirelens.assess.pipeline import ScreeningResult

    return ScreeningResult(
        assessment=CandidateAssessment(
            document_id=f"doc-{score_weight}",
            rubric_id="rb-1",
            assessments=[
                RequirementAssessment(
                    requirement_id="r1",
                    requirement_text="nice",
                    kind=RequirementKind.NICE_TO_HAVE,
                    weight=score_weight,
                    verdict=verdict,
                    samples=[verdict],
                ),
                RequirementAssessment(
                    requirement_id="r2",
                    requirement_text="must",
                    kind=RequirementKind.MUST_HAVE,
                    weight=100.0 - score_weight,
                    verdict=must_verdict,
                    samples=[must_verdict],
                ),
            ],
        ),
        resume=CitedResume(document_id="x"),
    )


class TestRanking:
    def test_must_have_compliance_beats_raw_score(self) -> None:
        high_but_incomplete = make_result(90.0, Verdict.STRONG, Verdict.NONE)
        lower_but_complete = make_result(50.0, Verdict.PARTIAL, Verdict.CLEAR)

        ordered = rank([high_but_incomplete, lower_but_complete])
        assert ordered[0] is lower_but_complete
        assert ordered[0].assessment.score < ordered[1].assessment.score


class PipelineProvider(LLMProvider):
    """Answers every stage of the pipeline based on prompt content."""

    name = "pipeline"
    model = "pipeline-model"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        convo = "\n".join(m.content for m in request.messages)
        self.prompts.append(request.messages[-1].content)

        if "Compile the following job description" in convo:
            payload = {
                "role_title": "Backend Engineer",
                "seniority": "mid",
                "requirements": [
                    {
                        "text": "Has run containers in production",
                        "kind": "must_have",
                        "category": "experience",
                        "evidence_hint": "Kubernetes deployed operated production",
                    },
                    {
                        "text": "Has front-end design experience",
                        "kind": "nice_to_have",
                        "category": "technical_skill",
                        "evidence_hint": "React CSS Figma interface design",
                    },
                ],
            }
        elif "Extract paid professional experience" in convo:
            payload = {
                "work": [
                    {
                        "company": {"value": "Fintech Co.", "quote": "Fintech Co."},
                        "position": {"value": "Backend Engineer", "quote": "Backend Engineer"},
                        "highlights": [
                            {
                                "value": "Operated payments on Kubernetes",
                                "quote": "Deployed and operated the payments service on Kubernetes.",
                            }
                        ],
                    }
                ]
            }
        elif "REQUIREMENT:" in convo:
            verdict = "clear" if "containers" in convo else "none"
            payload = {"verdict": verdict, "reasoning": "scripted", "evidence_unit_ids": []}
        elif "Uncertainties to resolve" in convo:
            payload = {
                "questions": [
                    {
                        "question": "Walk me through how you operated the payments service.",
                        "rationale": "Establishes depth on the only must-have.",
                        "targets": "r00",
                    }
                ]
            }
        else:
            payload = {}

        return CompletionResponse(content=json.dumps(payload), model=self.model, usage=Usage(1, 1))

    async def aclose(self) -> None:
        return None


JD = """
We are hiring a Backend Engineer to own our payments platform. You will design and
operate services handling millions of events per day. Requirements: experience
running containers in production. Nice to have: front-end design experience.
"""


@pytest.fixture
def resume_doc(tmp_path: Path):
    from hirelens.ingest import read_document

    path = tmp_path / "candidate.txt"
    path.write_text(SOURCE, encoding="utf-8")
    return read_document(path)


class TestScreeningPipeline:
    def build(self, tmp_path: Path) -> ScreeningPipeline:
        settings = settings_for(tmp_path, self_consistency_k=3, blind_mode=True)
        return ScreeningPipeline(
            LLMClient(PipelineProvider(), settings=settings),
            settings=settings,
            embedder=HashingEmbedder(),
        )

    async def test_end_to_end_produces_a_scored_assessment(
        self, resume_doc, tmp_path: Path
    ) -> None:
        pipeline = self.build(tmp_path)
        _rubric, results = await pipeline.screen_batch([resume_doc], JD, with_questions=True)

        assert len(results) == 1
        a = results[0].assessment
        assert a.role_title == "Backend Engineer"
        assert 0 < a.score < 100
        assert len(a.assessments) == 2

    async def test_met_and_unmet_requirements_are_distinguished(
        self, resume_doc, tmp_path: Path
    ) -> None:
        pipeline = self.build(tmp_path)
        _, results = await pipeline.screen_batch([resume_doc], JD)
        by_text = {a.requirement_text: a for a in results[0].assessment.assessments}

        assert by_text["Has run containers in production"].verdict is Verdict.CLEAR
        assert by_text["Has front-end design experience"].verdict is Verdict.NONE

    async def test_blind_mode_labels_the_candidate_anonymously(
        self, resume_doc, tmp_path: Path
    ) -> None:
        """Labelling a blinded assessment with the filename would leak identity."""
        pipeline = self.build(tmp_path)
        _, results = await pipeline.screen_batch([resume_doc], JD)
        assert results[0].assessment.candidate_label.startswith("candidate-")
        assert "candidate.txt" not in results[0].assessment.candidate_label

    async def test_interview_questions_are_generated(self, resume_doc, tmp_path: Path) -> None:
        pipeline = self.build(tmp_path)
        _, results = await pipeline.screen_batch([resume_doc], JD, with_questions=True)
        assert results[0].assessment.questions

    async def test_questions_can_be_skipped(self, resume_doc, tmp_path: Path) -> None:
        pipeline = self.build(tmp_path)
        _, results = await pipeline.screen_batch([resume_doc], JD, with_questions=False)
        assert results[0].assessment.questions == []

    async def test_the_rubric_is_compiled_once_for_the_whole_batch(
        self, resume_doc, tmp_path: Path
    ) -> None:
        """Per-candidate rubrics would make the scores incomparable."""
        provider = PipelineProvider()
        settings = settings_for(tmp_path, self_consistency_k=1)
        pipeline = ScreeningPipeline(
            LLMClient(provider, settings=settings), settings=settings, embedder=HashingEmbedder()
        )
        await pipeline.screen_batch([resume_doc, resume_doc, resume_doc], JD, with_questions=False)

        compiles = [p for p in provider.prompts if "Compile the following job description" in p]
        assert len(compiles) == 1

    async def test_assessment_carries_grounding_metrics(self, resume_doc, tmp_path: Path) -> None:
        pipeline = self.build(tmp_path)
        _, results = await pipeline.screen_batch([resume_doc], JD)
        a = results[0].assessment
        assert 0.0 <= a.grounding_rate <= 1.0
        assert a.citation_validity_rate == 1.0

    async def test_result_reports_timing_and_usage(self, resume_doc, tmp_path: Path) -> None:
        pipeline = self.build(tmp_path)
        _, results = await pipeline.screen_batch([resume_doc], JD)
        assert results[0].elapsed_s >= 0.0
        assert results[0].llm_usage["api_calls"] > 0
        assert results[0].evidence_unit_count > 0


class TestDegenerateOutputIsRejected:
    """The pipeline must refuse to score when a stage returned nothing usable.

    Both guards were written after a real incident. Swapping to a provider that
    cannot be sent a response schema produced a rubric with zero must-haves and
    an extraction with one evidence unit from an 1100-character resume. The
    pipeline reported 0 out of 100 with grounding 100%, citations valid 100% and
    agreement 100%.

    Those metrics are ratios, so they measure internal consistency and are
    trivially perfect over an almost empty set. Absolute floors are what was
    missing.
    """

    JD = (
        "Senior Backend Engineer\n\n"
        "Requirements\n"
        "- Strong experience running containerised workloads in production\n"
        "- Experience with high-throughput event streaming such as Kafka\n"
        "- A track record of measurably improving system performance\n"
    )

    @staticmethod
    def _rubric(kinds: list[RequirementKind]) -> Rubric:
        return Rubric(
            rubric_id="r1",
            role_title="Senior Backend Engineer",
            seniority="senior",
            requirements=[
                Requirement(
                    requirement_id=f"q{index}",
                    text=f"requirement {index}",
                    kind=kind,
                    category=RequirementCategory.EXPERIENCE,
                    weight=100 / max(len(kinds), 1),
                    evidence_hint="hint",
                )
                for index, kind in enumerate(kinds)
            ],
        )

    def test_rubric_with_no_must_haves_is_flagged(self) -> None:
        rubric = self._rubric([RequirementKind.NICE_TO_HAVE] * 8)
        assert "not one must-have" in RubricCompiler._degeneracy(rubric, self.JD)

    def test_a_normal_rubric_is_clean(self) -> None:
        rubric = self._rubric([RequirementKind.MUST_HAVE] * 3 + [RequirementKind.NICE_TO_HAVE] * 2)
        assert RubricCompiler._degeneracy(rubric, self.JD) == ""

    def test_a_posting_with_no_stated_requirements_may_have_no_must_haves(self) -> None:
        """Not every posting has hard requirements, and inventing them is worse."""
        casual = "We are looking for someone to help with our backend. Come talk to us."
        rubric = self._rubric([RequirementKind.NICE_TO_HAVE] * 3)
        assert RubricCompiler._degeneracy(rubric, casual) == ""

    def test_the_message_reads_as_a_correction_to_the_model(self) -> None:
        """It is fed straight back into the retry prompt, so it must be an instruction.

        Environment-variable advice belongs in the final error a person reads,
        not in a correction addressed to a model that cannot act on it.
        """
        message = RubricCompiler._degeneracy(
            self._rubric([RequirementKind.NICE_TO_HAVE] * 8), self.JD
        )
        assert "8 requirements" in message
        assert "HIRELENS_" not in message

    def test_an_empty_rubric_cannot_even_be_constructed(self) -> None:
        """The schema catches this one first, so the guard never sees it."""
        with pytest.raises(ValueError, match="at least one requirement"):
            self._rubric([])

    # -- extraction ---------------------------------------------------------

    @staticmethod
    def _document(length: int) -> SourceDocument:
        return SourceDocument(
            document_id="d1",
            filename="cv.pdf",
            source_format="pdf",
            text="x" * length,
            blocks=[],
            page_count=1,
        )

    def test_one_unit_from_a_full_resume_is_rejected(self) -> None:
        """The exact observed failure: 1 unit from 1100 characters."""
        with pytest.raises(ValueError, match="Refusing to score"):
            _reject_degenerate_extraction(self._document(1100), [object()])

    def test_a_healthy_extraction_passes(self) -> None:
        # Real resumes yield roughly one unit per 50 to 80 characters.
        _reject_degenerate_extraction(self._document(1100), [object()] * 20)

    def test_a_short_document_is_exempt(self) -> None:
        """A stub really can produce almost nothing, and that is not a bug."""
        _reject_degenerate_extraction(self._document(200), [])

    def test_the_error_names_both_numbers(self) -> None:
        """So the reader can judge the call rather than trust the threshold."""
        with pytest.raises(ValueError) as caught:
            _reject_degenerate_extraction(self._document(1200), [object()])
        message = str(caught.value)
        assert "1 evidence unit" in message and "1200-character" in message
