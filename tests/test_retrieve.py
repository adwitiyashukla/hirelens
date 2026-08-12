from __future__ import annotations

import json

import pytest

from hirelens.config import Provider, Settings
from hirelens.llm.base import CompletionRequest, CompletionResponse, LLMProvider, Usage
from hirelens.llm.client import LLMClient
from hirelens.retrieve.chunking import chunk_resume, coverage, merge_overlapping
from hirelens.retrieve.embeddings import HashingEmbedder, cosine
from hirelens.retrieve.hybrid import BM25, HybridRetriever, tokenize
from hirelens.schemas.evidence import Citation, Cited, EvidenceUnit, Span
from hirelens.schemas.job import (
    RawRequirement,
    RawRubric,
    RequirementCategory,
    RequirementKind,
    Rubric,
)
from hirelens.schemas.resume import (
    Basics,
    CitedResume,
    Education,
    Project,
    Skill,
    WorkExperience,
)

DOC_ID = "doc-1"

SOURCE = """Backend Engineer, Fintech Co.
Cut p99 checkout latency from 1.2s to 180ms in the payments hot path.
Built a Kafka consumer group processing 2M settlement events per day.
Deployed and operated the reconciliation service on Kubernetes.
kvstore - Raft-based distributed key-value store in Go.
B.Tech Computer Science, NIT Trichy
Go, Python, PostgreSQL, Kafka, Kubernetes, Terraform
"""


def cite(fragment: str) -> Cited[str]:
    start = SOURCE.index(fragment)
    return Cited(
        value=fragment,
        citations=[
            Citation(
                document_id=DOC_ID,
                span=Span(start=start, end=start + len(fragment)),
                quote=fragment,
                page=1,
            )
        ],
    )


@pytest.fixture
def resume() -> CitedResume:
    return CitedResume(
        document_id=DOC_ID,
        basics=Basics(),
        work=[
            WorkExperience(
                company=cite("Fintech Co."),
                position=cite("Backend Engineer"),
                highlights=[
                    cite("Cut p99 checkout latency from 1.2s to 180ms in the payments hot path."),
                    cite("Built a Kafka consumer group processing 2M settlement events per day."),
                    cite("Deployed and operated the reconciliation service on Kubernetes."),
                ],
            )
        ],
        projects=[
            Project(
                name=cite("kvstore"),
                description=cite("Raft-based distributed key-value store in Go."),
            )
        ],
        education=[Education(institution=cite("NIT Trichy"), degree=cite("B.Tech"))],
        skills=[
            Skill(name=cite(s))
            for s in ["Python", "PostgreSQL", "Kafka", "Kubernetes", "Terraform"]
        ],
    )


def raw(text: str, kind: RequirementKind, hint: str = "") -> RawRequirement:
    return RawRequirement(
        text=text, kind=kind, category=RequirementCategory.TECHNICAL_SKILL, evidence_hint=hint
    )


class TestRubric:
    def test_weights_are_normalised_to_100(self) -> None:
        rubric = Rubric.from_raw(
            RawRubric(
                role_title="Backend Engineer",
                requirements=[
                    raw("Knows Python", RequirementKind.MUST_HAVE),
                    raw("Knows Kubernetes", RequirementKind.NICE_TO_HAVE),
                    raw("Has spoken at a conference", RequirementKind.BONUS),
                ],
            ),
            source_text="jd text",
        )
        assert sum(r.weight for r in rubric.requirements) == pytest.approx(100.0)

    def test_must_haves_outweigh_nice_to_haves(self) -> None:
        rubric = Rubric.from_raw(
            RawRubric(
                requirements=[
                    raw("Knows Python", RequirementKind.MUST_HAVE),
                    raw("Knows Rust", RequirementKind.NICE_TO_HAVE),
                ]
            ),
            source_text="jd",
        )
        must, nice = rubric.requirements
        assert must.weight > nice.weight
        assert must.is_blocking and not nice.is_blocking

    def test_rubric_size_does_not_change_the_total(self) -> None:
        small = Rubric.from_raw(
            RawRubric(requirements=[raw(f"Req {i}", RequirementKind.MUST_HAVE) for i in range(3)]),
            source_text="a",
        )
        large = Rubric.from_raw(
            RawRubric(requirements=[raw(f"Req {i}", RequirementKind.MUST_HAVE) for i in range(14)]),
            source_text="b",
        )
        assert sum(r.weight for r in small.requirements) == pytest.approx(100.0)
        assert sum(r.weight for r in large.requirements) == pytest.approx(100.0)

    def test_duplicate_requirements_are_dropped(self) -> None:
        rubric = Rubric.from_raw(
            RawRubric(
                requirements=[
                    raw("Knows Python", RequirementKind.MUST_HAVE),
                    raw("knows   PYTHON", RequirementKind.MUST_HAVE),
                    raw("Knows Go", RequirementKind.MUST_HAVE),
                ]
            ),
            source_text="jd",
        )
        assert len(rubric.requirements) == 2

    def test_empty_rubric_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="did not compile"):
            Rubric.from_raw(RawRubric(requirements=[]), source_text="jd")

    def test_query_prefers_the_evidence_hint(self) -> None:
        rubric = Rubric.from_raw(
            RawRubric(
                requirements=[
                    raw(
                        "Comfortable owning services in production",
                        RequirementKind.MUST_HAVE,
                        "deployed operated production on-call",
                    )
                ]
            ),
            source_text="jd",
        )
        assert rubric.requirements[0].query == "deployed operated production on-call"

    def test_ids_are_stable_for_the_same_jd(self) -> None:
        args = RawRubric(requirements=[raw("Knows Go", RequirementKind.MUST_HAVE)])
        first = Rubric.from_raw(args, source_text="identical jd")
        second = Rubric.from_raw(args, source_text="identical jd")
        assert first.rubric_id == second.rubric_id
        assert first.requirements[0].requirement_id == second.requirements[0].requirement_id


class ScriptedProvider(LLMProvider):
    name = "scripted"
    model = "scripted-model"

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls += 1
        return CompletionResponse(
            content=json.dumps(self.payload), model=self.model, usage=Usage(1, 1)
        )

    async def aclose(self) -> None:
        return None


JD = """
We are hiring a Backend Engineer to own our payments platform. You will design and
operate services handling millions of events per day. Requirements: strong Python,
experience with Kubernetes in production, and familiarity with event streaming.
Nice to have: Go, Terraform. Competitive salary and free lunch.
"""


class TestRubricCompiler:
    async def test_compiles_a_job_description(self, tmp_path) -> None:
        from hirelens.assess.rubric import RubricCompiler

        provider = ScriptedProvider(
            {
                "role_title": "Backend Engineer",
                "seniority": "mid",
                "requirements": [
                    {
                        "text": "Has strong professional Python experience",
                        "kind": "must_have",
                        "category": "technical_skill",
                        "evidence_hint": "Python backend services",
                    },
                    {
                        "text": "Has run Kubernetes in production",
                        "kind": "must_have",
                        "category": "experience",
                        "evidence_hint": "Kubernetes deployed production cluster",
                    },
                    {
                        "text": "Familiar with event streaming",
                        "kind": "nice_to_have",
                        "category": "technical_skill",
                        "evidence_hint": "Kafka event stream consumer",
                    },
                ],
            }
        )
        settings = Settings(
            llm_provider=Provider.OLLAMA,
            cache_enabled=False,
            cache_dir=tmp_path,
            requests_per_minute=0,
        )
        compiler = RubricCompiler(LLMClient(provider, settings=settings), settings=settings)
        rubric = await compiler.compile(JD)

        assert rubric.role_title == "Backend Engineer"
        assert len(rubric.must_haves) == 2
        assert sum(r.weight for r in rubric.requirements) == pytest.approx(100.0)

    async def test_rejects_a_stub_job_description(self, tmp_path) -> None:
        from hirelens.assess.rubric import RubricCompiler

        settings = Settings(
            llm_provider=Provider.OLLAMA,
            cache_enabled=False,
            cache_dir=tmp_path,
            requests_per_minute=0,
        )
        compiler = RubricCompiler(
            LLMClient(ScriptedProvider({}), settings=settings), settings=settings
        )
        with pytest.raises(ValueError, match=r"only \d+ characters"):
            await compiler.compile("Backend engineer wanted")


class TestChunking:
    def test_produces_units_for_every_section(self, resume: CitedResume) -> None:
        sections = {u.section for u in chunk_resume(resume)}
        assert {"work", "projects", "education", "skills"} <= sections

    def test_each_highlight_becomes_its_own_unit(self, resume: CitedResume) -> None:
        work_units = [u for u in chunk_resume(resume) if u.section == "work"]
        assert len(work_units) >= 3

    def test_units_carry_context_for_retrieval(self, resume: CitedResume) -> None:
        unit = next(u for u in chunk_resume(resume) if "p99" in u.text)
        assert "Backend Engineer" in unit.text
        assert "Fintech Co." in unit.text

    def test_span_covers_only_the_claim_not_the_context(self, resume: CitedResume) -> None:
        unit = next(u for u in chunk_resume(resume) if "p99" in u.text)
        assert SOURCE[unit.span.start : unit.span.end].startswith("Cut p99")
        assert "Backend Engineer" not in SOURCE[unit.span.start : unit.span.end]

    def test_every_unit_span_resolves_in_the_source(self, resume: CitedResume) -> None:
        for unit in chunk_resume(resume):
            assert SOURCE[unit.span.start : unit.span.end].strip()

    def test_ungrounded_values_produce_no_units(self) -> None:
        ghost = CitedResume(
            document_id=DOC_ID,
            work=[WorkExperience(company=Cited.inferred("Ghost Corp"))],
        )
        assert chunk_resume(ghost) == []

    def test_combined_skill_unit_exists(self, resume: CitedResume) -> None:
        units = chunk_resume(resume)
        assert any(u.text.startswith("Skills:") for u in units)

    def test_units_convert_back_into_citations(self, resume: CitedResume) -> None:
        for unit in chunk_resume(resume):
            citation = unit.as_citation()
            assert citation.document_id == DOC_ID
            assert citation.page == 1

    def test_every_unit_citation_verifies_against_the_source(self, resume: CitedResume) -> None:
        for unit in chunk_resume(resume):
            assert unit.as_citation().verify(SOURCE), unit.unit_id

    def test_searchable_text_is_wider_than_the_quote_where_context_applies(
        self, resume: CitedResume
    ) -> None:
        unit = next(u for u in chunk_resume(resume) if "p99" in u.text)
        assert "Backend Engineer" in unit.text
        assert "Backend Engineer" not in unit.claim
        assert unit.claim == SOURCE[unit.span.start : unit.span.end]

    def test_merge_overlapping_drops_contained_spans(self) -> None:
        outer = EvidenceUnit(
            unit_id="a", document_id=DOC_ID, text="outer", span=Span(start=0, end=50)
        )
        inner = EvidenceUnit(
            unit_id="b", document_id=DOC_ID, text="inner", span=Span(start=10, end=20)
        )
        assert [u.unit_id for u in merge_overlapping([outer, inner])] == ["a"]

    def test_coverage_is_a_fraction(self, resume: CitedResume) -> None:
        value = coverage(chunk_resume(resume), len(SOURCE))
        assert 0.0 < value <= 1.0


class TestHashingEmbedder:
    def test_is_deterministic(self) -> None:
        a = HashingEmbedder().embed_one("Kubernetes in production")
        b = HashingEmbedder().embed_one("Kubernetes in production")
        assert a == b

    def test_vectors_are_unit_length(self) -> None:
        vector = HashingEmbedder().embed_one("distributed systems in Go")
        assert sum(v * v for v in vector) == pytest.approx(1.0, abs=1e-6)

    def test_identical_text_is_maximally_similar(self) -> None:
        embedder = HashingEmbedder()
        vector = embedder.embed_one("Kafka event streaming")
        assert cosine(vector, vector) == pytest.approx(1.0)

    def test_shared_vocabulary_scores_above_unrelated_text(self) -> None:
        embedder = HashingEmbedder()
        query = embedder.embed_one("Kubernetes production deployment")
        related = embedder.embed_one("Deployed the service on Kubernetes in production")
        unrelated = embedder.embed_one("Wrote poetry about autumn leaves")
        assert cosine(query, related) > cosine(query, unrelated)

    def test_character_ngrams_survive_morphology(self) -> None:
        embedder = HashingEmbedder()
        assert cosine(
            embedder.embed_one("deployment pipeline"), embedder.embed_one("deploying pipelines")
        ) > cosine(embedder.embed_one("deployment pipeline"), embedder.embed_one("marketing copy"))

    def test_empty_batch(self) -> None:
        assert HashingEmbedder().embed([]) == []


class TestCosine:
    def test_orthogonal_vectors_score_zero(self) -> None:
        assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_is_safe(self) -> None:
        assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_mismatched_dimensions_are_safe(self) -> None:
        assert cosine([1.0], [1.0, 2.0]) == 0.0


class TestTokenize:
    def test_keeps_technology_punctuation(self) -> None:
        tokens = tokenize("Built in C++ and C# with .NET and Node.js")
        assert "c++" in tokens
        assert "c#" in tokens
        assert ".net" in tokens
        assert "node.js" in tokens

    def test_drops_stopwords(self) -> None:
        assert "the" not in tokenize("the service in the cluster")


class TestBM25:
    def test_ranks_the_matching_document_first(self) -> None:
        bm25 = BM25(
            [
                "Wrote documentation and ran onboarding sessions",
                "Deployed the payments service on Kubernetes",
                "Designed a marketing landing page",
            ]
        )
        scores = bm25.scores("Kubernetes deployment")
        assert scores.index(max(scores)) == 1

    def test_absent_terms_score_zero(self) -> None:
        bm25 = BM25(["Python backend service"])
        assert bm25.scores("underwater basket weaving") == [0.0]

    def test_idf_is_never_negative(self) -> None:
        bm25 = BM25(["python service", "python api", "python worker"])
        assert all(value >= 0 for value in bm25.idf.values())

    def test_empty_corpus_is_safe(self) -> None:
        assert BM25([]).scores("anything") == []


class TestHybridRetriever:
    def build(self, resume: CitedResume) -> HybridRetriever:
        return HybridRetriever(units=chunk_resume(resume), embedder=HashingEmbedder())

    def test_exact_technology_term_is_retrieved(self, resume: CitedResume) -> None:
        hits = self.build(resume).search("Kubernetes", top_k=3)
        assert hits
        assert any("Kubernetes" in hit.unit.text for hit in hits)

    def test_retrieves_the_right_bullet_for_a_performance_requirement(
        self, resume: CitedResume
    ) -> None:
        hits = self.build(resume).search("latency performance optimisation p99", top_k=3)
        assert any("p99" in hit.unit.text for hit in hits)

    def test_results_are_ordered_by_fused_score(self, resume: CitedResume) -> None:
        hits = self.build(resume).search("Kafka event streaming", top_k=5)
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_is_respected(self, resume: CitedResume) -> None:
        assert len(self.build(resume).search("Go", top_k=2)) <= 2

    def test_hits_record_which_retriever_found_them(self, resume: CitedResume) -> None:
        hits = self.build(resume).search("Kubernetes production", top_k=3)
        assert hits
        assert all(hit.found_by != "none" for hit in hits)
        assert any("lexical" in hit.found_by for hit in hits)

    def test_fusion_beats_a_single_ranker_on_a_split_query(self, resume: CitedResume) -> None:
        hits = self.build(resume).search("Kafka streaming and reducing checkout latency", top_k=4)
        text = " ".join(hit.unit.text for hit in hits)
        assert "Kafka" in text
        assert "p99" in text

    def test_hits_convert_into_verifiable_citations(self, resume: CitedResume) -> None:
        hits = self.build(resume).search("Kubernetes", top_k=1)
        citation = hits[0].unit.as_citation()
        assert citation.verify(SOURCE)

    def test_nonsense_query_returns_nothing(self, resume: CitedResume) -> None:
        assert self.build(resume).search("zzzz qqqq wwww", top_k=5) == []

    def test_empty_index_is_safe(self) -> None:
        retriever = HybridRetriever(units=[], embedder=HashingEmbedder())
        assert retriever.search("anything") == []

    def test_search_many_keys_by_requirement(self, resume: CitedResume) -> None:
        results = self.build(resume).search_many(
            {"r01": "Kubernetes production", "r02": "distributed systems Raft"}, top_k=2
        )
        assert set(results) == {"r01", "r02"}
        assert all(len(hits) <= 2 for hits in results.values())
