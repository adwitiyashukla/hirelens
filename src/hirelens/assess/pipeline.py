from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence

from pydantic import BaseModel, Field

from hirelens.assess.judge import RequirementJudge
from hirelens.assess.questions import InterviewPackGenerator
from hirelens.assess.risks import detect_risks
from hirelens.assess.rubric import RubricCompiler
from hirelens.config import Settings, get_settings
from hirelens.extract.extractor import ResumeExtractor
from hirelens.ingest.document import SourceDocument
from hirelens.llm.client import LLMClient
from hirelens.retrieve.chunking import chunk_resume, merge_overlapping
from hirelens.retrieve.embeddings import Embedder, get_embedder
from hirelens.retrieve.hybrid import HybridRetriever
from hirelens.schemas.assessment import CandidateAssessment
from hirelens.schemas.job import Rubric
from hirelens.schemas.resume import CitedResume

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 4

_MIN_CHARS_TO_EXPECT_EVIDENCE = 400

_CHARS_PER_EXPECTED_UNIT = 400


def _reject_degenerate_extraction(document: SourceDocument, units: Sequence[object]) -> None:
    length = len(document.text)
    if length < _MIN_CHARS_TO_EXPECT_EVIDENCE:
        return

    expected = max(2, length // _CHARS_PER_EXPECTED_UNIT)
    if len(units) >= expected:
        return

    raise ValueError(
        f"Extraction produced {len(units)} evidence unit(s) from a "
        f"{length}-character document, where at least {expected} were expected. "
        f"The model returned almost nothing usable, so any score computed from "
        f"this would be a confident zero built on no evidence.\n\n"
        f"Refusing to score rather than reporting that. Try again, or switch "
        f"provider with HIRELENS_LLM_PROVIDER."
    )


class ScreeningResult(BaseModel):
    assessment: CandidateAssessment
    resume: CitedResume
    evidence_unit_count: int = 0
    elapsed_s: float = 0.0
    llm_usage: dict[str, object] = Field(default_factory=dict)


class ScreeningPipeline:
    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or LLMClient(settings=self.settings)
        self._embedder = embedder

        self.extractor = ResumeExtractor(self.client, settings=self.settings)
        self.compiler = RubricCompiler(self.client, settings=self.settings)
        self.judge = RequirementJudge(self.client, settings=self.settings)
        self.questions = InterviewPackGenerator(self.client, settings=self.settings)

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder(self.settings.embedding_model)
        return self._embedder

    async def compile_rubric(self, job_description: str) -> Rubric:
        return await self.compiler.compile(job_description)

    async def screen(
        self,
        document: SourceDocument,
        rubric: Rubric,
        *,
        top_k: int = DEFAULT_TOP_K,
        with_questions: bool = True,
        blind: bool | None = None,
    ) -> ScreeningResult:
        started = time.perf_counter()

        extraction = await self.extractor.extract(document, blind=blind)
        resume = extraction.resume

        units = merge_overlapping(chunk_resume(resume))
        _reject_degenerate_extraction(document, units)

        retriever = HybridRetriever(units=units, embedder=self.embedder)
        hits = retriever.search_many(
            {r.requirement_id: r.query for r in rubric.requirements}, top_k=top_k
        )

        assessments = await self.judge.judge_all(list(rubric.requirements), hits)

        assessment = CandidateAssessment(
            document_id=document.document_id,
            candidate_label=self._label(document, blind),
            rubric_id=rubric.rubric_id,
            role_title=rubric.role_title,
            assessments=assessments,
            grounding_rate=resume.grounding.grounding_rate,
            citation_validity_rate=resume.grounding.citation_validity_rate,
        )
        assessment.risks = detect_risks(resume, assessment)

        if with_questions:
            assessment.questions = await self.questions.generate(assessment, resume)

        return ScreeningResult(
            assessment=assessment,
            resume=resume,
            evidence_unit_count=len(units),
            elapsed_s=round(time.perf_counter() - started, 2),
            llm_usage=self.client.usage_summary(),
        )

    async def screen_batch(
        self,
        documents: list[SourceDocument],
        job_description: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        with_questions: bool = True,
    ) -> tuple[Rubric, list[ScreeningResult]]:
        rubric = await self.compile_rubric(job_description)

        results = await asyncio.gather(
            *(
                self.screen(document, rubric, top_k=top_k, with_questions=with_questions)
                for document in documents
            ),
            return_exceptions=True,
        )

        successes: list[ScreeningResult] = []
        for document, result in zip(documents, results, strict=True):
            if isinstance(result, BaseException):
                logger.error("screening failed for %s: %s", document.filename, result)
            else:
                successes.append(result)

        return rubric, rank(successes)

    def _label(self, document: SourceDocument, blind: bool | None) -> str:
        is_blind = self.settings.blind_mode if blind is None else blind
        return f"candidate-{document.document_id[:8]}" if is_blind else document.filename

    async def aclose(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> ScreeningPipeline:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


def rank(results: list[ScreeningResult]) -> list[ScreeningResult]:
    return sorted(
        results,
        key=lambda r: (not r.assessment.meets_all_must_haves, -r.assessment.score),
    )
