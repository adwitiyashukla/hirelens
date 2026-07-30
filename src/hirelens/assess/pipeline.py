"""The end-to-end screening pipeline.

Document plus job description in, :class:`CandidateAssessment` out.

Two things this orchestrator is careful about.

**The rubric is compiled once per batch, not once per candidate.** Compiling per
candidate would produce slightly different requirements for each one, and
comparing scores across candidates would then be meaningless. A batch shares one
rubric object; this is a correctness property, not an optimisation.

**One LLM client is shared across every stage.** Extraction, rubric compilation,
judging and question generation all draw on the same cache, the same concurrency
semaphore and the same token accounting. That is what keeps a batch of resumes
inside a free-tier per-minute quota, and what makes the usage numbers in the eval
harness real rather than per-stage guesses.
"""

from __future__ import annotations

import asyncio
import logging
import time

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

#: Evidence units retrieved per requirement. Small on purpose: the judge's
#: reliability comes from seeing little enough that it cannot wander, and beyond
#: about five chunks the marginal one is almost always noise.
DEFAULT_TOP_K = 4


class ScreeningResult(BaseModel):
    """One candidate's assessment plus the artefacts that produced it."""

    assessment: CandidateAssessment
    resume: CitedResume
    evidence_unit_count: int = 0
    elapsed_s: float = 0.0
    llm_usage: dict[str, object] = Field(default_factory=dict)


class ScreeningPipeline:
    """Runs the full screen for one or many candidates against one rubric."""

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
        """Loaded lazily so ``hirelens parse`` never pays for a model it will not use."""
        if self._embedder is None:
            self._embedder = get_embedder(self.settings.embedding_model)
        return self._embedder

    # -- public API ----------------------------------------------------------

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
        """Screen one candidate against an already-compiled rubric."""
        started = time.perf_counter()

        extraction = await self.extractor.extract(document, blind=blind)
        resume = extraction.resume

        units = merge_overlapping(chunk_resume(resume))
        retriever = HybridRetriever(units=units, embedder=self.embedder)
        hits = retriever.search_many(
            {r.requirement_id: r.query for r in rubric.requirements}, top_k=top_k
        )

        assessments = await self.judge.judge_all(list(rubric.requirements), hits)

        assessment = CandidateAssessment(
            document_id=document.document_id,
            # Blind mode must reach the report too. Labelling a blinded assessment
            # with the candidate's filename would leak the identity we just spent
            # the extraction stage removing.
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
        """Screen many candidates against one job description, then rank them."""
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
                # One unreadable resume must not lose the whole batch.
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
    """Order candidates for a shortlist.

    Candidates meeting every must-have come first regardless of score, because a
    75 that is missing a hard requirement is not better than a 62 that meets them
    all. Within each group, by score.
    """
    return sorted(
        results,
        key=lambda r: (not r.assessment.meets_all_must_haves, -r.assessment.score),
    )
