from __future__ import annotations

import logging

from hirelens.config import Settings, get_settings
from hirelens.llm.base import LLMError
from hirelens.llm.client import LLMClient
from hirelens.schemas.assessment import (
    CandidateAssessment,
    InterviewQuestion,
    RawInterviewPack,
    Verdict,
)
from hirelens.schemas.resume import CitedResume

logger = logging.getLogger(__name__)

SYSTEM_MESSAGE = """\
You write interview questions for a hiring manager preparing a first-round \
conversation. You are given specific uncertainties about one candidate, and you \
write questions that would resolve them.

Rules:

1. SPECIFIC, NOT GENERIC. Every question must reference the actual gap or claim \
you were given. Never write "Tell me about a challenging project" or "What are \
your strengths".
2. ONE UNCERTAINTY PER QUESTION.
3. OPEN AND NEUTRAL. Ask what they did, how it worked, what they would change. \
Do not write leading or sceptical questions, and never imply the candidate is \
exaggerating. The interviewer needs information, not a cross-examination.
4. NEVER ask about age, family, nationality, visa status, health, religion, \
politics, salary history, or anything about a university's reputation. These are \
unlawful or discriminatory in many jurisdictions and are irrelevant regardless.
5. For each question give a one-line rationale explaining what it would resolve, \
addressed to the interviewer.
6. Write 4 to 7 questions. Return JSON only.\
"""


def build_prompt(gaps: list[str], role_title: str) -> str:
    listed = "\n".join(f"- {gap}" for gap in gaps)
    role = role_title or "the role"
    return (
        f"The candidate is being considered for: {role}\n\n"
        f"Uncertainties to resolve in the interview:\n{listed}\n\n"
        f"Write questions that would resolve these specific uncertainties."
    )


_MAX_GAPS = 8


def collect_gaps(assessment: CandidateAssessment, resume: CitedResume) -> list[str]:
    gaps: list[str] = []

    for item in assessment.unmet_must_haves:
        gaps.append(
            f"Must-have '{item.requirement_text}' has no clear supporting evidence in the resume."
        )

    for item in assessment.assessments:
        if item.verdict is Verdict.PARTIAL:
            gaps.append(f"'{item.requirement_text}' has partial evidence only: {item.reasoning}")

    for item in assessment.needs_review:
        if not any(item.requirement_text in gap for gap in gaps):
            gaps.append(
                f"'{item.requirement_text}' produced inconsistent assessments; the "
                f"evidence is borderline."
            )

    for quote in resume.grounding.unlocatable_quotes[:2]:
        gaps.append(f"This claim could not be verified against the document text: '{quote}'")

    vague = [
        highlight.value
        for job in resume.work
        for highlight in job.highlights
        if not any(character.isdigit() for character in highlight.value)
    ]
    for claim in vague[:2]:
        gaps.append(f"Achievement stated without a measurable outcome: '{claim}'")

    return gaps[:_MAX_GAPS]


class InterviewPackGenerator:
    def __init__(
        self, client: LLMClient | None = None, *, settings: Settings | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or LLMClient(settings=self.settings)

    async def generate(
        self, assessment: CandidateAssessment, resume: CitedResume
    ) -> list[InterviewQuestion]:
        gaps = collect_gaps(assessment, resume)
        if not gaps:
            logger.debug("no gaps worth probing for %s", assessment.document_id)
            return []

        try:
            pack = await self.client.structured(
                RawInterviewPack,
                system=SYSTEM_MESSAGE,
                user=build_prompt(gaps, assessment.role_title),
                temperature=max(self.settings.judge_temperature, 0.4),
            )
        except LLMError as exc:
            logger.warning("interview pack generation failed: %s", exc)
            return []

        return pack.questions[:7]
