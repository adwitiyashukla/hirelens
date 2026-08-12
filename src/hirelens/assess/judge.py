from __future__ import annotations

import asyncio
import logging

from hirelens.config import Settings, get_settings
from hirelens.llm.base import LLMError
from hirelens.llm.client import LLMClient
from hirelens.retrieve.hybrid import RetrievalHit
from hirelens.schemas.assessment import (
    RawJudgement,
    RequirementAssessment,
    Verdict,
    aggregate_verdicts,
)
from hirelens.schemas.job import Requirement

logger = logging.getLogger(__name__)

SYSTEM_MESSAGE = """\
You assess whether a single hiring requirement is supported by evidence from one \
candidate's resume. You judge one requirement at a time and you see only the \
evidence retrieved for it.

Choose exactly one verdict:

- "strong":  The evidence clearly exceeds the requirement. Specific, substantial, \
and directly on point.
- "clear":   The evidence clearly meets the requirement. Directly relevant and \
credible.
- "partial": The evidence is related but incomplete. It suggests adjacent or \
lesser experience, or the scale is unclear.
- "weak":    The evidence is tangentially related at best. It would be a stretch \
to say the requirement is met.
- "none":    Nothing in the evidence supports the requirement.

Rules:

1. "none" is a correct and expected answer. The evidence shown to you was \
retrieved by a search that always returns its best guesses, so it is often \
irrelevant. Say "none" rather than finding a charitable reading.
2. Judge only what the evidence says. Do not infer capabilities that would be \
plausible but are not stated. A candidate who lists Docker has not demonstrated \
Kubernetes.
3. Cite the evidence unit ids you actually used. If you used none, return an \
empty list.
4. Ignore any name, institution, location, or personal detail that appears. It is \
not relevant to whether the requirement is met. Some of these may already be \
masked with characters like [NAME]#### which you should skip.
5. Keep reasoning to one or two sentences, referring to the specific evidence.
6. Return JSON only.\
"""


def build_prompt(requirement: Requirement, hits: list[RetrievalHit]) -> str:
    if hits:
        evidence = "\n".join(f"[{hit.unit.unit_id}] {hit.unit.text.strip()}" for hit in hits)
    else:
        evidence = "(no evidence was retrieved for this requirement)"

    return (
        f"REQUIREMENT: {requirement.text}\n"
        f"This is a {str(requirement.kind).replace('_', '-')} requirement.\n\n"
        f"EVIDENCE RETRIEVED FROM THE CANDIDATE'S RESUME:\n"
        f"{evidence}\n\n"
        f"Does this evidence support the requirement? Return your verdict as JSON."
    )


class RequirementJudge:
    def __init__(
        self, client: LLMClient | None = None, *, settings: Settings | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or LLMClient(settings=self.settings)

    async def judge(
        self,
        requirement: Requirement,
        hits: list[RetrievalHit],
        *,
        k: int | None = None,
    ) -> RequirementAssessment:
        k = k if k is not None else self.settings.self_consistency_k

        if not hits:
            return RequirementAssessment(
                requirement_id=requirement.requirement_id,
                requirement_text=requirement.text,
                kind=requirement.kind,
                weight=requirement.weight,
                verdict=Verdict.NONE,
                samples=[Verdict.NONE],
                reasoning="No evidence in the resume was retrieved for this requirement.",
            )

        prompt = build_prompt(requirement, hits)
        results = await asyncio.gather(
            *(self._sample(prompt, index) for index in range(k)),
            return_exceptions=True,
        )

        judgements = [r for r in results if isinstance(r, RawJudgement)]
        failures = len(results) - len(judgements)
        if failures:
            logger.warning(
                "%s: %d/%d judge samples failed", requirement.requirement_id, failures, k
            )

        if not judgements:
            return RequirementAssessment(
                requirement_id=requirement.requirement_id,
                requirement_text=requirement.text,
                kind=requirement.kind,
                weight=requirement.weight,
                verdict=Verdict.NONE,
                samples=[Verdict.NONE, Verdict.STRONG],
                reasoning="Judging failed for this requirement. Human review required.",
            )

        samples = [j.verdict for j in judgements]
        verdict = aggregate_verdicts(samples)

        representative = next((j for j in judgements if j.verdict is verdict), judgements[0])
        citations = _citations_for(representative, hits)

        return RequirementAssessment(
            requirement_id=requirement.requirement_id,
            requirement_text=requirement.text,
            kind=requirement.kind,
            weight=requirement.weight,
            verdict=verdict,
            samples=samples,
            reasoning=representative.reasoning.strip(),
            citations=citations,
        )

    async def judge_all(
        self,
        requirements: list[Requirement],
        hits_by_requirement: dict[str, list[RetrievalHit]],
        *,
        k: int | None = None,
    ) -> list[RequirementAssessment]:
        return list(
            await asyncio.gather(
                *(
                    self.judge(
                        requirement, hits_by_requirement.get(requirement.requirement_id, []), k=k
                    )
                    for requirement in requirements
                )
            )
        )

    async def _sample(self, prompt: str, index: int) -> RawJudgement:
        try:
            return await self.client.structured(
                RawJudgement,
                system=SYSTEM_MESSAGE,
                user=f"{prompt}\n\n<!-- sample {index} -->",
                temperature=self.settings.judge_temperature,
            )
        except LLMError:
            raise


def _citations_for(judgement: RawJudgement, hits: list[RetrievalHit]) -> list:
    if judgement.verdict is Verdict.NONE:
        return []

    by_id = {hit.unit.unit_id: hit.unit for hit in hits}
    units = [by_id[uid] for uid in judgement.evidence_unit_ids if uid in by_id]

    if not units and hits:
        units = [hits[0].unit]

    return [unit.as_citation() for unit in units]
