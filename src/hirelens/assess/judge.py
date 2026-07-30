"""Judge one requirement against only the evidence retrieved for it.

The scoping is the whole design. A conventional resume scorer puts the entire
resume and the entire rubric in one prompt and asks for a verdict on everything.
That prompt has three problems:

* **Hallucination surface.** With the whole resume in context, a model asked about
  Kubernetes can find *something* to say by reaching for an unrelated bullet. With
  only four retrieved chunks in context, there is nothing to reach for.
* **Contamination.** One badly-judged requirement drags the others with it,
  because the model is producing them in a single autoregressive pass and
  conditioning each on the last.
* **Cost and parallelism.** One large sequential call instead of many small
  concurrent ones, and no way to retry a single bad requirement.

So each call sees: one requirement, its definition, and the handful of evidence
units the retriever surfaced. Nothing else. Not the candidate's name, not their
university, not the other requirements, not the overall score so far.

Each requirement is judged ``k`` times at a non-zero temperature. This is
**self-consistency sampling**: the spread across samples is a usable signal about
how ambiguous the evidence actually is, and the median is more robust than any
single draw. Temperature zero would produce a fake confidence interval of width
zero, which is worse than no interval at all.
"""

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
    """The user turn: one requirement, and only its evidence."""
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
    """Scores requirements against retrieved evidence."""

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
        """Judge one requirement, sampling ``k`` times."""
        k = k if k is not None else self.settings.self_consistency_k

        # Nothing retrieved means nothing to judge. Returning "none" directly
        # saves k API calls per empty requirement, which across a batch is a
        # large share of a free-tier quota spent confirming an empty list is
        # empty.
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
            # Every sample failed. Record it as ambiguous rather than scoring
            # zero: a provider outage is not evidence about the candidate.
            return RequirementAssessment(
                requirement_id=requirement.requirement_id,
                requirement_text=requirement.text,
                kind=requirement.kind,
                weight=requirement.weight,
                verdict=Verdict.NONE,
                samples=[Verdict.NONE, Verdict.STRONG],  # maximal spread = flagged
                reasoning="Judging failed for this requirement. Human review required.",
            )

        samples = [j.verdict for j in judgements]
        verdict = aggregate_verdicts(samples)

        # Explain using a sample that agreed with the aggregate, so the reasoning
        # shown to the user matches the verdict shown to the user.
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
        """Judge every requirement concurrently.

        The client's semaphore bounds real concurrency, so this stays inside
        free-tier rate limits even with a dozen requirements times k samples.
        """
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
        """One sample. The seed varies so the cache does not collapse all k calls.

        Without a per-sample seed, every sample would be an identical request,
        hit the same cache entry, and return the same verdict k times. The
        confidence band would then always be zero: a completely fake measurement
        of stability. This is the single easiest way to accidentally fake
        self-consistency, so it is worth being explicit about.
        """
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
    """Turn the cited unit ids back into verified citations.

    Three rules:

    * A ``none`` verdict carries no citations, whatever the model returned.
      "Nothing here supports the requirement" and "here is the supporting
      evidence" cannot both be true, and models do sometimes fill in the ids
      field out of habit. Displaying a citation next to a zero score would
      undermine the one property the whole report is built on.
    * Unit ids the model invented are dropped. They refer to nothing.
    * A positive verdict with no usable ids falls back to the top hit, so a score
      above zero is never left with no evidence attached at all.
    """
    if judgement.verdict is Verdict.NONE:
        return []

    by_id = {hit.unit.unit_id: hit.unit for hit in hits}
    units = [by_id[uid] for uid in judgement.evidence_unit_ids if uid in by_id]

    if not units and hits:
        units = [hits[0].unit]

    return [unit.as_citation() for unit in units]
