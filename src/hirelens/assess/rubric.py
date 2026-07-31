"""Compile a free-text job description into a scoring rubric.

One LLM call, heavily constrained. The prompt does most of the work, and it is
written around three failure modes that show up immediately without it:

**Compound requirements.** Models love "3+ years of Python and experience with
distributed systems". That is two requirements, they have different evidence, and
a candidate can satisfy one and not the other. The prompt demands atomicity and
demonstrates the split.

**Restating the JD.** Left alone, a model returns the job description's bullet
points verbatim, including "competitive salary" and "we are a fast-paced team".
Those are not requirements and scoring against them is noise.

**Proxy discrimination.** A JD saying "graduate of a top-tier university" or
"native English speaker" encodes a demographic filter, not a capability. The
prompt refuses to compile those, and the refusal is logged so a recruiter can see
what was dropped and why. This is the same concern the fairness audit measures,
handled at the earliest possible point.
"""

from __future__ import annotations

import logging

from hirelens.config import Settings, get_settings
from hirelens.llm.client import LLMClient
from hirelens.schemas.job import RawRubric, RequirementKind, Rubric

logger = logging.getLogger(__name__)

SYSTEM_MESSAGE = """\
You convert job descriptions into structured, checkable hiring requirements. You \
do not evaluate candidates and you never see a resume.

Rules:

1. ATOMIC. Each requirement must be exactly one checkable capability. Split \
compound statements.
   Bad:  "3+ years of Python and experience with distributed systems"
   Good: "Has 3 or more years of professional Python experience"
         "Has built or operated distributed systems"

2. CAPABILITIES, NOT PERKS. Extract only what the candidate must be able to do or \
must have done. Ignore salary, benefits, company culture, location policy, \
application instructions, and equal-opportunity boilerplate.

3. EVIDENCE-ORIENTED. For each requirement write an "evidence_hint": a short \
phrase using the words a RESUME would use, not the words the job description \
used. This is used as a search query.
   Requirement:   "Comfortable owning services in production"
   evidence_hint: "deployed operated production service on-call incident"

4. CLASSIFY BY WHERE IT APPEARS IN THE POSTING.
   - must_have: appears under a heading such as "Requirements", "You have", \
"Essential", "What we need", or is stated as required elsewhere.
   - nice_to_have: appears under "Nice to have", "Preferred", "Desirable", or is \
described as a plus.
   - bonus: peripheral, would be a pleasant surprise.
   Both extremes are wrong. Do not mark everything must_have, and do not mark \
everything nice_to_have. A posting that has a requirements section always has at \
least one must_have. A typical role has 3 to 6 of them.

5. REFUSE PROXIES. Do NOT create requirements based on: university prestige or \
ranking, nationality, native-language status, age, gender, race, "culture fit", \
or years-since-graduation. These are demographic proxies, not capabilities. If \
the job description contains one, skip it. If an underlying capability is \
implied, extract that instead.
   JD says:  "Graduate of a top-tier engineering school"
   Extract:  nothing, unless a specific technical skill is stated elsewhere
   JD says:  "Native English speaker"
   Extract:  "Can communicate technical work clearly in written English"

6. Aim for 6 to 14 requirements. Return JSON only.\
"""


def build_prompt(job_description: str) -> str:
    return (
        "Compile the following job description into requirements.\n\n"
        "Also identify the role title and the seniority level (one of: intern, "
        "junior, mid, senior, staff, principal). Infer seniority from the "
        "responsibilities and experience asked for, and leave it empty if it is "
        "genuinely unclear.\n\n"
        f"--- BEGIN JOB DESCRIPTION ---\n{job_description.strip()}\n"
        "--- END JOB DESCRIPTION ---"
    )


# A job description shorter than this is not a job description.
_MIN_JD_CHARS = 80


class RubricCompiler:
    """Turns job description text into a :class:`Rubric`."""

    def __init__(
        self, client: LLMClient | None = None, *, settings: Settings | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or LLMClient(settings=self.settings)

    async def compile(self, job_description: str) -> Rubric:
        """Compile ``job_description`` into a weighted rubric."""
        text = job_description.strip()
        if len(text) < _MIN_JD_CHARS:
            raise ValueError(
                f"Job description is only {len(text)} characters. Provide the full "
                f"posting: the rubric is only as good as the description it is "
                f"compiled from."
            )

        rubric = await self._compile_once(text)

        # One corrective retry before giving up. The failure this handles is a
        # model marking every requirement nice_to_have, which is a reasoning slip
        # rather than a malformed response, so the schema repair loop in
        # ``structured`` never sees it. Telling the model plainly what it got
        # wrong fixes it more often than not, and one extra call is a much better
        # trade than making a person re-run the whole batch.
        problem = self._degeneracy(rubric, text)
        if problem:
            logger.warning("rubric looks wrong (%s), retrying once with a correction", problem)
            rubric = await self._compile_once(text, correction=problem)

            if remaining := self._degeneracy(rubric, text):
                raise ValueError(
                    f"{remaining}\n\n"
                    f"This persisted after a corrective retry, so the model is not "
                    f"following the classification rule. Try a different model with "
                    f"HIRELENS_GROQ_MODEL or HIRELENS_GEMINI_MODEL, or switch provider "
                    f"with HIRELENS_LLM_PROVIDER."
                )

        logger.info("compiled rubric %s: %s", rubric.rubric_id, rubric.summary())
        return rubric

    async def _compile_once(self, text: str, *, correction: str = "") -> Rubric:
        prompt = build_prompt(text)
        if correction:
            prompt += (
                f"\n\nYour previous attempt was rejected: {correction}\n"
                f"Re-read the posting's section headings. Every requirement listed "
                f"under a heading like 'Requirements' is a must_have. Only the ones "
                f"under 'Nice to have' or similar are nice_to_have."
            )

        raw = await self.client.structured(
            RawRubric,
            system=SYSTEM_MESSAGE,
            user=prompt,
            # Compilation is a structuring task, not a creative one. Determinism
            # here also means the same JD produces the same rubric across a batch,
            # which is required for candidates to be comparable at all.
            # The retry needs a different sample, though: a deterministic decode
            # would reproduce the same wrong answer exactly.
            temperature=self.settings.extraction_temperature if not correction else 0.3,
        )
        return Rubric.from_raw(raw, source_text=text)

    @staticmethod
    def _degeneracy(rubric: Rubric, job_description: str) -> str:
        """Describe what is wrong with this rubric, or return "" if nothing is.

        Returns a message rather than raising so the caller can feed it back to
        the model as a correction. The same sentence then serves twice: once as
        the retry instruction, once as the error a person reads if the retry
        also fails.

        This exists because of a specific failure. A model that could not see the
        response schema returned every requirement as ``nice_to_have``, so the
        rubric had no must-haves, every candidate trivially "met" all of them,
        and a strong candidate scored 0 out of 100 while the report showed
        grounding 100% and agreement 100%. Perfect scores over an empty set.

        The lesson is that the quality metrics measure the pipeline's internal
        consistency, not whether it understood anything, so a separate structural
        check is needed. These conditions are cheap and catch the failure at the
        point it happens rather than eight steps downstream.
        """
        # An empty rubric is not checked here: ``Rubric`` already refuses to
        # construct without at least one requirement, so this is only ever
        # reached with a non-empty list. A check for it here would be dead code
        # that reads like protection.

        # A posting with a requirements heading and no hard requirement means the
        # model flattened the distinction rather than found it absent.
        lowered = job_description.lower()
        states_requirements = any(
            marker in lowered
            for marker in ("requirement", "must have", "you will need", "essential")
        )
        must_haves = [r for r in rubric.requirements if r.kind is RequirementKind.MUST_HAVE]

        if states_requirements and not must_haves:
            return (
                f"The rubric has {len(rubric.requirements)} requirements but not one "
                f"must-have, from a posting that explicitly lists requirements. Every "
                f"requirement was put in the same category, which would make every "
                f"candidate score zero against it."
            )

        return ""
