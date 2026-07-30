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
from hirelens.schemas.job import RawRubric, Rubric

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

4. CLASSIFY HONESTLY.
   - must_have: the JD states or clearly implies this is required
   - nice_to_have: described as preferred, desirable, or a plus
   - bonus: peripheral, would be a pleasant surprise
   Do not mark everything must_have. A typical role has 3 to 6 genuine must-haves.

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

        raw = await self.client.structured(
            RawRubric,
            system=SYSTEM_MESSAGE,
            user=build_prompt(text),
            # Compilation is a structuring task, not a creative one. Determinism
            # here also means the same JD produces the same rubric across a batch,
            # which is required for candidates to be comparable at all.
            temperature=self.settings.extraction_temperature,
        )

        rubric = Rubric.from_raw(raw, source_text=text)
        logger.info("compiled rubric %s: %s", rubric.rubric_id, rubric.summary())
        return rubric
