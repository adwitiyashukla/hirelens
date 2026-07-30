"""Job descriptions compiled into scoring rubrics.

The single biggest functional difference between HireLens and a typical resume
scorer is that the rubric is **derived from the job description at runtime**
rather than hardcoded. A hardcoded rubric ("open source 35, self projects 30,
production 25, technical skills 10") is a rubric for exactly one role at exactly
one company, and scoring a data analyst or a designer against it is meaningless.

A compiled rubric is a list of **atomic requirements**. Atomic matters: "5 years
of Python and experience leading a team" is two requirements with different
evidence and possibly different answers, and collapsing them into one produces a
score nobody can act on. Each requirement carries:

* a **weight**, so a must-have counts for more than a nice-to-have,
* a **kind**, so a missing must-have can gate the candidate rather than just
  costing points,
* an **evidence hint**, a short phrase used as the retrieval query, which is what
  connects the rubric to the resume.

Weights are normalised at construction so every rubric sums to 100 regardless of
how many requirements the model produced. Without that, a JD that compiles into
twelve requirements would score systematically differently from one that compiles
into six, and cross-role comparisons would be nonsense.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hirelens._compat import StrEnum


class RequirementKind(StrEnum):
    MUST_HAVE = "must_have"
    NICE_TO_HAVE = "nice_to_have"
    BONUS = "bonus"


class RequirementCategory(StrEnum):
    """Coarse grouping, used only for presentation and reporting."""

    TECHNICAL_SKILL = "technical_skill"
    EXPERIENCE = "experience"
    DOMAIN = "domain"
    EDUCATION = "education"
    SOFT_SKILL = "soft_skill"
    OTHER = "other"


#: Default relative weights by kind, applied before normalisation. A must-have is
#: worth roughly three times a nice-to-have, which matches how human screeners
#: describe their own weighting when asked.
_KIND_WEIGHTS: dict[RequirementKind, float] = {
    RequirementKind.MUST_HAVE: 3.0,
    RequirementKind.NICE_TO_HAVE: 1.0,
    RequirementKind.BONUS: 0.5,
}


class RawRequirement(BaseModel):
    """One requirement as the model returns it, before normalisation."""

    model_config = ConfigDict(extra="ignore")

    text: str = Field(
        min_length=3,
        description="One single checkable requirement, phrased as a capability",
    )
    kind: RequirementKind = RequirementKind.NICE_TO_HAVE
    category: RequirementCategory = RequirementCategory.OTHER
    evidence_hint: str = Field(
        default="",
        description=(
            "A short phrase describing what evidence in a resume would satisfy this. "
            "Used as a search query, so use the words a resume would use."
        ),
    )


class RawRubric(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role_title: str = ""
    seniority: str = Field(default="", description="intern, junior, mid, senior, staff")
    requirements: list[RawRequirement] = Field(default_factory=list)


class Requirement(BaseModel):
    """A normalised, weighted requirement ready to be judged."""

    model_config = ConfigDict(frozen=True)

    requirement_id: str
    text: str
    kind: RequirementKind
    category: RequirementCategory
    weight: float = Field(gt=0, description="Share of the total rubric, summing to 100")
    evidence_hint: str

    @property
    def query(self) -> str:
        """The text used to retrieve evidence for this requirement.

        The hint is preferred because it is phrased in resume vocabulary. A JD says
        "must be comfortable owning services in production"; a resume says
        "deployed and operated the payments service". Searching with the hint
        retrieves the bullet; searching with the JD sentence often does not.
        """
        return self.evidence_hint or self.text

    @property
    def is_blocking(self) -> bool:
        return self.kind is RequirementKind.MUST_HAVE


class Rubric(BaseModel):
    """A compiled job description."""

    model_config = ConfigDict(frozen=True)

    rubric_id: str
    role_title: str
    seniority: str
    requirements: tuple[Requirement, ...]
    source_text: str = ""

    @model_validator(mode="after")
    def _check_weights(self) -> Rubric:
        if not self.requirements:
            raise ValueError("a rubric needs at least one requirement")
        total = sum(r.weight for r in self.requirements)
        if abs(total - 100.0) > 0.5:
            raise ValueError(f"requirement weights must sum to 100, got {total:.2f}")
        return self

    # -- views ---------------------------------------------------------------

    def of_kind(self, kind: RequirementKind) -> list[Requirement]:
        return [r for r in self.requirements if r.kind is kind]

    @property
    def must_haves(self) -> list[Requirement]:
        return self.of_kind(RequirementKind.MUST_HAVE)

    def by_id(self, requirement_id: str) -> Requirement | None:
        return next((r for r in self.requirements if r.requirement_id == requirement_id), None)

    def summary(self) -> str:
        counts = {kind: len(self.of_kind(kind)) for kind in RequirementKind}
        parts = [
            f"{count} {str(kind).replace('_', ' ')}" for kind, count in counts.items() if count
        ]
        return f"{self.role_title or 'role'}: {', '.join(parts)}"

    # -- construction --------------------------------------------------------

    @classmethod
    def from_raw(cls, raw: RawRubric, *, source_text: str = "") -> Rubric:
        """Normalise a model-produced rubric into a scoreable one.

        Deduplicates on normalised text (models routinely emit "Python" twice with
        slightly different phrasing), assigns stable ids, and rescales weights to
        sum to 100.
        """
        seen: set[str] = set()
        kept: list[RawRequirement] = []
        for item in raw.requirements:
            key = " ".join(item.text.lower().split())
            if key and key not in seen:
                seen.add(key)
                kept.append(item)

        if not kept:
            raise ValueError(
                "The job description did not compile into any requirements. It may be "
                "too short or may not describe a role."
            )

        rubric_id = hashlib.sha256((source_text or raw.role_title).encode()).hexdigest()[:12]
        raw_weights = [_KIND_WEIGHTS[item.kind] for item in kept]
        scale = 100.0 / sum(raw_weights)

        requirements = tuple(
            Requirement(
                requirement_id=f"{rubric_id}-r{index:02d}",
                text=item.text.strip(),
                kind=item.kind,
                category=item.category,
                weight=round(weight * scale, 4),
                evidence_hint=item.evidence_hint.strip(),
            )
            for index, (item, weight) in enumerate(zip(kept, raw_weights, strict=True))
        )

        # Rounding can leave the total a hair off 100; absorb the drift into the
        # first requirement rather than letting the validator reject a valid rubric.
        drift = 100.0 - sum(r.weight for r in requirements)
        if abs(drift) > 1e-9:
            first = requirements[0]
            requirements = (
                first.model_copy(update={"weight": round(first.weight + drift, 4)}),
                *requirements[1:],
            )

        return cls(
            rubric_id=rubric_id,
            role_title=raw.role_title.strip(),
            seniority=raw.seniority.strip(),
            requirements=requirements,
            source_text=source_text,
        )
