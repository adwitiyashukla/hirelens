from __future__ import annotations

import statistics

from pydantic import BaseModel, ConfigDict, Field

from hirelens._compat import StrEnum
from hirelens.schemas.evidence import Citation
from hirelens.schemas.job import RequirementKind


class Verdict(StrEnum):
    STRONG = "strong"
    CLEAR = "clear"
    PARTIAL = "partial"
    WEAK = "weak"
    NONE = "none"


VERDICT_VALUES: dict[Verdict, float] = {
    Verdict.STRONG: 1.0,
    Verdict.CLEAR: 0.8,
    Verdict.PARTIAL: 0.5,
    Verdict.WEAK: 0.2,
    Verdict.NONE: 0.0,
}

VERDICT_ORDER: tuple[Verdict, ...] = (
    Verdict.NONE,
    Verdict.WEAK,
    Verdict.PARTIAL,
    Verdict.CLEAR,
    Verdict.STRONG,
)

AMBIGUITY_THRESHOLD = 0.3


class RawJudgement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    verdict: Verdict = Field(description="One of: strong, clear, partial, weak, none")
    reasoning: str = Field(
        default="",
        max_length=600,
        description="One or two sentences citing the specific evidence used",
    )
    evidence_unit_ids: list[str] = Field(
        default_factory=list,
        description="Ids of the evidence units that justify this verdict. Empty if none did.",
    )


class RequirementAssessment(BaseModel):
    requirement_id: str
    requirement_text: str
    kind: RequirementKind
    weight: float
    verdict: Verdict
    samples: list[Verdict] = Field(
        default_factory=list, description="Every sampled verdict, kept for auditability"
    )
    reasoning: str = ""
    citations: list[Citation] = Field(default_factory=list)

    @property
    def value(self) -> float:
        return VERDICT_VALUES[self.verdict]

    @property
    def points(self) -> float:
        return self.weight * self.value

    @property
    def max_points(self) -> float:
        return self.weight

    @property
    def value_range(self) -> tuple[float, float]:
        if not self.samples:
            return (self.value, self.value)
        values = [VERDICT_VALUES[s] for s in self.samples]
        return (min(values), max(values))

    @property
    def spread(self) -> float:
        low, high = self.value_range
        return high - low

    @property
    def is_ambiguous(self) -> bool:
        return self.spread > AMBIGUITY_THRESHOLD

    @property
    def agreement(self) -> float:
        if not self.samples:
            return 1.0
        return sum(1 for s in self.samples if s is self.verdict) / len(self.samples)

    @property
    def is_met(self) -> bool:
        return self.verdict in (Verdict.STRONG, Verdict.CLEAR)

    @property
    def is_unmet(self) -> bool:
        return self.verdict in (Verdict.NONE, Verdict.WEAK)


class RiskLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RiskFlag(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    level: RiskLevel
    message: str
    citations: tuple[Citation, ...] = ()


class InterviewQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    rationale: str = Field(description="Why this is worth asking this candidate")
    targets: str = Field(default="", description="Requirement id or claim it probes")


class RawInterviewPack(BaseModel):
    model_config = ConfigDict(extra="ignore")
    questions: list[InterviewQuestion] = Field(default_factory=list)


class CandidateAssessment(BaseModel):
    document_id: str
    candidate_label: str = Field(
        default="",
        description="Filename or blinded id. Never the candidate's name in blind mode.",
    )
    rubric_id: str
    role_title: str = ""
    assessments: list[RequirementAssessment] = Field(default_factory=list)
    risks: list[RiskFlag] = Field(default_factory=list)
    questions: list[InterviewQuestion] = Field(default_factory=list)
    grounding_rate: float = 1.0
    citation_validity_rate: float = 1.0

    @property
    def score(self) -> float:
        return round(sum(a.points for a in self.assessments), 1)

    @property
    def score_range(self) -> tuple[float, float]:
        low = sum(a.weight * a.value_range[0] for a in self.assessments)
        high = sum(a.weight * a.value_range[1] for a in self.assessments)
        return (round(low, 1), round(high, 1))

    @property
    def uncertainty(self) -> float:
        low, high = self.score_range
        return round(high - low, 1)

    @property
    def unmet_must_haves(self) -> list[RequirementAssessment]:
        return [a for a in self.assessments if a.kind is RequirementKind.MUST_HAVE and a.is_unmet]

    @property
    def meets_all_must_haves(self) -> bool:
        return not self.unmet_must_haves

    @property
    def needs_review(self) -> list[RequirementAssessment]:
        return [a for a in self.assessments if a.is_ambiguous]

    @property
    def mean_agreement(self) -> float:
        if not self.assessments:
            return 1.0
        return round(statistics.fmean(a.agreement for a in self.assessments), 3)

    @property
    def band(self) -> str:
        if not self.meets_all_must_haves:
            return "missing a must-have"
        if self.score >= 75:
            return "strong fit"
        if self.score >= 55:
            return "possible fit"
        if self.score >= 35:
            return "weak fit"
        return "not a fit"

    def summary(self) -> str:
        low, high = self.score_range
        return (
            f"{self.score:.0f}/100 (band {low:.0f} to {high:.0f}), {self.band}, "
            f"agreement {self.mean_agreement:.0%}"
        )

    def sorted_assessments(self) -> list[RequirementAssessment]:
        return sorted(
            self.assessments,
            key=lambda a: (a.kind is not RequirementKind.MUST_HAVE, -(a.max_points - a.points)),
        )


def aggregate_verdicts(samples: list[Verdict]) -> Verdict:
    if not samples:
        return Verdict.NONE
    ranks = sorted(VERDICT_ORDER.index(sample) for sample in samples)
    return VERDICT_ORDER[ranks[(len(ranks) - 1) // 2]]
