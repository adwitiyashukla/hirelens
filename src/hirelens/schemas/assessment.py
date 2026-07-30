"""Assessment types: verdicts, confidence bands, risk flags, and the final report.

Two decisions in this module do most of the work.

**Verdicts are ordinal, not numeric.** The judge is never asked "score this 0 to
100". It picks one of five labels, each with a written definition. Models are
poorly calibrated number generators: ask for a score out of 100 and you get 75
and 80 for indistinguishable evidence, clustering on round numbers, drifting with
prompt phrasing. Ask them to choose between "clear evidence" and "partial
evidence", each defined, and agreement with human raters rises sharply. This is
the same reason human hiring panels use anchored rating scales instead of asking
interviewers for a percentage. The numeric coefficient is applied afterwards, by
us, deterministically.

**Uncertainty is reported, not hidden.** Each requirement is judged several times
and we keep the whole distribution, not just the winner. A requirement where the
model said "clear" three times and "partial" twice is genuinely ambiguous, and the
recruiter is told so. A single point estimate would have thrown that away and
looked more confident than the evidence supports.
"""

from __future__ import annotations

import statistics

from pydantic import BaseModel, ConfigDict, Field

from hirelens._compat import StrEnum
from hirelens.schemas.evidence import Citation
from hirelens.schemas.job import RequirementKind


class Verdict(StrEnum):
    """Anchored ordinal scale. Order matters: these are ranked, not categorical."""

    STRONG = "strong"
    CLEAR = "clear"
    PARTIAL = "partial"
    WEAK = "weak"
    NONE = "none"


#: Numeric value of each verdict, applied by us rather than by the model.
#:
#: The gap between NONE and WEAK is deliberately larger than between WEAK and
#: PARTIAL. "There is no evidence at all" and "there is a hint of something" are
#: qualitatively different states for a recruiter, whereas the middle of the scale
#: is a continuum. Compressing the top end (STRONG 1.0, CLEAR 0.8) reflects that
#: exceeding a requirement is worth little more than meeting it.
VERDICT_VALUES: dict[Verdict, float] = {
    Verdict.STRONG: 1.0,
    Verdict.CLEAR: 0.8,
    Verdict.PARTIAL: 0.5,
    Verdict.WEAK: 0.2,
    Verdict.NONE: 0.0,
}

#: Ranked worst to best, so medians and comparisons are well defined.
VERDICT_ORDER: tuple[Verdict, ...] = (
    Verdict.NONE,
    Verdict.WEAK,
    Verdict.PARTIAL,
    Verdict.CLEAR,
    Verdict.STRONG,
)

#: A requirement whose samples span more than this fraction of the scale is
#: flagged for human review rather than presented as a settled answer.
AMBIGUITY_THRESHOLD = 0.3


class RawJudgement(BaseModel):
    """What the judge model returns for one requirement, one sample."""

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
    """The aggregated judgement for one requirement, with its uncertainty."""

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

    # -- scoring -------------------------------------------------------------

    @property
    def value(self) -> float:
        """Coefficient in [0, 1] for the aggregated verdict."""
        return VERDICT_VALUES[self.verdict]

    @property
    def points(self) -> float:
        """Points contributed to the overall score."""
        return self.weight * self.value

    @property
    def max_points(self) -> float:
        return self.weight

    # -- uncertainty ---------------------------------------------------------

    @property
    def value_range(self) -> tuple[float, float]:
        """Lowest and highest coefficient across samples."""
        if not self.samples:
            return (self.value, self.value)
        values = [VERDICT_VALUES[s] for s in self.samples]
        return (min(values), max(values))

    @property
    def spread(self) -> float:
        """How far apart the samples were, on the 0 to 1 scale."""
        low, high = self.value_range
        return high - low

    @property
    def is_ambiguous(self) -> bool:
        """True when the model could not make up its mind.

        Surfaced to the recruiter as "needs human review". This is a feature: a
        system that says "I am not sure about this one" is more useful, and more
        honest, than one that averages the disagreement away.
        """
        return self.spread > AMBIGUITY_THRESHOLD

    @property
    def agreement(self) -> float:
        """Fraction of samples that matched the aggregated verdict."""
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
    """Something a human should look at before interviewing.

    Risk flags never change the score. They are observations, not penalties. A
    two-year employment gap might be caregiving, illness, study, or a startup that
    failed, and deducting points for it would encode exactly the kind of bias this
    project exists to measure. The recruiter is shown the fact and decides.
    """

    model_config = ConfigDict(frozen=True)

    code: str
    level: RiskLevel
    message: str
    citations: tuple[Citation, ...] = ()


class InterviewQuestion(BaseModel):
    """A question targeted at a specific gap or unverified claim."""

    model_config = ConfigDict(frozen=True)

    question: str
    rationale: str = Field(description="Why this is worth asking this candidate")
    targets: str = Field(default="", description="Requirement id or claim it probes")


class RawInterviewPack(BaseModel):
    model_config = ConfigDict(extra="ignore")
    questions: list[InterviewQuestion] = Field(default_factory=list)


class CandidateAssessment(BaseModel):
    """Everything HireLens concluded about one candidate against one rubric."""

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

    # -- headline numbers ----------------------------------------------------

    @property
    def score(self) -> float:
        """Weighted fit score out of 100."""
        return round(sum(a.points for a in self.assessments), 1)

    @property
    def score_range(self) -> tuple[float, float]:
        """Confidence band, from the per-requirement sample spread.

        This is the interval the samples actually produced, not a statistical
        confidence interval, and the README says so. It answers "how much did the
        answer move when we asked again", which is the question a recruiter
        actually has.
        """
        low = sum(a.weight * a.value_range[0] for a in self.assessments)
        high = sum(a.weight * a.value_range[1] for a in self.assessments)
        return (round(low, 1), round(high, 1))

    @property
    def uncertainty(self) -> float:
        """Width of the confidence band, in points."""
        low, high = self.score_range
        return round(high - low, 1)

    # -- gating --------------------------------------------------------------

    @property
    def unmet_must_haves(self) -> list[RequirementAssessment]:
        """Must-haves with no real supporting evidence.

        Kept separate from the score on purpose. A candidate can score 71 and
        still be missing a hard requirement, and a single number cannot express
        that. The recruiter sees both.
        """
        return [a for a in self.assessments if a.kind is RequirementKind.MUST_HAVE and a.is_unmet]

    @property
    def meets_all_must_haves(self) -> bool:
        return not self.unmet_must_haves

    @property
    def needs_review(self) -> list[RequirementAssessment]:
        return [a for a in self.assessments if a.is_ambiguous]

    # -- stability -----------------------------------------------------------

    @property
    def mean_agreement(self) -> float:
        """Average per-requirement sample agreement. The self-consistency metric."""
        if not self.assessments:
            return 1.0
        return round(statistics.fmean(a.agreement for a in self.assessments), 3)

    # -- presentation --------------------------------------------------------

    @property
    def band(self) -> str:
        """Coarse label. Deliberately coarse: the underlying precision is not real."""
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
        """Must-haves first, then by points lost, so the biggest gaps surface."""
        return sorted(
            self.assessments,
            key=lambda a: (a.kind is not RequirementKind.MUST_HAVE, -(a.max_points - a.points)),
        )


def aggregate_verdicts(samples: list[Verdict]) -> Verdict:
    """Combine sampled verdicts into one.

    The median on the ordinal scale, not the mode and not the mean.

    The mode is unstable at small k: with five samples, two-two-one splits are
    common and the winner is decided by a single draw. The mean would require
    treating the labels as numbers before aggregating, which reintroduces exactly
    the calibration problem the ordinal scale exists to avoid. The median is
    robust to a single outlying sample, which is the failure this is guarding
    against, and it is well defined on a ranked scale.

    With an even number of samples straddling two levels we take the lower one.
    Ties should resolve conservatively: overstating a candidate's fit is the more
    costly error for the person reading the report.
    """
    if not samples:
        return Verdict.NONE
    ranks = sorted(VERDICT_ORDER.index(sample) for sample in samples)
    return VERDICT_ORDER[ranks[(len(ranks) - 1) // 2]]
