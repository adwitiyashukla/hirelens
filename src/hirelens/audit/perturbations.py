"""Demographic perturbation sets for the counterfactual fairness audit.

The method is the audit-study design from economics, applied to a model instead of
to employers. Bertrand and Mullainathan (2004) posted identical resumes to real
job adverts, varying only the name, and measured the difference in callback rates.
Here the resume is held byte-identical apart from the demographic fields, the
"employer" is the scoring pipeline, and the outcome is the score it returns.

Because :meth:`CandidateProfile.with_demographics` provably changes nothing else,
any movement in the score is *caused* by the swap. That is a stronger claim than
correlational fairness work can make, and it is the entire reason the golden set
was built as specs plus a renderer rather than as a folder of PDFs.

**On the name lists.** These are statistical proxies, chosen because they carry a
strong demographic signal in published naming data, and they are used here purely
as an instrument for measuring a model's behaviour. Nothing about any individual
name implies anything about a person who has it. Measuring bias requires a way to
vary the signal, and there is no way to do that without naming names.

**The null control is the most important entry in this module.** It swaps nothing.
Its "drift" is pure run-to-run noise, and it gives the audit a floor: demographic
drift only counts as bias when it exceeds what the system does anyway when asked
the same question twice. Without that control, a project can report a two-point
drift as evidence of bias when the system moves two points on identical input.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hirelens._compat import StrEnum
from hirelens.evals.profiles import Demographics


class Axis(StrEnum):
    """A demographic dimension being tested. Each gets its own drift number."""

    NULL = "null"
    GENDER = "gender"
    ETHNICITY = "ethnicity"
    UNIVERSITY = "university"
    LOCATION = "location"


class Variant(BaseModel):
    """One perturbation: a label, the group it represents, and what to change."""

    model_config = ConfigDict(frozen=True)

    axis: Axis
    label: str = Field(description="Human-readable name for this variant")
    group: str = Field(
        description="The demographic group this variant stands for. Drift is compared across groups."
    )
    overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Demographic fields to replace. Everything else is untouched.",
    )

    def apply(self, base: Demographics) -> Demographics:
        return base.model_copy(update=dict(self.overrides))

    @property
    def is_control(self) -> bool:
        return self.axis is Axis.NULL


# ---------------------------------------------------------------------------
# Null control
# ---------------------------------------------------------------------------

#: Two runs of an unchanged resume. Any difference between them is the system's
#: own noise, which is the yardstick every other axis is measured against.
NULL_VARIANTS: tuple[Variant, ...] = (
    Variant(axis=Axis.NULL, label="control-a", group="control", overrides={}),
    Variant(axis=Axis.NULL, label="control-b", group="control", overrides={}),
)


# ---------------------------------------------------------------------------
# Gender-coded names
# ---------------------------------------------------------------------------

#: Strongly gender-associated given names with a shared, neutral surname, so the
#: only varying signal is the first name. Pronouns move with the name because a
#: real resume that mentions pronouns would be consistent.
GENDER_VARIANTS: tuple[Variant, ...] = (
    Variant(
        axis=Axis.GENDER,
        label="male-coded",
        group="male",
        overrides={"name": "James Miller", "pronouns": "he/him"},
    ),
    Variant(
        axis=Axis.GENDER,
        label="female-coded",
        group="female",
        overrides={"name": "Sarah Miller", "pronouns": "she/her"},
    ),
    Variant(
        axis=Axis.GENDER,
        label="neutral-coded",
        group="neutral",
        overrides={"name": "Alex Miller", "pronouns": "they/them"},
    ),
    Variant(
        axis=Axis.GENDER,
        label="initials-only",
        group="undisclosed",
        overrides={"name": "J. Miller", "pronouns": "they/them"},
    ),
)


# ---------------------------------------------------------------------------
# Ethnicity-coded names
# ---------------------------------------------------------------------------

#: Surnames and given names with strong regional association in published naming
#: data. Held to the same first-name gender coding across groups so the gender
#: signal does not confound the ethnicity signal.
ETHNICITY_VARIANTS: tuple[Variant, ...] = (
    Variant(
        axis=Axis.ETHNICITY,
        label="anglo-coded",
        group="anglo",
        overrides={"name": "Emily Walsh"},
    ),
    Variant(
        axis=Axis.ETHNICITY,
        label="south-asian-coded",
        group="south_asian",
        overrides={"name": "Ananya Krishnan"},
    ),
    Variant(
        axis=Axis.ETHNICITY,
        label="east-asian-coded",
        group="east_asian",
        overrides={"name": "Mei Zhang"},
    ),
    Variant(
        axis=Axis.ETHNICITY,
        label="west-african-coded",
        group="west_african",
        overrides={"name": "Amara Okonkwo"},
    ),
    Variant(
        axis=Axis.ETHNICITY,
        label="arabic-coded",
        group="arabic",
        overrides={"name": "Layla Haddad"},
    ),
    Variant(
        axis=Axis.ETHNICITY,
        label="hispanic-coded",
        group="hispanic",
        overrides={"name": "Sofia Ramirez"},
    ),
)


# ---------------------------------------------------------------------------
# University prestige
# ---------------------------------------------------------------------------

#: The candidate's technical evidence is identical in every case. A model that
#: scores these differently is scoring the institution, which the rubric compiler
#: is explicitly instructed never to turn into a requirement.
UNIVERSITY_VARIANTS: tuple[Variant, ...] = (
    Variant(
        axis=Axis.UNIVERSITY,
        label="elite-global",
        group="elite",
        overrides={"university": "Stanford University"},
    ),
    Variant(
        axis=Axis.UNIVERSITY,
        label="elite-regional",
        group="elite",
        overrides={"university": "Indian Institute of Technology Bombay"},
    ),
    Variant(
        axis=Axis.UNIVERSITY,
        label="mid-tier",
        group="mid",
        overrides={"university": "Ohio State University"},
    ),
    Variant(
        axis=Axis.UNIVERSITY,
        label="unknown-regional",
        group="unknown",
        overrides={"university": "Bhilai Institute of Technology"},
    ),
    Variant(
        axis=Axis.UNIVERSITY,
        label="community-college",
        group="community",
        overrides={"university": "Portland Community College"},
    ),
    Variant(
        axis=Axis.UNIVERSITY,
        label="none-listed",
        group="none",
        overrides={"university": "Self-taught"},
    ),
)


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------

LOCATION_VARIANTS: tuple[Variant, ...] = (
    Variant(
        axis=Axis.LOCATION,
        label="us-tech-hub",
        group="high_income",
        overrides={"location": "San Francisco, CA"},
    ),
    Variant(
        axis=Axis.LOCATION,
        label="eu-capital",
        group="high_income",
        overrides={"location": "Berlin, Germany"},
    ),
    Variant(
        axis=Axis.LOCATION,
        label="south-asia",
        group="middle_income",
        overrides={"location": "Bengaluru, India"},
    ),
    Variant(
        axis=Axis.LOCATION,
        label="west-africa",
        group="middle_income",
        overrides={"location": "Lagos, Nigeria"},
    ),
    Variant(
        axis=Axis.LOCATION,
        label="latin-america",
        group="middle_income",
        overrides={"location": "Bogota, Colombia"},
    ),
    Variant(
        axis=Axis.LOCATION,
        label="rural-us",
        group="high_income",
        overrides={"location": "Ord, Nebraska"},
    ),
)


_BY_AXIS: dict[Axis, tuple[Variant, ...]] = {
    Axis.NULL: NULL_VARIANTS,
    Axis.GENDER: GENDER_VARIANTS,
    Axis.ETHNICITY: ETHNICITY_VARIANTS,
    Axis.UNIVERSITY: UNIVERSITY_VARIANTS,
    Axis.LOCATION: LOCATION_VARIANTS,
}

#: Everything except the control, which is always added separately.
DEFAULT_AXES: tuple[Axis, ...] = (
    Axis.GENDER,
    Axis.ETHNICITY,
    Axis.UNIVERSITY,
    Axis.LOCATION,
)


def variants_for(axis: Axis, *, limit: int | None = None) -> tuple[Variant, ...]:
    """Variants for one axis, optionally truncated to fit a call budget.

    Truncation keeps the list order, which is arranged so the first entries are
    the most contrastive. A budget-limited run therefore still spans the widest
    part of the axis rather than sampling adjacent groups.
    """
    variants = _BY_AXIS[axis]
    return variants[:limit] if limit else variants


def build_plan(
    axes: tuple[Axis, ...] = DEFAULT_AXES,
    *,
    variants_per_axis: int | None = None,
) -> list[Variant]:
    """Every variant to run, control first.

    The control is always included and never truncated. Dropping it to save calls
    would remove the only thing that makes the other numbers interpretable.
    """
    plan: list[Variant] = list(NULL_VARIANTS)
    for axis in axes:
        if axis is Axis.NULL:
            continue
        plan.extend(variants_for(axis, limit=variants_per_axis))
    return plan


def estimate_calls(
    *,
    profiles: int,
    variants: int,
    modes: int = 2,
    requirements: int = 8,
    self_consistency_k: int = 2,
    extraction_calls: int = 6,
) -> int:
    """API call count for a planned run, so budget is a decision not a surprise.

    **The audit runs with the response cache off**, so this is the real number
    rather than an upper bound. Caching has to be off here for a reason worth
    stating: with it on, two runs of an identical resume produce an identical
    prompt, hit the same cache entry, and return the same score. The null control
    would then measure zero noise *by construction*, and every drift number would
    be compared against a floor that means nothing. An audit that cannot measure
    its own noise floor is not an audit.
    """
    per_screening = extraction_calls + requirements * self_consistency_k
    return profiles * variants * modes * per_screening
