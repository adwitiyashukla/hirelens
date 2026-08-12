from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hirelens._compat import StrEnum
from hirelens.evals.profiles import Demographics


class Axis(StrEnum):
    NULL = "null"
    GENDER = "gender"
    ETHNICITY = "ethnicity"
    UNIVERSITY = "university"
    LOCATION = "location"


class Variant(BaseModel):
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


NULL_VARIANTS: tuple[Variant, ...] = (
    Variant(axis=Axis.NULL, label="control-a", group="control", overrides={}),
    Variant(axis=Axis.NULL, label="control-b", group="control", overrides={}),
)

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

DEFAULT_AXES: tuple[Axis, ...] = (
    Axis.GENDER,
    Axis.ETHNICITY,
    Axis.UNIVERSITY,
    Axis.LOCATION,
)


def variants_for(axis: Axis, *, limit: int | None = None) -> tuple[Variant, ...]:
    variants = _BY_AXIS[axis]
    return variants[:limit] if limit else variants


def build_plan(
    axes: tuple[Axis, ...] = DEFAULT_AXES,
    *,
    variants_per_axis: int | None = None,
) -> list[Variant]:
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
    per_screening = extraction_calls + requirements * self_consistency_k
    return profiles * variants * modes * per_screening
