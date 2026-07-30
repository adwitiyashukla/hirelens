"""The structured resume, with every field carrying its evidence.

Two schemas live here and the distinction matters.

**Raw extraction schemas** (``Raw*``) are what we ask the model for. They contain
plain values plus a ``quote`` string: the verbatim text the model claims it read
the value from. They deliberately do *not* contain character offsets, because
language models cannot count characters. Asking for offsets produces confident,
wrong numbers roughly half the time.

**Cited schemas** (``Cited*``) are what we hand downstream. We build them by taking
each ``quote`` and locating it in the source document ourselves, with a real string
search. The model's job is to say *what* it read; finding *where* it is is our job,
and a computer is much better at it.

That split is the reason the citation validity rate in this project is high rather
than aspirational.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hirelens.schemas.evidence import Cited, VerificationResult

# ---------------------------------------------------------------------------
# Raw extraction schemas: what the model returns
# ---------------------------------------------------------------------------


class RawField(BaseModel):
    """A single extracted value plus the text it was read from."""

    model_config = ConfigDict(extra="ignore")

    value: str = Field(description="The extracted value, normalised")
    quote: str = Field(
        default="",
        description=(
            "The exact verbatim text from the resume that this value was read from. "
            "Copy it character for character. Leave empty if inferred rather than read."
        ),
    )


class RawProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    network: str = Field(description="GitHub, LinkedIn, Twitter, personal site, ...")
    url: str
    quote: str = ""


class RawBasics(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: RawField | None = None
    email: RawField | None = None
    phone: RawField | None = None
    location: RawField | None = None
    headline: RawField | None = Field(
        default=None, description="Self-described role, e.g. 'Backend Engineer'"
    )
    profiles: list[RawProfile] = Field(default_factory=list)


class RawWork(BaseModel):
    model_config = ConfigDict(extra="ignore")

    company: RawField
    position: RawField | None = None
    start_date: RawField | None = None
    end_date: RawField | None = Field(
        default=None, description="Omit entirely if the role is current"
    )
    is_current: bool = False
    highlights: list[RawField] = Field(
        default_factory=list, description="One entry per bullet point"
    )


class RawEducation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    institution: RawField
    degree: RawField | None = None
    field_of_study: RawField | None = None
    start_date: RawField | None = None
    end_date: RawField | None = None
    score: RawField | None = Field(default=None, description="GPA or CGPA as written")


class RawProject(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: RawField
    description: RawField | None = None
    url: RawField | None = None
    technologies: list[RawField] = Field(default_factory=list)
    highlights: list[RawField] = Field(default_factory=list)


class RawSkill(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: RawField
    category: str = Field(
        default="", description="languages, frameworks, tools, cloud, ... if stated"
    )


class RawAward(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: RawField
    awarder: RawField | None = None
    date: RawField | None = None


# Section wrappers. Each extraction call targets exactly one of these, so a model
# that struggles with one section cannot corrupt the others.
#
# Every field here has a default, so a bare ``{}`` is valid. That is deliberate:
# "this resume has no awards section" is a correct answer, and a schema that
# rejected it would burn three repair attempts arguing with a model that was
# right the first time.


class RawBasicsSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    basics: RawBasics = Field(default_factory=RawBasics)


class RawWorkSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    work: list[RawWork] = Field(default_factory=list)


class RawEducationSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    education: list[RawEducation] = Field(default_factory=list)


class RawProjectsSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    projects: list[RawProject] = Field(default_factory=list)


class RawSkillsSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    skills: list[RawSkill] = Field(default_factory=list)


class RawAwardsSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    awards: list[RawAward] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Cited schemas: what the rest of the system consumes
# ---------------------------------------------------------------------------


class Profile(BaseModel):
    network: Cited[str]
    url: Cited[str]

    @property
    def is_github(self) -> bool:
        return "github" in self.network.value.lower() or "github.com" in self.url.value.lower()


class Basics(BaseModel):
    name: Cited[str] | None = None
    email: Cited[str] | None = None
    phone: Cited[str] | None = None
    location: Cited[str] | None = None
    headline: Cited[str] | None = None
    profiles: list[Profile] = Field(default_factory=list)

    def github_url(self) -> str | None:
        return next((p.url.value for p in self.profiles if p.is_github), None)


class WorkExperience(BaseModel):
    company: Cited[str]
    position: Cited[str] | None = None
    start_date: Cited[str] | None = None
    end_date: Cited[str] | None = None
    is_current: bool = False
    highlights: list[Cited[str]] = Field(default_factory=list)


class Education(BaseModel):
    institution: Cited[str]
    degree: Cited[str] | None = None
    field_of_study: Cited[str] | None = None
    start_date: Cited[str] | None = None
    end_date: Cited[str] | None = None
    score: Cited[str] | None = None


class Project(BaseModel):
    name: Cited[str]
    description: Cited[str] | None = None
    url: Cited[str] | None = None
    technologies: list[Cited[str]] = Field(default_factory=list)
    highlights: list[Cited[str]] = Field(default_factory=list)

    @property
    def has_link(self) -> bool:
        """Projects with no link are much harder to verify, and the rubric says so."""
        return self.url is not None and bool(self.url.value.strip())


class Skill(BaseModel):
    name: Cited[str]
    category: str = ""


class Award(BaseModel):
    title: Cited[str]
    awarder: Cited[str] | None = None
    date: Cited[str] | None = None


class GroundingStats(BaseModel):
    """How much of this resume is actually backed by verified evidence.

    Reported per-parse and aggregated by the evaluation harness into the
    "citation validity rate" metric. A drop here is the earliest warning that a
    prompt change has made the model start inventing things.
    """

    model_config = ConfigDict(frozen=True)

    total_fields: int = 0
    grounded_fields: int = 0
    total_citations: int = 0
    valid_citations: int = 0
    unlocatable_quotes: list[str] = Field(default_factory=list)

    @property
    def grounding_rate(self) -> float:
        """Fraction of extracted fields that carry any citation at all."""
        return self.grounded_fields / self.total_fields if self.total_fields else 0.0

    @property
    def citation_validity_rate(self) -> float:
        """Fraction of citations whose span really contains the quoted text."""
        return self.valid_citations / self.total_citations if self.total_citations else 1.0

    def summary(self) -> str:
        return (
            f"{self.grounded_fields}/{self.total_fields} fields grounded "
            f"({self.grounding_rate:.0%}), "
            f"{self.valid_citations}/{self.total_citations} citations valid "
            f"({self.citation_validity_rate:.0%})"
        )


class CitedResume(BaseModel):
    """A fully parsed resume where every value points back at the source."""

    document_id: str
    basics: Basics = Field(default_factory=Basics)
    work: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    awards: list[Award] = Field(default_factory=list)
    grounding: GroundingStats = Field(default_factory=GroundingStats)
    failed_sections: list[str] = Field(
        default_factory=list,
        description="Sections whose extraction failed. Recorded, never silently dropped.",
    )

    # -- traversal -----------------------------------------------------------

    def all_cited_values(self) -> list[Cited[str]]:
        """Every ``Cited`` field in the resume, flattened.

        Used for grounding statistics and for the fairness harness, which needs to
        walk every value to find the ones worth perturbing.
        """
        out: list[Cited[str]] = []

        def push(*values: Cited[str] | None) -> None:
            out.extend(v for v in values if v is not None)

        b = self.basics
        push(b.name, b.email, b.phone, b.location, b.headline)
        for profile in b.profiles:
            push(profile.network, profile.url)
        for job in self.work:
            push(job.company, job.position, job.start_date, job.end_date)
            out.extend(job.highlights)
        for edu in self.education:
            push(
                edu.institution,
                edu.degree,
                edu.field_of_study,
                edu.start_date,
                edu.end_date,
                edu.score,
            )
        for project in self.projects:
            push(project.name, project.description, project.url)
            out.extend(project.technologies)
            out.extend(project.highlights)
        for skill in self.skills:
            push(skill.name)
        for award in self.awards:
            push(award.title, award.awarder, award.date)
        return out

    def verify(self, document_text: str) -> VerificationResult:
        """Re-verify every citation against the source. Cheap, so do it freely."""
        result = VerificationResult(total=0, valid=0)
        for value in self.all_cited_values():
            result = result + value.verify(document_text)
        return result

    # -- convenience ---------------------------------------------------------

    @property
    def skill_names(self) -> list[str]:
        return [s.name.value for s in self.skills]

    @property
    def is_empty(self) -> bool:
        return not any([self.work, self.education, self.projects, self.skills])
