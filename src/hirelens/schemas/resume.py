from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hirelens.schemas.evidence import Cited, VerificationResult


class RawField(BaseModel):
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
        return self.url is not None and bool(self.url.value.strip())


class Skill(BaseModel):
    name: Cited[str]
    category: str = ""


class Award(BaseModel):
    title: Cited[str]
    awarder: Cited[str] | None = None
    date: Cited[str] | None = None


class GroundingStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_fields: int = 0
    grounded_fields: int = 0
    total_citations: int = 0
    valid_citations: int = 0
    unlocatable_quotes: list[str] = Field(default_factory=list)

    @property
    def grounding_rate(self) -> float:
        return self.grounded_fields / self.total_fields if self.total_fields else 0.0

    @property
    def citation_validity_rate(self) -> float:
        return self.valid_citations / self.total_citations if self.total_citations else 1.0

    def summary(self) -> str:
        return (
            f"{self.grounded_fields}/{self.total_fields} fields grounded "
            f"({self.grounding_rate:.0%}), "
            f"{self.valid_citations}/{self.total_citations} citations valid "
            f"({self.citation_validity_rate:.0%})"
        )


class CitedResume(BaseModel):
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

    def all_cited_values(self) -> list[Cited[str]]:
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
        result = VerificationResult(total=0, valid=0)
        for value in self.all_cited_values():
            result = result + value.verify(document_text)
        return result

    @property
    def skill_names(self) -> list[str]:
        return [s.name.value for s in self.skills]

    @property
    def is_empty(self) -> bool:
        return not any([self.work, self.education, self.projects, self.skills])
