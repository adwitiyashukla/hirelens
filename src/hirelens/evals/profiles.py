"""Candidate profiles as data, and a deterministic renderer that turns them into resumes.

The golden set is not a folder of PDFs. It is a set of small structured specs plus
a renderer, and that choice pays for itself three times:

**Version control.** A profile is a few lines of JSON in git. You can diff it,
review a change to it, and see exactly what varied between two eval runs. Binary
PDFs give you none of that.

**Reproducibility.** The renderer is deterministic, so the same spec produces
byte-identical text every time. The document id is content-addressed, so the LLM
response cache hits, so ``make eval`` twice in a row gives the same numbers. A
metric you cannot reproduce is not a metric.

**Counterfactual perturbation.** This is the one that matters most. Demographic
attributes (``name``, ``pronouns``, ``university``, ``location``) are separate
fields from everything that describes the candidate's ability. Phase 6 swaps only
those fields, re-renders, and re-scores. Because nothing else in the document can
possibly change, any score movement is *caused* by the demographic swap. With a
folder of real PDFs you could never make that claim.

Synthetic rather than real, for the obvious reason: putting real resumes in a
public repository is a privacy problem, and a scraped set would carry whatever
demographic skew the source had straight into the fairness numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from hirelens._compat import StrEnum


class QualityTier(StrEnum):
    """How well the profile was written to fit its target role.

    This is the generator's intent, not ground truth. The human labels are ground
    truth. The tier exists so the set can be built to span a deliberate range
    rather than accidentally clustering, and so a badly miscalibrated system is
    visible immediately.
    """

    STRONG = "strong"
    SOLID = "solid"
    MIXED = "mixed"
    WEAK = "weak"
    MISMATCHED = "mismatched"


class Demographics(BaseModel):
    """The attributes the fairness audit perturbs. Nothing here indicates ability."""

    model_config = ConfigDict(frozen=True)

    name: str
    pronouns: str = Field(default="they/them", description="they/them, she/her, he/him")
    university: str = "State University"
    location: str = "Remote"
    email: str = ""

    @property
    def contact_email(self) -> str:
        if self.email:
            return self.email
        handle = self.name.lower().replace(" ", ".").replace("'", "")
        return f"{handle}@example.com"


class Role(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    company: str
    start: str
    end: str = "present"
    bullets: tuple[str, ...] = ()


class ProfileProject(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    url: str = ""
    bullets: tuple[str, ...] = ()


class CandidateProfile(BaseModel):
    """Everything needed to render one synthetic resume."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    demographics: Demographics
    headline: str = ""
    degree: str = "B.S. Computer Science"
    graduation_year: str = "2021"
    github: str = ""
    roles: tuple[Role, ...] = ()
    projects: tuple[ProfileProject, ...] = ()
    skills: tuple[str, ...] = ()
    awards: tuple[str, ...] = ()
    quality: QualityTier = QualityTier.MIXED
    target_role: str = ""
    notes: str = Field(default="", description="Why this profile exists. Never rendered.")

    # -- perturbation --------------------------------------------------------

    def with_demographics(self, demographics: Demographics) -> CandidateProfile:
        """A copy differing only in demographic attributes.

        The whole counterfactual audit rests on this method changing nothing else.
        """
        return self.model_copy(update={"demographics": demographics})

    # -- rendering -----------------------------------------------------------

    def render(self) -> str:
        """Deterministic plain-text resume.

        Plain text rather than PDF on purpose: the ingestion layer is tested
        separately on real PDFs, and generating PDFs here would add a rendering
        step whose layout quirks could move eval numbers for reasons unrelated to
        the pipeline being measured.
        """
        d = self.demographics
        lines: list[str] = [d.name.upper()]

        contact = [d.contact_email, d.location]
        if self.github:
            contact.append(f"github.com/{self.github}")
        lines.append(" | ".join(part for part in contact if part))

        if self.headline:
            lines += ["", self.headline]

        if self.roles:
            lines += ["", "EXPERIENCE"]
            for role in self.roles:
                lines.append(f"{role.title}, {role.company} ({role.start} - {role.end})")
                lines += list(role.bullets)
                lines.append("")
            lines.pop()

        if self.projects:
            lines += ["", "PROJECTS"]
            for project in self.projects:
                suffix = f" - {project.url}" if project.url else ""
                lines.append(f"{project.name}: {project.description}{suffix}")
                lines += list(project.bullets)

        lines += ["", "EDUCATION", f"{self.degree}, {d.university} ({self.graduation_year})"]

        if self.skills:
            lines += ["", "SKILLS", ", ".join(self.skills)]

        if self.awards:
            lines += ["", "AWARDS"]
            lines += list(self.awards)

        return "\n".join(lines) + "\n"

    def write(self, directory: Path) -> Path:
        """Render to a .txt file and return the path."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.candidate_id}.txt"
        path.write_text(self.render(), encoding="utf-8")
        return path


class JobSpec(BaseModel):
    """A job description in the golden set."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    title: str
    text: str


class GoldenSet(BaseModel):
    """Profiles plus jobs. Labels live separately so they survive regeneration."""

    profiles: tuple[CandidateProfile, ...]
    jobs: tuple[JobSpec, ...]

    def profile(self, candidate_id: str) -> CandidateProfile | None:
        return next((p for p in self.profiles if p.candidate_id == candidate_id), None)

    def job(self, job_id: str) -> JobSpec | None:
        return next((j for j in self.jobs if j.job_id == job_id), None)

    @property
    def pair_count(self) -> int:
        return len(self.profiles) * len(self.jobs)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> GoldenSet:
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
