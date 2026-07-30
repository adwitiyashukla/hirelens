"""Split a parsed resume into retrievable evidence units.

The unit of retrieval is one **claim**, not one page and not one section. A
requirement like "has operated services in production" should match a single
bullet point, and the highlight the recruiter sees should be that bullet, not the
whole Experience section.

Chunking from the *cited* resume rather than from raw text is what makes this
cheap: extraction already found the claim boundaries and already resolved each
one to a span, so a chunk is just a cited value with context attached.

Context attachment is the one subtlety. A bullet reading "Reduced p99 latency to
180ms" retrieves poorly on its own, because the terms a job description uses
("performance optimisation", "backend") do not appear in it. So each unit's
searchable text is the claim plus its parent context ("Backend Engineer at Fintech
Co."), while the span still points at just the claim. The retriever sees enough to
match; the recruiter sees exactly the sentence that mattered.
"""

from __future__ import annotations

from hirelens.schemas.evidence import Cited, EvidenceUnit, Span
from hirelens.schemas.resume import CitedResume


def chunk_resume(resume: CitedResume) -> list[EvidenceUnit]:
    """Produce every retrievable evidence unit in ``resume``.

    Only grounded values become units. An ungrounded value has no span, so it
    could never be highlighted, and retrieving it would produce a citation that
    points nowhere.
    """
    units: list[EvidenceUnit] = []
    counter = _Counter(resume.document_id)

    _add_work(resume, units, counter)
    _add_projects(resume, units, counter)
    _add_education(resume, units, counter)
    _add_skills(resume, units, counter)
    _add_awards(resume, units, counter)
    _add_headline(resume, units, counter)

    return units


# ---------------------------------------------------------------------------


class _Counter:
    """Issues stable, readable unit ids."""

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        self._n = 0

    def next_id(self, section: str) -> str:
        self._n += 1
        return f"{self.document_id}-{section}-{self._n:03d}"


def _unit(
    cited: Cited[str],
    *,
    section: str,
    counter: _Counter,
    context: str = "",
    text_override: str | None = None,
) -> EvidenceUnit | None:
    """Build one unit from a cited value, or None if it is not grounded."""
    if not cited.citations:
        return None
    citation = cited.citations[0]

    claim = text_override if text_override is not None else citation.quote.strip()
    if not claim:
        return None

    # Context is prepended for retrieval only. The span still covers just the
    # claim, so the highlight stays tight and `quote` stays verifiable.
    searchable = f"{context} {claim}".strip() if context else claim

    return EvidenceUnit(
        unit_id=counter.next_id(section),
        document_id=citation.document_id,
        text=searchable,
        # The exact document text at the span, taken from the citation rather than
        # from the model's value, so it verifies by construction.
        quote=citation.quote,
        span=citation.span,
        section=section,
        page=citation.page,
    )


def _push(units: list[EvidenceUnit], unit: EvidenceUnit | None) -> None:
    if unit is not None:
        units.append(unit)


def _add_work(resume: CitedResume, units: list[EvidenceUnit], counter: _Counter) -> None:
    for job in resume.work:
        role = job.position.value if job.position else ""
        company = job.company.value
        context = f"{role} at {company}" if role else company

        # The role line itself is evidence of seniority and domain.
        _push(units, _unit(job.company, section="work", counter=counter, context=role))

        for highlight in job.highlights:
            _push(units, _unit(highlight, section="work", counter=counter, context=context))


def _add_projects(resume: CitedResume, units: list[EvidenceUnit], counter: _Counter) -> None:
    for project in resume.projects:
        name = project.name.value
        context = f"Project {name}"

        _push(units, _unit(project.name, section="projects", counter=counter))
        if project.description:
            _push(
                units,
                _unit(project.description, section="projects", counter=counter, context=context),
            )
        for highlight in project.highlights:
            _push(units, _unit(highlight, section="projects", counter=counter, context=context))

        # Technologies are grouped into one unit rather than one per item: a
        # requirement asking for "Go and Kubernetes" should be able to match a
        # single chunk containing both.
        techs = [t for t in project.technologies if t.citations]
        if techs:
            _push(
                units,
                _unit(
                    techs[0],
                    section="projects",
                    counter=counter,
                    text_override=f"{context} uses {', '.join(t.value for t in techs)}",
                ),
            )


def _add_education(resume: CitedResume, units: list[EvidenceUnit], counter: _Counter) -> None:
    for entry in resume.education:
        parts = [
            entry.degree.value if entry.degree else "",
            entry.field_of_study.value if entry.field_of_study else "",
        ]
        descriptor = " ".join(p for p in parts if p).strip()
        _push(
            units,
            _unit(
                entry.institution,
                section="education",
                counter=counter,
                text_override=f"{descriptor} {entry.institution.value}".strip(),
            ),
        )


def _add_skills(resume: CitedResume, units: list[EvidenceUnit], counter: _Counter) -> None:
    """One unit per skill, plus a combined unit for the whole list.

    Both are useful. Per-skill units let "must know Kubernetes" match precisely.
    The combined unit lets "broad backend toolchain" match the breadth, which no
    individual skill demonstrates.
    """
    grounded = [s for s in resume.skills if s.name.citations]
    for skill in grounded:
        _push(
            units,
            _unit(
                skill.name,
                section="skills",
                counter=counter,
                text_override=f"{skill.category} {skill.name.value}".strip(),
            ),
        )

    if len(grounded) > 1:
        _push(
            units,
            _unit(
                grounded[0].name,
                section="skills",
                counter=counter,
                text_override="Skills: " + ", ".join(s.name.value for s in grounded),
            ),
        )


def _add_awards(resume: CitedResume, units: list[EvidenceUnit], counter: _Counter) -> None:
    for award in resume.awards:
        awarder = award.awarder.value if award.awarder else ""
        _push(
            units,
            _unit(
                award.title,
                section="awards",
                counter=counter,
                text_override=f"{award.title.value} {awarder}".strip(),
            ),
        )


def _add_headline(resume: CitedResume, units: list[EvidenceUnit], counter: _Counter) -> None:
    if resume.basics.headline:
        _push(units, _unit(resume.basics.headline, section="basics", counter=counter))


# ---------------------------------------------------------------------------


def merge_overlapping(units: list[EvidenceUnit]) -> list[EvidenceUnit]:
    """Drop units whose span is fully contained in another unit's span.

    Extraction sometimes produces a highlight and a project description that
    resolve to the same line. Keeping both means the same sentence appears twice
    in a retrieval result, which wastes prompt budget and makes an assessment look
    better-supported than it is.
    """
    if not units:
        return []

    ordered = sorted(units, key=lambda u: (u.span.start, -len(u.span)))
    kept: list[EvidenceUnit] = []
    for unit in ordered:
        contained = any(
            other.span.start <= unit.span.start and unit.span.end <= other.span.end
            for other in kept
        )
        if not contained:
            kept.append(unit)
    return kept


def coverage(units: list[EvidenceUnit], document_length: int) -> float:
    """Fraction of the document covered by evidence units.

    A diagnostic rather than a scoring input. Low coverage on a dense resume means
    extraction missed content, which is worth surfacing before anyone trusts the
    score built on top of it.
    """
    if document_length <= 0 or not units:
        return 0.0

    merged: list[Span] = []
    for span in sorted((u.span for u in units), key=lambda s: s.start):
        if merged and span.start <= merged[-1].end:
            merged[-1] = Span(start=merged[-1].start, end=max(merged[-1].end, span.end))
        else:
            merged.append(span)

    return min(sum(len(s) for s in merged) / document_length, 1.0)
