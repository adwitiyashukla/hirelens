from __future__ import annotations

import re

from hirelens.schemas.assessment import CandidateAssessment, RiskFlag, RiskLevel
from hirelens.schemas.resume import CitedResume

_HAS_NUMBER = re.compile(r"\d")

_VAGUE_PHRASES = (
    "responsible for",
    "worked on",
    "helped with",
    "involved in",
    "participated in",
    "assisted with",
    "familiar with",
    "exposure to",
)

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_YEAR = re.compile(r"(19|20)\d{2}")
_MONTH_WORD = re.compile(r"[a-z]{3,9}", re.IGNORECASE)
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[/-](\d{4})\b")

_GAP_MONTHS = 12

_MIN_GROUNDING_RATE = 0.7


def parse_month_index(value: str) -> int | None:
    text = value.strip().lower()
    if not text:
        return None

    if any(word in text for word in ("present", "current", "now", "ongoing")):
        return _PRESENT

    numeric = _NUMERIC_DATE.search(text)
    if numeric:
        month, year = int(numeric.group(1)), int(numeric.group(2))
        if 1 <= month <= 12:
            return year * 12 + month

    year_match = _YEAR.search(text)
    if not year_match:
        return None
    year = int(year_match.group(0))

    for word in _MONTH_WORD.findall(text):
        month = _MONTHS.get(word[:3])
        if month:
            return year * 12 + month

    return year * 12 + 6


_PRESENT = 99999 * 12


def detect_risks(
    resume: CitedResume, assessment: CandidateAssessment | None = None
) -> list[RiskFlag]:
    flags: list[RiskFlag] = []

    flags.extend(_employment_gaps(resume))
    flags.extend(_unlinked_projects(resume))
    flags.extend(_vague_claims(resume))
    flags.extend(_grounding_risk(resume))
    if assessment is not None:
        flags.extend(_assessment_risks(assessment))

    return flags


def _employment_gaps(resume: CitedResume) -> list[RiskFlag]:
    periods: list[tuple[int, int]] = []
    for job in resume.work:
        start = parse_month_index(job.start_date.value) if job.start_date else None
        if start is None:
            continue
        if job.is_current or job.end_date is None:
            end = _PRESENT
        else:
            end = parse_month_index(job.end_date.value) or start
        periods.append((start, min(end, _PRESENT)))

    if len(periods) < 2:
        return []

    periods.sort()
    flags: list[RiskFlag] = []
    latest_end = periods[0][1]

    for start, end in periods[1:]:
        gap = start - latest_end
        if gap >= _GAP_MONTHS and latest_end != _PRESENT:
            flags.append(
                RiskFlag(
                    code="employment_gap",
                    level=RiskLevel.LOW,
                    message=(
                        f"Roughly {gap // 12} year(s) between listed roles. This is an "
                        f"observation only and does not affect the score. There are many "
                        f"ordinary reasons for a gap; ask if it is relevant."
                    ),
                )
            )
        latest_end = max(latest_end, end)

    return flags


def _unlinked_projects(resume: CitedResume) -> list[RiskFlag]:
    unlinked = [p for p in resume.projects if not p.has_link]
    if len(unlinked) < 2:
        return []

    return [
        RiskFlag(
            code="projects_without_links",
            level=RiskLevel.MEDIUM if len(unlinked) >= 2 else RiskLevel.LOW,
            message=(
                f"{len(unlinked)} project(s) have no repository or demo link: "
                f"{', '.join(p.name.value for p in unlinked[:4])}. "
                f"Their claims cannot be independently checked."
            ),
            citations=tuple(c for p in unlinked[:4] for c in p.name.citations[:1]),
        )
    ]


def _vague_claims(resume: CitedResume) -> list[RiskFlag]:
    highlights = [h for job in resume.work for h in job.highlights]
    highlights += [h for project in resume.projects for h in project.highlights]

    if len(highlights) < 3:
        return []

    vague = [
        h
        for h in highlights
        if not _HAS_NUMBER.search(h.value)
        and any(phrase in h.value.lower() for phrase in _VAGUE_PHRASES)
    ]

    if len(vague) < 2:
        return []

    return [
        RiskFlag(
            code="vague_claims",
            level=RiskLevel.LOW,
            message=(
                f"{len(vague)} of {len(highlights)} achievement bullets describe "
                f"involvement without a measurable outcome. Worth probing for the "
                f"candidate's specific contribution."
            ),
            citations=tuple(c for h in vague[:3] for c in h.citations[:1]),
        )
    ]


def _grounding_risk(resume: CitedResume) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    grounding = resume.grounding

    if grounding.total_fields and grounding.grounding_rate < _MIN_GROUNDING_RATE:
        flags.append(
            RiskFlag(
                code="low_grounding",
                level=RiskLevel.HIGH,
                message=(
                    f"Only {grounding.grounding_rate:.0%} of extracted fields could be "
                    f"traced back to the document. The parse is unreliable for this "
                    f"file, so the score below should not be trusted without reading "
                    f"the original."
                ),
            )
        )

    if resume.failed_sections:
        flags.append(
            RiskFlag(
                code="failed_sections",
                level=RiskLevel.HIGH,
                message=(
                    f"Could not parse: {', '.join(resume.failed_sections)}. "
                    f"Requirements depending on those sections may be scored too low."
                ),
            )
        )

    return flags


def _assessment_risks(assessment: CandidateAssessment) -> list[RiskFlag]:
    flags: list[RiskFlag] = []

    unmet = assessment.unmet_must_haves
    if unmet:
        flags.append(
            RiskFlag(
                code="unmet_must_have",
                level=RiskLevel.HIGH,
                message=(
                    f"{len(unmet)} must-have requirement(s) have no supporting evidence: "
                    + "; ".join(a.requirement_text for a in unmet[:3])
                ),
            )
        )

    ambiguous = assessment.needs_review
    if ambiguous:
        flags.append(
            RiskFlag(
                code="ambiguous_judgement",
                level=RiskLevel.MEDIUM,
                message=(
                    f"{len(ambiguous)} requirement(s) produced inconsistent verdicts across "
                    f"repeated runs: "
                    + "; ".join(a.requirement_text for a in ambiguous[:3])
                    + ". The evidence is genuinely borderline and a human should decide."
                ),
            )
        )

    return flags
