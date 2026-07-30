"""Detect and mask the parts of a resume that should not influence a score.

This is a fairness control, not a privacy nicety. The categories detected here are
exactly the ones the counterfactual audit in Phase 6 perturbs: name, gender-coded
terms, institution, and location. Removing them before the model ever sees the
text is the cheapest possible bias mitigation, and leaving them in would make the
audit measure a problem we had chosen not to fix.

**The masking is length-preserving**, and that is the whole trick. Replacing
"Priya Narayanan" with "[NAME]" would shift every subsequent character by nine
positions and silently invalidate every span in the document. Instead the mask
occupies exactly the same number of characters, so the redacted view and the
original share one coordinate system and a citation resolved against one is valid
against the other.

Detection is rules-based rather than a NER model on purpose. A resume header is
one of the most structurally predictable documents there is, spaCy or Presidio
would add hundreds of megabytes of dependency for a marginal recall gain, and
rules are auditable, which for a fairness control is worth more than the last few
percent. The limitation is recorded honestly in the README rather than hidden.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from hirelens._compat import StrEnum
from hirelens.schemas.evidence import Span


class PIICategory(StrEnum):
    NAME = "name"
    EMAIL = "email"
    PHONE = "phone"
    LOCATION = "location"
    INSTITUTION = "institution"
    URL = "url"
    GENDER_TERM = "gender_term"


#: Categories masked when blind mode is on. URLs are deliberately excluded: a
#: GitHub link is real professional signal and the enrichment stage needs it. The
#: username it exposes is a residual identity leak, and that trade-off is stated
#: in the README rather than quietly resolved.
DEFAULT_BLIND_CATEGORIES: frozenset[PIICategory] = frozenset(
    {
        PIICategory.NAME,
        PIICategory.EMAIL,
        PIICategory.PHONE,
        PIICategory.LOCATION,
        PIICategory.INSTITUTION,
        PIICategory.GENDER_TERM,
    }
)

_MASK_CHAR = "#"

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(
    r"(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3,5}[\s.-]?\d{3,4}(?:[\s.-]?\d{2,4})?"
)
_URL = re.compile(r"\b(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|org|net|io|dev|me|ai)/\S*")

# "University of X", "X University", "IIT Bombay", "NIT Trichy", and friends.
_INSTITUTION = re.compile(
    r"\b(?:"
    r"University\s+of\s+[A-Z][\w.-]*(?:\s+[A-Z][\w.-]*)?"
    r"|[A-Z][\w.-]*(?:\s+[A-Z][\w.-]*)?\s+(?:University|College|Institute|Polytechnic|Academy)"
    r"|(?:IIT|NIT|IIIT|BITS|MIT|ETH|EPFL|KTH)\s+[A-Z][\w.-]*"
    r"|\b(?:IIT|NIT|IIIT|BITS)\b"
    r")",
)

# Gendered terms that carry no job-relevant information.
_GENDER_TERMS = re.compile(
    r"\b(?:he|him|his|she|her|hers|mr|mrs|ms|miss|sir|madam|male|female|"
    r"fraternity|sorority|husband|wife|maternity|paternity)\b",
    re.IGNORECASE,
)

# A conservative city list. Deliberately small: a broad gazetteer would start
# masking ordinary words and damage the text the model has to reason over.
_CITY_TOKENS = frozenset(
    {
        "bengaluru",
        "bangalore",
        "mumbai",
        "delhi",
        "hyderabad",
        "chennai",
        "pune",
        "kolkata",
        "ahmedabad",
        "jaipur",
        "lucknow",
        "noida",
        "gurgaon",
        "gurugram",
        "london",
        "manchester",
        "berlin",
        "munich",
        "paris",
        "amsterdam",
        "dublin",
        "toronto",
        "vancouver",
        "sydney",
        "melbourne",
        "singapore",
        "dubai",
        "york",
        "francisco",
        "seattle",
        "austin",
        "boston",
        "chicago",
        "denver",
    }
)
_CITY = re.compile(r"\b(?:" + "|".join(sorted(_CITY_TOKENS)) + r")\b", re.IGNORECASE)

# How many characters from the top of the document count as the header block,
# where the candidate's name lives.
_HEADER_WINDOW = 400

_NAME_STOPWORDS = frozenset(
    {
        "curriculum",
        "vitae",
        "resume",
        "cv",
        "profile",
        "summary",
        "objective",
        "experience",
        "education",
        "skills",
        "projects",
        "contact",
        "about",
    }
)


class PIISpan(BaseModel):
    """One detected piece of identifying information."""

    model_config = ConfigDict(frozen=True)

    span: Span
    category: PIICategory
    text: str

    def __len__(self) -> int:
        return len(self.span)


class RedactionReport(BaseModel):
    """What was found, and what the masked view looks like."""

    spans: list[PIISpan] = Field(default_factory=list)
    redacted_text: str = ""

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for item in self.spans:
            out[str(item.category)] = out.get(str(item.category), 0) + 1
        return out

    def spans_for(self, category: PIICategory) -> list[PIISpan]:
        return [s for s in self.spans if s.category is category]

    def summary(self) -> str:
        if not self.spans:
            return "no PII detected"
        return ", ".join(f"{count} {name}" for name, count in sorted(self.counts.items()))


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_pii(text: str) -> list[PIISpan]:
    """Find every identifying span in the document, de-overlapped.

    Overlaps are real: an institution line often contains a city, and a contact
    line contains both an email and a phone-shaped fragment. We keep the longest
    match at each position so masking stays clean and the report does not
    double-count.
    """
    found: list[PIISpan] = []

    def add(match: re.Match[str], category: PIICategory) -> None:
        value = match.group(0).strip()
        if not value:
            return
        start = match.start() + (len(match.group(0)) - len(match.group(0).lstrip()))
        found.append(
            PIISpan(span=Span(start=start, end=start + len(value)), category=category, text=value)
        )

    for match in _EMAIL.finditer(text):
        add(match, PIICategory.EMAIL)
    for match in _URL.finditer(text):
        add(match, PIICategory.URL)
    for match in _INSTITUTION.finditer(text):
        add(match, PIICategory.INSTITUTION)
    for match in _CITY.finditer(text):
        add(match, PIICategory.LOCATION)
    for match in _GENDER_TERMS.finditer(text):
        add(match, PIICategory.GENDER_TERM)

    # Phone numbers are matched last and filtered, because the pattern is loose
    # enough to swallow dates ("2019 - 2023") and metrics ("40k transactions").
    for match in _PHONE.finditer(text):
        candidate = match.group(0)
        digits = sum(character.isdigit() for character in candidate)
        if 7 <= digits <= 15 and not _looks_like_a_year_range(candidate):
            add(match, PIICategory.PHONE)

    name = _detect_name(text)
    if name is not None:
        found.append(name)

    return _drop_overlaps(found)


def _looks_like_a_year_range(candidate: str) -> bool:
    """Reject '2019 - 2023' and similar, which the phone pattern happily matches."""
    return bool(
        re.fullmatch("\\s*(19|20)\\d{2}\\s*[-\u2013\u2014]\\s*(19|20)\\d{2}\\s*", candidate)
    )


def _detect_name(text: str) -> PIISpan | None:
    """Find the candidate's name in the header block.

    Resumes put the name first, on its own line, in title case or all caps, with
    no digits and no punctuation beyond a hyphen or apostrophe. That is a narrow
    enough shape to match on directly.
    """
    header = text[:_HEADER_WINDOW]
    offset = 0

    for line in header.split("\n"):
        stripped = line.strip()
        if not stripped:
            offset += len(line) + 1
            continue

        words = stripped.split()
        looks_like_a_name = (
            2 <= len(words) <= 4
            and len(stripped) <= 45
            and not any(character.isdigit() for character in stripped)
            and "@" not in stripped
            and all(re.fullmatch(r"[A-Za-z][A-Za-z'.\-]*", word) for word in words)
            and all(word.lower() not in _NAME_STOPWORDS for word in words)
            and (stripped.isupper() or all(word[0].isupper() for word in words))
        )
        if looks_like_a_name:
            start = offset + line.index(stripped)
            return PIISpan(
                span=Span(start=start, end=start + len(stripped)),
                category=PIICategory.NAME,
                text=stripped,
            )
        offset += len(line) + 1

    return None


def _drop_overlaps(spans: list[PIISpan]) -> list[PIISpan]:
    """Keep the longest span wherever several overlap, then sort by position."""
    ordered = sorted(spans, key=lambda s: (-len(s.span), s.span.start))
    kept: list[PIISpan] = []
    for candidate in ordered:
        if not any(candidate.span.overlaps(existing.span) for existing in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda s: s.span.start)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def redact(
    text: str,
    *,
    categories: Iterable[PIICategory] | None = None,
    spans: list[PIISpan] | None = None,
) -> RedactionReport:
    """Produce a masked view of ``text`` with identical character offsets.

    Every span of a masked category is replaced by a same-length placeholder, so
    ``len(redacted_text) == len(text)`` and any span valid in one is valid in the
    other. That invariant is asserted in the tests because everything downstream
    depends on it.
    """
    selected = frozenset(categories) if categories is not None else DEFAULT_BLIND_CATEGORIES
    detected = spans if spans is not None else detect_pii(text)

    characters = list(text)
    for item in detected:
        if item.category not in selected:
            continue
        placeholder = _placeholder_for(item.category, len(item.span))
        for index, character in enumerate(placeholder):
            characters[item.span.start + index] = character

    return RedactionReport(spans=detected, redacted_text="".join(characters))


def _placeholder_for(category: PIICategory, width: int) -> str:
    """A same-width marker: readable if it fits, plain mask characters if not."""
    label = f"[{str(category).upper()}]"
    if len(label) <= width:
        return label + _MASK_CHAR * (width - len(label))
    return _MASK_CHAR * width
