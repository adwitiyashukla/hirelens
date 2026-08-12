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

_INSTITUTION = re.compile(
    r"\b(?:"
    r"University\s+of\s+[A-Z][\w.-]*(?:\s+[A-Z][\w.-]*)?"
    r"|[A-Z][\w.-]*(?:\s+[A-Z][\w.-]*)?\s+(?:University|College|Institute|Polytechnic|Academy)"
    r"|(?:IIT|NIT|IIIT|BITS|MIT|ETH|EPFL|KTH)\s+[A-Z][\w.-]*"
    r"|\b(?:IIT|NIT|IIIT|BITS)\b"
    r")",
)

_GENDER_TERMS = re.compile(
    r"\b(?:he|him|his|she|her|hers|mr|mrs|ms|miss|sir|madam|male|female|"
    r"fraternity|sorority|husband|wife|maternity|paternity)\b",
    re.IGNORECASE,
)

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
    model_config = ConfigDict(frozen=True)

    span: Span
    category: PIICategory
    text: str

    def __len__(self) -> int:
        return len(self.span)


class RedactionReport(BaseModel):
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


def detect_pii(text: str) -> list[PIISpan]:
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
    return bool(
        re.fullmatch("\\s*(19|20)\\d{2}\\s*[-\u2013\u2014]\\s*(19|20)\\d{2}\\s*", candidate)
    )


def _detect_name(text: str) -> PIISpan | None:
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
    ordered = sorted(spans, key=lambda s: (-len(s.span), s.span.start))
    kept: list[PIISpan] = []
    for candidate in ordered:
        if not any(candidate.span.overlaps(existing.span) for existing in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda s: s.span.start)


def redact(
    text: str,
    *,
    categories: Iterable[PIICategory] | None = None,
    spans: list[PIISpan] | None = None,
) -> RedactionReport:
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
    label = f"[{str(category).upper()}]"
    if len(label) <= width:
        return label + _MASK_CHAR * (width - len(label))
    return _MASK_CHAR * width
