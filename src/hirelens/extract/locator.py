from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from hirelens.schemas.evidence import Span

logger = logging.getLogger(__name__)

_FUZZY_THRESHOLD = 0.6

_MIN_QUOTE_LENGTH = 4

_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class LocatedSpan:
    span: Span
    score: float
    strategy: str


class SpanLocator:
    def __init__(self, text: str) -> None:
        self.text = text
        self._normalised, self._index_map = _build_normalised_index(text)
        self._line_spans = _line_spans(text)

    def locate(self, quote: str, *, within: Span | None = None) -> LocatedSpan | None:
        quote = quote.strip()
        if not quote:
            return None

        if len(quote) < _MIN_QUOTE_LENGTH:
            return self._locate_unique_short(quote, within)

        for strategy in (self._locate_exact, self._locate_normalised, self._locate_fuzzy):
            found = strategy(quote, within)
            if found is not None:
                return found
        return None

    def _locate_unique_short(self, quote: str, within: Span | None) -> LocatedSpan | None:
        lo, hi = self._bounds(within)
        pattern = re.compile(rf"(?<![\w+#]){re.escape(quote)}(?![\w+#])", re.IGNORECASE)
        matches = list(pattern.finditer(self.text, lo, hi))
        if len(matches) != 1:
            return None
        match = matches[0]
        return LocatedSpan(
            span=Span(start=match.start(), end=match.end()),
            score=0.9,
            strategy="unique-short",
        )

    def locate_all(
        self, quotes: list[str], *, within: Span | None = None
    ) -> dict[str, LocatedSpan]:
        results: dict[str, LocatedSpan] = {}
        for quote in quotes:
            found = self.locate(quote, within=within)
            if found is not None:
                results[quote] = found
        return results

    def _locate_exact(self, quote: str, within: Span | None) -> LocatedSpan | None:
        lo, hi = self._bounds(within)
        position = self.text.find(quote, lo, hi)
        if position == -1:
            return None
        return LocatedSpan(
            span=Span(start=position, end=position + len(quote)), score=1.0, strategy="exact"
        )

    def _locate_normalised(self, quote: str, within: Span | None) -> LocatedSpan | None:
        needle = _normalise(quote)
        if len(needle) < _MIN_QUOTE_LENGTH:
            return None

        lo, hi = self._bounds(within)
        norm_lo = self._to_normalised_index(lo, default=0)
        norm_hi = self._to_normalised_index(hi, default=len(self._normalised))

        position = self._normalised.find(needle, norm_lo, norm_hi)
        if position == -1:
            return None

        start = self._index_map[position]
        end = self._index_map[position + len(needle) - 1] + 1
        return LocatedSpan(span=Span(start=start, end=end), score=0.95, strategy="normalised")

    def _locate_fuzzy(self, quote: str, within: Span | None) -> LocatedSpan | None:
        needle_tokens = _tokens(quote)
        if not needle_tokens:
            return None

        lo, hi = self._bounds(within)
        candidates = [s for s in self._line_spans if s.start >= lo and s.end <= hi]
        if not candidates:
            return None

        best: LocatedSpan | None = None

        for index, span in enumerate(candidates):
            windows = [span]
            if index + 1 < len(candidates):
                windows.append(Span(start=span.start, end=candidates[index + 1].end))

            for window in windows:
                score = _jaccard(needle_tokens, _tokens(self.text[window.start : window.end]))
                if score >= _FUZZY_THRESHOLD and (best is None or score > best.score):
                    best = LocatedSpan(span=window, score=score, strategy="fuzzy")

        if best is not None:
            logger.debug("fuzzy located %r at %s (score %.2f)", quote[:40], best.span, best.score)
        return best

    def _bounds(self, within: Span | None) -> tuple[int, int]:
        if within is None:
            return 0, len(self.text)
        return max(0, within.start), min(len(self.text), within.end)

    def _to_normalised_index(self, original_index: int, *, default: int) -> int:
        if not self._index_map:
            return default
        lo, hi = 0, len(self._index_map)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._index_map[mid] < original_index:
                lo = mid + 1
            else:
                hi = mid
        return lo


def _build_normalised_index(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    index_map: list[int] = []
    previous_was_space = True

    for position, char in enumerate(text):
        if char.isspace():
            if previous_was_space:
                continue
            chars.append(" ")
            index_map.append(position)
            previous_was_space = True
        else:
            chars.append(char.lower())
            index_map.append(position)
            previous_was_space = False

    return "".join(chars), index_map


def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip().lower()


def _line_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    offset = 0
    for line in text.split("\n"):
        if line.strip():
            spans.append(Span(start=offset, end=offset + len(line)))
        offset += len(line) + 1
    return spans


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
