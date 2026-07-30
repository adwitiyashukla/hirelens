"""Find where a quoted string lives in the source document.

This module exists because of one empirical fact: **language models cannot count
characters.** Ask a model for the start and end offsets of a phrase and it will
return confident, plausible, wrong integers. Every project that asks an LLM for
character positions and trusts them is producing citations that point at the wrong
text, and because nobody checks, nobody notices.

So we split the responsibility. The model reports *what* it read, verbatim. This
module works out *where* that text is, using an actual string search. Models are
good at the first job and hopeless at the second; computers are the reverse.

Three strategies, tried in order:

1. **Exact** substring search. Handles the majority of cases.
2. **Normalised** search, collapsing whitespace and case. Handles the very common
   case where a model tidies up a quote it copied across a line break. An index
   map carries positions back to the original text.
3. **Fuzzy** sliding window over candidate lines, scored by token overlap. Handles
   a model that paraphrased slightly or dropped a trailing clause.

If all three fail we return nothing and record the quote as unlocatable, which
shows up in the grounding statistics. A quote we cannot find is exactly the signal
we want: either the model invented it, or our extraction is drifting.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from hirelens.schemas.evidence import Span

logger = logging.getLogger(__name__)

# Below this token-overlap score we would rather have no citation than a wrong
# one. An ungrounded field is honest; a misattributed field is a lie with a
# footnote.
_FUZZY_THRESHOLD = 0.6

# Quotes shorter than this are too generic to locate safely. A model quoting "Go"
# or "2023" could match dozens of positions, and picking one at random would
# produce a citation that highlights the wrong line in the UI.
_MIN_QUOTE_LENGTH = 4

_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class LocatedSpan:
    """A resolved span plus how confident we are in it."""

    span: Span
    score: float
    strategy: str


class SpanLocator:
    """Locates quotes within one document.

    Built once per document because the normalised index is the expensive part and
    is reused across every quote in every section.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self._normalised, self._index_map = _build_normalised_index(text)
        # Line spans are the candidate windows for fuzzy matching.
        self._line_spans = _line_spans(text)

    # -- public API ----------------------------------------------------------

    def locate(self, quote: str, *, within: Span | None = None) -> LocatedSpan | None:
        """Find ``quote`` in the document, optionally restricted to ``within``.

        The ``within`` hint matters more than it looks. When extracting the work
        section we already know which slice of the document it came from, and
        restricting the search stops a common word like "Python" being cited to
        the skills list when it appeared in a job bullet.
        """
        quote = quote.strip()
        if not quote:
            return None

        if len(quote) < _MIN_QUOTE_LENGTH:
            # Short quotes are ambiguous by default, but plenty of real skills are
            # short: Go, C, R, C++, SQL. Rather than refusing all of them we accept
            # a short quote when it occurs exactly once in scope, because a unique
            # match cannot be misattributed.
            return self._locate_unique_short(quote, within)

        for strategy in (self._locate_exact, self._locate_normalised, self._locate_fuzzy):
            found = strategy(quote, within)
            if found is not None:
                return found
        return None

    def _locate_unique_short(self, quote: str, within: Span | None) -> LocatedSpan | None:
        """Accept a short quote only when it appears exactly once in scope."""
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
        """Locate many quotes, skipping the ones that cannot be found."""
        results: dict[str, LocatedSpan] = {}
        for quote in quotes:
            found = self.locate(quote, within=within)
            if found is not None:
                results[quote] = found
        return results

    # -- strategies ----------------------------------------------------------

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

        # Translate the restriction window into normalised coordinates.
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
        """Best-scoring line, or run of lines, that resembles the quote.

        A quote can span several lines when a bullet wraps, so we also try merging
        each line with the one after it before giving up.
        """
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
            # Merge with the following line to catch wrapped bullets.
            if index + 1 < len(candidates):
                windows.append(Span(start=span.start, end=candidates[index + 1].end))

            for window in windows:
                score = _jaccard(needle_tokens, _tokens(self.text[window.start : window.end]))
                if score >= _FUZZY_THRESHOLD and (best is None or score > best.score):
                    best = LocatedSpan(span=window, score=score, strategy="fuzzy")

        if best is not None:
            logger.debug("fuzzy located %r at %s (score %.2f)", quote[:40], best.span, best.score)
        return best

    # -- helpers -------------------------------------------------------------

    def _bounds(self, within: Span | None) -> tuple[int, int]:
        if within is None:
            return 0, len(self.text)
        return max(0, within.start), min(len(self.text), within.end)

    def _to_normalised_index(self, original_index: int, *, default: int) -> int:
        """Map an original offset into normalised coordinates.

        The map runs the other way, so we binary-search it. Exact precision is not
        required here: this only bounds a search window.
        """
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


# ---------------------------------------------------------------------------
# Normalisation with an index map back to the original text
# ---------------------------------------------------------------------------


def _build_normalised_index(text: str) -> tuple[str, list[int]]:
    """Return lowercase whitespace-collapsed text plus a per-character index map.

    ``index_map[i]`` is the offset in ``text`` that normalised character ``i`` came
    from. Without this we could find a match in the normalised string and have no
    way to express it as a span in the real document.
    """
    chars: list[str] = []
    index_map: list[int] = []
    previous_was_space = True  # suppresses leading whitespace

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
    """One span per non-blank line."""
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
