"""Hybrid retrieval: BM25 and dense vectors, fused by reciprocal rank.

Neither retrieval method is sufficient alone, and the failure modes are almost
perfectly complementary.

**BM25 misses paraphrase.** A requirement saying "experience owning services in
production" shares no terms with a bullet saying "on-call for the payments
service, deployed twice weekly". Lexically these are strangers.

**Dense retrieval misses exact terms.** Embedding models put "Kubernetes",
"Docker" and "containerisation" close together, which is usually what you want and
occasionally catastrophic: a hard requirement for Kubernetes should not be
satisfied by a resume that only says Docker. Rare tokens, version numbers and
library names are exactly where dense retrieval is weakest, and exactly where
hiring requirements are most specific.

Running both and fusing the rankings beats either one. We fuse with **Reciprocal
Rank Fusion**, which scores a document by ``sum(1 / (k + rank))`` across rankers.
RRF uses only ranks, never raw scores, which matters here because BM25 scores are
unbounded corpus statistics and cosine similarities are bounded to [-1, 1]. Any
attempt to combine them by weighted sum requires normalisation constants that have
to be retuned whenever the corpus or the embedding model changes. RRF needs no
tuning and is well established as a strong default.

BM25 is implemented here rather than imported. It is about forty lines, it removes
a dependency, and the corpus is one resume, so the industrial-strength
implementations buy nothing.

**Retrieval favours recall; judging decides.** A requirement the candidate simply
does not meet ("front-end design experience" against a backend resume) will still
return the least-irrelevant units available, because RRF ranks whatever the
rankers found and a corpus of sixteen chunks always has a least-bad answer. That
is deliberate. Adding a score threshold here would be guessing at a cutoff on an
uncalibrated scale, and a threshold set slightly too high silently hides the one
piece of evidence that mattered. Instead the judge sees the retrieved evidence and
is explicitly permitted to answer "none of this supports the requirement", which
is a decision made with the requirement text in hand rather than by a number.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from hirelens.retrieve.embeddings import Embedder, Vector, cosine
from hirelens.schemas.evidence import EvidenceUnit

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9+#.]+")

# Standard BM25 parameters. k1 controls term-frequency saturation, b controls
# length normalisation. These are the usual defaults and there is no reason to
# tune them on a corpus of one resume.
_BM25_K1 = 1.5
_BM25_B = 0.75

# RRF smoothing constant. 60 is the value from the original paper and is what
# almost every implementation uses. Larger k flattens the contribution of top
# ranks; smaller k makes the fusion behave more like a strict rank-1 vote.
_RRF_K = 60

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "using",
        "used",
        "able",
        "experience",
        "experienced",
        "strong",
        "good",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, stopwords removed.

    The token pattern keeps ``+``, ``#`` and ``.`` so that C++, C#, .NET and
    Node.js survive as single tokens. Dropping them would make several very
    common hiring requirements unsearchable.
    """
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """One evidence unit retrieved for a query, with provenance."""

    unit: EvidenceUnit
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None

    @property
    def found_by(self) -> str:
        """Which retriever(s) surfaced this. Useful for debugging a bad match."""
        parts = []
        if self.bm25_rank is not None:
            parts.append("lexical")
        if self.dense_rank is not None:
            parts.append("semantic")
        return "+".join(parts) or "none"


class BM25:
    """Okapi BM25 over a small in-memory corpus."""

    def __init__(self, documents: list[str]) -> None:
        self.corpus = [tokenize(document) for document in documents]
        self.size = len(self.corpus)
        self.lengths = [len(doc) for doc in self.corpus]
        self.average_length = (sum(self.lengths) / self.size) if self.size else 0.0
        self.term_frequencies = [Counter(doc) for doc in self.corpus]

        document_frequency: Counter[str] = Counter()
        for doc in self.corpus:
            document_frequency.update(set(doc))

        # Standard BM25 IDF with the +1 inside the log, which keeps the value
        # positive for terms appearing in most documents. Without it, a term in
        # more than half the corpus gets a negative weight, and on a corpus this
        # small that happens constantly.
        self.idf = {
            term: math.log(1 + (self.size - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        results = [0.0] * self.size
        if not terms or not self.size:
            return results

        for index, frequencies in enumerate(self.term_frequencies):
            length = self.lengths[index]
            total = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                numerator = frequency * (_BM25_K1 + 1)
                denominator = frequency + _BM25_K1 * (
                    1 - _BM25_B + _BM25_B * length / (self.average_length or 1.0)
                )
                total += self.idf.get(term, 0.0) * numerator / denominator
            results[index] = total
        return results


@dataclass
class HybridRetriever:
    """Indexes one candidate's evidence and answers requirement queries."""

    units: list[EvidenceUnit]
    embedder: Embedder
    _bm25: BM25 = field(init=False)
    _vectors: list[Vector] = field(init=False)

    def __post_init__(self) -> None:
        texts = [unit.text for unit in self.units]
        self._bm25 = BM25(texts)
        self._vectors = self.embedder.embed(texts) if texts else []
        logger.debug("indexed %d evidence units with %s", len(self.units), self.embedder.name)

    def search(self, query: str, *, top_k: int = 5, pool: int = 20) -> list[RetrievalHit]:
        """Retrieve the ``top_k`` most relevant evidence units for ``query``.

        ``pool`` is how deep each individual ranker is considered before fusion.
        Fusing only the top few from each would throw away the case RRF exists to
        handle: a unit ranked 8th by both rankers is often a better answer than
        one ranked 1st by a single ranker and nowhere by the other.
        """
        if not self.units:
            return []

        bm25_ranks = _ranks(self._bm25.scores(query), pool)

        dense_ranks: dict[int, int] = {}
        if self._vectors:
            query_vector = self.embedder.embed_one(query)
            similarities = [cosine(query_vector, vector) for vector in self._vectors]
            dense_ranks = _ranks(similarities, pool)

        fused: dict[int, float] = {}
        for ranks in (bm25_ranks, dense_ranks):
            for index, rank in ranks.items():
                fused[index] = fused.get(index, 0.0) + 1.0 / (_RRF_K + rank)

        ordered = sorted(fused.items(), key=lambda pair: -pair[1])[:top_k]
        return [
            RetrievalHit(
                unit=self.units[index],
                score=round(score, 6),
                bm25_rank=bm25_ranks.get(index),
                dense_rank=dense_ranks.get(index),
            )
            for index, score in ordered
        ]

    def search_many(
        self, queries: dict[str, str], *, top_k: int = 5
    ) -> dict[str, list[RetrievalHit]]:
        """Run one search per key. Keys are usually requirement ids."""
        return {key: self.search(query, top_k=top_k) for key, query in queries.items()}


def _ranks(scores: list[float], pool: int) -> dict[int, int]:
    """Map document index to 1-based rank, keeping the top ``pool`` positives.

    Zero-scoring documents are excluded rather than ranked. A BM25 score of zero
    means no query term appeared at all, and giving that a rank would let an
    irrelevant unit accumulate fusion credit purely for existing.
    """
    ordered = sorted(
        (index for index, score in enumerate(scores) if score > 0.0),
        key=lambda index: -scores[index],
    )
    return {index: rank for rank, index in enumerate(ordered[:pool], start=1)}
