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

_BM25_K1 = 1.5
_BM25_B = 0.75

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
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    unit: EvidenceUnit
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None

    @property
    def found_by(self) -> str:
        parts = []
        if self.bm25_rank is not None:
            parts.append("lexical")
        if self.dense_rank is not None:
            parts.append("semantic")
        return "+".join(parts) or "none"


class BM25:
    def __init__(self, documents: list[str]) -> None:
        self.corpus = [tokenize(document) for document in documents]
        self.size = len(self.corpus)
        self.lengths = [len(doc) for doc in self.corpus]
        self.average_length = (sum(self.lengths) / self.size) if self.size else 0.0
        self.term_frequencies = [Counter(doc) for doc in self.corpus]

        document_frequency: Counter[str] = Counter()
        for doc in self.corpus:
            document_frequency.update(set(doc))

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
        return {key: self.search(query, top_k=top_k) for key, query in queries.items()}


def _ranks(scores: list[float], pool: int) -> dict[int, int]:
    ordered = sorted(
        (index for index, score in enumerate(scores) if score > 0.0),
        key=lambda index: -scores[index],
    )
    return {index: rank for rank, index in enumerate(ordered[:pool], start=1)}
