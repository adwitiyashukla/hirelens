"""Local embeddings, with a dependency-free fallback.

Embeddings are the one part of this pipeline that runs on every chunk of every
resume, so they are also the part that would dominate cost if they went through an
API. Running them locally makes the expensive stage free and offline, forever.
``bge-small-en-v1.5`` is 33M parameters, about 130MB on disk, and embeds a resume
in well under a second on a laptop CPU.

The fallback matters as much as the model. ``HashingEmbedder`` implements the same
interface using a deterministic hashed bag of character n-grams, no downloads and
no torch. It exists so that:

* the test suite runs in CI in seconds without pulling 2GB of PyTorch wheels,
* someone can clone the repo and see the pipeline work before deciding to install
  the heavier extras,
* a machine with no disk space or no network still gets a working system, just a
  less semantically capable one.

It is genuinely worse at paraphrase, which is the whole point of dense retrieval,
so it is never the silent default in production use. It is selected explicitly, or
automatically with a loud warning when sentence-transformers is absent.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod
from functools import lru_cache

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")

Vector = list[float]


class Embedder(ABC):
    """Turns text into vectors."""

    name: str
    dimensions: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[Vector]:
        """Embed a batch. Batching is required, not optional: per-item calls into
        a transformer are several times slower than one batched forward pass."""

    def embed_one(self, text: str) -> Vector:
        return self.embed([text])[0]


class SentenceTransformerEmbedder(Embedder):
    """Real dense embeddings via sentence-transformers, on local CPU."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "sentence-transformers is not installed. Install the retrieval "
                'extra with: pip install -e ".[retrieval]"\n'
                "Or use HashingEmbedder for a dependency-free approximation."
            ) from exc

        self.name = model_name
        logger.info("loading embedding model %s (first run downloads ~130MB)", model_name)
        self._model = SentenceTransformer(model_name, device="cpu")
        self.dimensions = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        # normalize_embeddings makes the vectors unit length, so cosine similarity
        # reduces to a dot product and the retriever does not have to normalise.
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32
        )
        return [list(map(float, v)) for v in vectors]


class HashingEmbedder(Embedder):
    """Deterministic hashed n-gram embeddings. No model, no download, no torch.

    Words and character 4-grams are hashed into a fixed-width vector with signed
    buckets, then L2-normalised. Character n-grams give it partial robustness to
    morphology ("deploy" against "deployed", "Kubernetes" against "kubernetes"),
    which a pure bag of words would not have.

    What it cannot do is paraphrase. "Shipped to production" and "operated live
    services" share almost no surface form and will score near zero, where a real
    embedding model scores them close. That is exactly why the hybrid retriever
    keeps BM25 alongside dense retrieval, and why this fallback is still usable:
    lexical matching was always carrying half the load.
    """

    def __init__(self, dimensions: int = 384, *, ngram: int = 4) -> None:
        self.name = f"hashing-{dimensions}d"
        self.dimensions = dimensions
        self.ngram = ngram

    def embed(self, texts: list[str]) -> list[Vector]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> Vector:
        vector = [0.0] * self.dimensions
        for feature, weight in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            # Sign from an independent byte, so collisions cancel on average
            # instead of always reinforcing.
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign * weight

        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0.0:
            return vector
        return [component / norm for component in vector]

    def _features(self, text: str) -> list[tuple[str, float]]:
        tokens = _TOKEN.findall(text.lower())
        features: list[tuple[str, float]] = [(f"w:{token}", 1.0) for token in tokens]
        for token in tokens:
            if len(token) > self.ngram:
                padded = f"^{token}$"
                for index in range(len(padded) - self.ngram + 1):
                    # Sub-token features are down-weighted so a whole-word match
                    # still dominates a partial one.
                    features.append((f"c:{padded[index : index + self.ngram]}", 0.35))
        return features


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity. Handles un-normalised input so it is safe on any embedder."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@lru_cache(maxsize=4)
def get_embedder(
    model_name: str = "BAAI/bge-small-en-v1.5", *, allow_fallback: bool = True
) -> Embedder:
    """Load the configured embedder, falling back loudly if it is unavailable.

    Cached because loading a transformer takes a couple of seconds and a batch run
    would otherwise pay that per resume.
    """
    try:
        return SentenceTransformerEmbedder(model_name)
    except ImportError:
        if not allow_fallback:
            raise
        logger.warning(
            "sentence-transformers is not installed, falling back to HashingEmbedder. "
            "Retrieval will still work but will not match paraphrases. "
            'Install real embeddings with: pip install -e ".[retrieval]"'
        )
        return HashingEmbedder()
