"""On-disk cache for LLM responses.

This is not a performance optimisation, it is a correctness and cost tool, and it
earns its place three times over:

1. **Free iteration.** Tweaking a downstream prompt should not re-pay for the
   twenty upstream calls that did not change. On a free tier with a daily request
   quota, that difference decides whether you can work on the project all evening
   or not.
2. **Reproducible evaluation.** ``make eval`` must produce the same numbers twice
   in a row or the metrics mean nothing. Caching keyed on the exact request gives
   us that for free.
3. **Offline development.** Once a resume has been processed, the whole pipeline
   replays with no network at all, which makes tests fast and CI cheap.

Sharded two levels deep by key prefix because a golden set of sixty resumes at
k=5 sampling produces a few thousand files, and flat directories of that size are
miserable on Windows.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

from hirelens.llm.base import CompletionRequest, CompletionResponse, Usage

logger = logging.getLogger(__name__)

_CACHE_VERSION = "v1"


class ResponseCache:
    """Content-addressed store mapping a request to the response it produced."""

    def __init__(self, directory: Path, *, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    # -- keys ----------------------------------------------------------------

    @staticmethod
    def key_for(request: CompletionRequest, model: str) -> str:
        material = f"{_CACHE_VERSION}|{request.cache_key_material(model)}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _path_for(self, key: str) -> Path:
        return self.directory / key[:2] / key[2:4] / f"{key}.json"

    # -- io ------------------------------------------------------------------

    def get(self, request: CompletionRequest, model: str) -> CompletionResponse | None:
        if not self.enabled:
            return None

        path = self._path_for(self.key_for(request, model))
        if not path.exists():
            self.misses += 1
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A truncated cache entry is not worth crashing over. Drop it and
            # treat the lookup as a miss.
            logger.debug("discarding unreadable cache entry %s", path)
            path.unlink(missing_ok=True)
            self.misses += 1
            return None

        self.hits += 1
        usage = payload.get("usage", {})
        return CompletionResponse(
            content=payload["content"],
            model=payload["model"],
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            ),
            cached=True,
            latency_s=0.0,
            finish_reason=payload.get("finish_reason"),
        )

    def put(
        self,
        request: CompletionRequest,
        model: str,
        response: CompletionResponse,
    ) -> None:
        if not self.enabled:
            return

        path = self._path_for(self.key_for(request, model))
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "content": response.content,
            "model": response.model,
            "usage": asdict(response.usage),
            "finish_reason": response.finish_reason,
        }
        # Write to a temp file and move it into place, so a process killed
        # mid-write cannot leave a half-JSON file behind.
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            logger.warning("could not write cache entry: %s", exc)
            tmp.unlink(missing_ok=True)

    def evict(self, request: CompletionRequest, model: str) -> bool:
        """Forget one entry. Returns whether anything was removed.

        Exists because caching a response that later fails schema validation
        makes the failure permanent. The prompt is deterministic, so the next
        run hits the same entry, gets the same malformed answer, and fails
        identically. The repair loop cannot help: it is being handed the same
        bad output every time.

        This was not theoretical. A rate-limited run cached some malformed
        extraction responses, and from then on that resume produced almost no
        evidence on every subsequent run, including runs where the provider was
        healthy. The only symptom visible to a user was a strong candidate being
        reported as a weak match, with no error anywhere.
        """
        if not self.enabled:
            return False

        path = self._path_for(self.key_for(request, model))
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        except OSError as exc:
            logger.warning("could not evict cache entry: %s", exc)
            return False
        return True

    # -- reporting -----------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, float | int]:
        return {"hits": self.hits, "misses": self.misses, "hit_rate": round(self.hit_rate, 3)}

    def clear(self) -> int:
        """Delete every entry. Returns how many were removed."""
        if not self.directory.exists():
            return 0
        removed = 0
        for entry in self.directory.rglob("*.json"):
            entry.unlink(missing_ok=True)
            removed += 1
        self.hits = self.misses = 0
        return removed
