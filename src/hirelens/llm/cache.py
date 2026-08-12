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
    def __init__(self, directory: Path, *, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key_for(request: CompletionRequest, model: str) -> str:
        material = f"{_CACHE_VERSION}|{request.cache_key_material(model)}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _path_for(self, key: str) -> Path:
        return self.directory / key[:2] / key[2:4] / f"{key}.json"

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
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            logger.warning("could not write cache entry: %s", exc)
            tmp.unlink(missing_ok=True)

    def evict(self, request: CompletionRequest, model: str) -> bool:
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

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, float | int]:
        return {"hits": self.hits, "misses": self.misses, "hit_rate": round(self.hit_rate, 3)}

    def clear(self) -> int:
        if not self.directory.exists():
            return 0
        removed = 0
        for entry in self.directory.rglob("*.json"):
            entry.unlink(missing_ok=True)
            removed += 1
        self.hits = self.misses = 0
        return removed
