from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from hirelens._compat import StrEnum


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(Role.SYSTEM, content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(Role.USER, content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls(Role.ASSISTANT, content)


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    messages: tuple[Message, ...]
    temperature: float = 0.0
    max_tokens: int | None = None
    json_schema: dict[str, Any] | None = None
    """When set, the provider is asked for JSON conforming to this schema.

    Support varies: Gemini and Ollama accept a real schema, Groq only supports a
    generic "must be valid JSON" mode. Each provider degrades on its own terms and
    the prompt always restates the schema in words, so correctness never depends
    on the provider honouring it.
    """
    seed: int | None = None

    def cache_key_material(self, model: str) -> str:
        return json.dumps(
            {
                "model": model,
                "messages": [[m.role.value, m.content] for m in self.messages],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "json_schema": self.json_schema,
                "seed": self.seed,
            },
            sort_keys=True,
            ensure_ascii=False,
        )


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class CompletionResponse:
    content: str
    model: str
    usage: Usage = field(default_factory=Usage)
    cached: bool = False
    latency_s: float = 0.0
    finish_reason: str | None = None

    def json(self) -> Any:
        return json.loads(extract_json(self.content))


class LLMError(RuntimeError):
    pass


class RateLimitError(LLMError):
    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class TransientProviderError(LLMError):
    pass


class InvalidResponseError(LLMError):
    pass


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        pass

    @abstractmethod
    async def aclose(self) -> None:
        pass

    async def __aenter__(self) -> LLMProvider:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> str:
    cleaned = _THINK_BLOCK.sub("", text).strip()

    fenced = _FENCE.search(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()

    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                json.loads(candidate)
            except json.JSONDecodeError:
                continue
            return candidate

    return cleaned
