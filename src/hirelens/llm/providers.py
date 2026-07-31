"""Concrete backends: Gemini, Groq, and Ollama.

All three are free to use. They differ in how they express three things: auth,
system prompts, and structured output. Those differences are the entire content of
this module; everything above it in the stack sees one interface.

Structured-output support, honestly stated:

* **Gemini** takes a real schema, but a restricted OpenAPI dialect rather than
  full JSON Schema. Pydantic emits ``$defs`` and ``$ref`` for nested models, which
  Gemini rejects, so we inline and strip before sending.
* **Groq** only has a generic "reply with valid JSON" mode, no schema, so the
  schema is written into the prompt instead. That is not a nicety. Without it the
  model invents field names, the repair loop spends its attempts, and it settles
  on a sparse object that validates because the missing fields have defaults. The
  result is an empty answer wearing a confident face, which is the worst of the
  three possible outcomes.
* **Ollama** accepts a full JSON Schema and constrains decoding to it, which is
  the strongest guarantee of the three.

Because the weakest of these is "valid JSON, shape unspecified", the caller always
validates with Pydantic, and the stages downstream check that what came back is
substantive rather than merely well-formed. The provider hint is an optimisation,
never the correctness mechanism.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from hirelens.llm.base import (
    CompletionRequest,
    CompletionResponse,
    InvalidResponseError,
    LLMProvider,
    RateLimitError,
    Role,
    TransientProviderError,
    Usage,
)

logger = logging.getLogger(__name__)


def _raise_for_status(response: httpx.Response, provider: str) -> None:
    """Translate an HTTP status into the right exception type.

    The distinction matters: the retry policy backs off on rate limits and
    transient errors, and gives up immediately on a bad API key, because retrying
    a 401 four times just wastes ten seconds before showing the same message.
    """
    status = response.status_code
    if status < 400:
        return

    body = response.text[:400]

    if status == 429:
        retry_after = response.headers.get("retry-after")
        raise RateLimitError(
            f"{provider} rate limit reached. Free tiers have per-minute and "
            f"per-day quotas; the client will back off and retry. ({body})",
            retry_after_s=float(retry_after) if retry_after else None,
        )
    if status in {401, 403}:
        raise InvalidResponseError(
            f"{provider} rejected the credentials ({status}). Check the API key in "
            f"your .env file. ({body})"
        )
    if status >= 500:
        raise TransientProviderError(f"{provider} server error {status}: {body}")
    raise InvalidResponseError(f"{provider} returned {status}: {body}")


class _HTTPProvider(LLMProvider):
    """Shared httpx client lifecycle."""

    def __init__(self, *, base_url: str, timeout_s: float, headers: dict[str, str]) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            headers=headers,
            # Resume batches issue many small requests; a healthy pool avoids
            # paying TLS setup on every one.
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )

    async def _post(self, url: str, payload: dict[str, Any]) -> httpx.Response:
        try:
            response = await self._client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise TransientProviderError(f"{self.name} timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise TransientProviderError(f"{self.name} connection failed: {exc}") from exc
        _raise_for_status(response, self.name)
        return response

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

# Keys Gemini's responseSchema dialect does not understand. Pydantic emits several
# of them for any non-trivial model.
_GEMINI_UNSUPPORTED = {
    "$schema",
    "$defs",
    "$ref",
    "additionalProperties",
    "definitions",
    "title",
    "default",
    "examples",
    "const",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "anyOf",
    "oneOf",
    "allOf",
}


def sanitize_schema_for_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """Reduce a Pydantic JSON Schema to the subset Gemini accepts.

    Nested Pydantic models become ``$ref`` pointers into ``$defs``, which Gemini
    rejects outright, so we inline them first and then drop the vocabulary it does
    not implement. Anything we cannot express is simply omitted: the prompt still
    describes the full schema in words and Pydantic still validates the result, so
    the worst case is a slightly less constrained decode, not a wrong answer.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, list):
            return [resolve(item) for item in node]
        if not isinstance(node, dict):
            return node

        if "$ref" in node:
            ref_name = str(node["$ref"]).rsplit("/", 1)[-1]
            return resolve(defs.get(ref_name, {"type": "object"}))

        # Optional fields arrive as anyOf[T, null]; keep the non-null branch.
        for union_key in ("anyOf", "oneOf"):
            if union_key in node:
                branches = [b for b in node[union_key] if b.get("type") != "null"]
                if branches:
                    return resolve(branches[0])

        return {k: resolve(v) for k, v in node.items() if k not in _GEMINI_UNSUPPORTED}

    result = resolve({k: v for k, v in schema.items() if k != "$defs"})
    return result if isinstance(result, dict) else {"type": "object"}


class GeminiProvider(_HTTPProvider):
    """Google AI Studio. Free tier, generous limits, the recommended default."""

    name = "gemini"

    def __init__(self, api_key: str, model: str, *, timeout_s: float = 90.0) -> None:
        super().__init__(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            timeout_s=timeout_s,
            headers={"x-goog-api-key": api_key, "content-type": "application/json"},
        )
        self.model = model

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        # Gemini takes the system prompt out of band rather than as a message.
        system_parts = [m.content for m in request.messages if m.role is Role.SYSTEM]
        contents = [
            {
                "role": "model" if m.role is Role.ASSISTANT else "user",
                "parts": [{"text": m.content}],
            }
            for m in request.messages
            if m.role is not Role.SYSTEM
        ]

        generation_config: dict[str, Any] = {"temperature": request.temperature}
        if request.max_tokens:
            generation_config["maxOutputTokens"] = request.max_tokens
        if request.json_schema:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = sanitize_schema_for_gemini(request.json_schema)

        payload: dict[str, Any] = {"contents": contents, "generationConfig": generation_config}
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

        started = time.perf_counter()
        response = await self._post(f"/models/{self.model}:generateContent", payload)
        elapsed = time.perf_counter() - started
        data = response.json()

        candidates = data.get("candidates") or []
        if not candidates:
            # Usually a safety block. Surface the reason rather than an IndexError.
            reason = (data.get("promptFeedback") or {}).get("blockReason", "unknown")
            raise InvalidResponseError(f"Gemini returned no candidates (reason: {reason})")

        candidate = candidates[0]
        text = "".join(
            part.get("text", "") for part in (candidate.get("content") or {}).get("parts", [])
        )
        meta = data.get("usageMetadata") or {}

        return CompletionResponse(
            content=text,
            model=self.model,
            usage=Usage(
                prompt_tokens=meta.get("promptTokenCount", 0),
                completion_tokens=meta.get("candidatesTokenCount", 0),
            ),
            latency_s=elapsed,
            finish_reason=candidate.get("finishReason"),
        )


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------


def _describe_schema(schema: dict[str, Any], indent: int = 0) -> list[str]:
    """Render a JSON Schema as an indented field list.

    The raw schema would also work, and is shorter to write, but it spends a lot
    of tokens on JSON Schema's own vocabulary (``properties``, ``items``,
    ``$defs``) that carries no information about the task. A plain outline of
    field names, types and enum values reads more like the examples these models
    were trained on, and leaves more budget for the resume itself.
    """
    lines: list[str] = []
    pad = "  " * indent

    for name, spec in (schema.get("properties") or {}).items():
        if not isinstance(spec, dict):
            continue

        kind = spec.get("type", "any")
        required = name in (schema.get("required") or [])
        suffix = "" if required else " (optional)"

        if enum := spec.get("enum"):
            lines.append(f'{pad}- "{name}": one of {enum}{suffix}')
        elif kind == "object":
            lines.append(f'{pad}- "{name}": object{suffix}')
            lines.extend(_describe_schema(spec, indent + 1))
        elif kind == "array":
            items = spec.get("items") or {}
            if items.get("type") == "object" or "properties" in items:
                lines.append(f'{pad}- "{name}": array of objects{suffix}')
                lines.extend(_describe_schema(items, indent + 1))
            else:
                lines.append(f'{pad}- "{name}": array of {items.get("type", "any")}{suffix}')
        else:
            lines.append(f'{pad}- "{name}": {kind}{suffix}')

    return lines


def _with_schema_instruction(
    messages: list[dict[str, str]], schema: dict[str, Any]
) -> list[dict[str, str]]:
    """Append the expected JSON shape to the final user message.

    Appended to the existing message rather than added as a new one: some
    OpenAI-compatible servers require the conversation to end on a user turn,
    and a trailing system message quietly changes behaviour on others.
    """
    resolved = sanitize_schema_for_gemini(schema)
    fields = _describe_schema(resolved)
    if not fields:
        return messages

    instruction = (
        "\n\nReturn a single JSON object with exactly these fields. "
        "Every field not marked optional must be present. "
        "Do not add fields that are not listed. Do not wrap the JSON in markdown.\n\n"
        + "\n".join(fields)
    )

    updated = list(messages)
    for index in range(len(updated) - 1, -1, -1):
        if updated[index].get("role") == "user":
            updated[index] = {**updated[index], "content": updated[index]["content"] + instruction}
            return updated

    updated.append({"role": "user", "content": instruction.strip()})
    return updated


class GroqProvider(_HTTPProvider):
    """Groq's OpenAI-compatible endpoint. Free tier, and unusually fast.

    The same request shaping works against OpenAI, Together, OpenRouter, vLLM and
    anything else that speaks the OpenAI chat API, so pointing ``base_url``
    somewhere else is all it takes to add a backend.
    """

    name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout_s: float = 90.0,
        base_url: str = "https://api.groq.com/openai/v1",
    ) -> None:
        super().__init__(
            base_url=base_url,
            timeout_s=timeout_s,
            headers={"authorization": f"Bearer {api_key}", "content-type": "application/json"},
        )
        self.model = model

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.json_schema:
            # Groq's OpenAI-compatible endpoint only offers "must parse as JSON".
            # It never sees the schema, so the model has to infer field names
            # from the prose of the prompt, and it infers them wrong: nested
            # objects get flattened, enum fields come back as free text, and
            # required keys go missing. The repair loop then burns its attempts
            # and eventually accepts a nearly empty object that happens to
            # validate, which is worse than an error because it looks like a
            # confident answer built on nothing.
            #
            # So the schema is put where the model can actually read it.
            payload["response_format"] = {"type": "json_object"}
            payload["messages"] = _with_schema_instruction(payload["messages"], request.json_schema)

        started = time.perf_counter()
        response = await self._post("/chat/completions", payload)
        elapsed = time.perf_counter() - started
        data = response.json()

        choices = data.get("choices") or []
        if not choices:
            raise InvalidResponseError("Groq returned no choices")

        choice = choices[0]
        usage = data.get("usage") or {}

        return CompletionResponse(
            content=choice["message"]["content"] or "",
            model=self.model,
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            ),
            latency_s=elapsed,
            finish_reason=choice.get("finish_reason"),
        )


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


class OllamaProvider(_HTTPProvider):
    """Local models. No key, no quota, no network.

    Worth keeping working even once you have API keys: it is the only backend that
    lets someone clone the repo and run the whole pipeline with zero setup beyond
    ``ollama pull``, and it makes the model-comparison table in the README a
    three-way rather than two-way comparison.
    """

    name = "ollama"

    def __init__(
        self,
        model: str,
        *,
        host: str = "http://localhost:11434",
        timeout_s: float = 300.0,
        context_window: int = 32768,
    ) -> None:
        # Local generation on CPU is slow, so the default timeout is far longer
        # than for the hosted providers.
        super().__init__(
            base_url=host.rstrip("/"),
            timeout_s=timeout_s,
            headers={"content-type": "application/json"},
        )
        self.model = model
        self.context_window = context_window

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                # Resumes plus a rubric routinely exceed the 2048-token default,
                # and silent truncation is the hardest bug in this pipeline to
                # notice, so we set this explicitly.
                "num_ctx": self.context_window,
            },
        }
        if request.max_tokens:
            payload["options"]["num_predict"] = request.max_tokens
        if request.seed is not None:
            payload["options"]["seed"] = request.seed
        if request.json_schema:
            # Ollama constrains decoding to the schema, the strongest guarantee
            # of the three providers.
            payload["format"] = request.json_schema

        started = time.perf_counter()
        try:
            response = await self._post("/api/chat", payload)
        except TransientProviderError as exc:
            raise TransientProviderError(
                f"Could not reach Ollama at {self._client.base_url}. Is it running? "
                f"Start it with `ollama serve`, then `ollama pull {self.model}`. ({exc})"
            ) from exc
        elapsed = time.perf_counter() - started
        data = response.json()

        return CompletionResponse(
            content=(data.get("message") or {}).get("content", ""),
            model=self.model,
            usage=Usage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
            ),
            latency_s=elapsed,
            finish_reason=data.get("done_reason"),
        )
