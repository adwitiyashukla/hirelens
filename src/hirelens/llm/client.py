from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from hirelens.config import Provider, Settings, get_settings
from hirelens.llm.base import (
    CompletionRequest,
    CompletionResponse,
    InvalidResponseError,
    LLMError,
    LLMProvider,
    Message,
    RateLimitError,
    TransientProviderError,
)
from hirelens.llm.cache import ResponseCache
from hirelens.llm.providers import GeminiProvider, GroqProvider, OllamaProvider

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)

_RETRYABLE = (RateLimitError, TransientProviderError)

_CHARS_PER_TOKEN = 3.6

_ASSUMED_COMPLETION_TOKENS = 700


def estimate_tokens(request: CompletionRequest) -> int:
    prompt_chars = sum(len(message.content) for message in request.messages)
    prompt_tokens = int(prompt_chars / _CHARS_PER_TOKEN)
    completion_tokens = request.max_tokens or _ASSUMED_COMPLETION_TOKENS
    return prompt_tokens + completion_tokens


class RateLimiter:
    def __init__(self, requests_per_minute: int, tokens_per_minute: int = 0) -> None:
        self.request_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self.seconds_per_token = 60.0 / tokens_per_minute if tokens_per_minute > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_request_slot = 0.0
        self._next_token_slot = 0.0

    @property
    def enabled(self) -> bool:
        return self.request_interval > 0.0 or self.seconds_per_token > 0.0

    async def acquire(self, estimated_tokens: int = 0) -> None:
        if not self.enabled:
            return

        async with self._lock:
            now = asyncio.get_running_loop().time()
            wait = 0.0

            if self.request_interval > 0.0:
                wait = max(wait, self._next_request_slot - now)
                self._next_request_slot = max(now, self._next_request_slot) + self.request_interval

            if self.seconds_per_token > 0.0 and estimated_tokens > 0:
                wait = max(wait, self._next_token_slot - now)
                cost = estimated_tokens * self.seconds_per_token
                self._next_token_slot = max(now, self._next_token_slot) + cost

        if wait > 0:
            await asyncio.sleep(wait)


def build_provider(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    settings.validate_credentials()

    if settings.llm_provider is Provider.GEMINI:
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            timeout_s=settings.request_timeout_s,
        )
    if settings.llm_provider is Provider.GROQ:
        return GroqProvider(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            timeout_s=settings.request_timeout_s,
        )
    return OllamaProvider(
        model=settings.ollama_model,
        host=settings.ollama_host,
        timeout_s=max(settings.request_timeout_s, 300.0),
    )


class LLMClient:
    def __init__(
        self,
        provider: LLMProvider | None = None,
        *,
        settings: Settings | None = None,
        cache: ResponseCache | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider or build_provider(self.settings)
        self.cache = cache or ResponseCache(
            self.settings.cache_dir, enabled=self.settings.cache_enabled
        )
        self._semaphore = asyncio.Semaphore(self.settings.max_concurrent_requests)
        self._rate_limiter = RateLimiter(
            self.settings.requests_per_minute, self.settings.tokens_per_minute
        )
        self.call_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        cached = self.cache.get(request, self.provider.model)
        if cached is not None:
            logger.debug("cache hit (%s)", self.provider.model)
            return cached

        async with self._semaphore:
            await self._rate_limiter.acquire(estimate_tokens(request))
            response = await self._complete_with_retries(request)

        self.cache.put(request, self.provider.model, response)
        self.call_count += 1
        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_completion_tokens += response.usage.completion_tokens
        return response

    async def _complete_with_retries(self, request: CompletionRequest) -> CompletionResponse:
        last_error: Exception | None = None

        for attempt in range(self.settings.max_retries + 1):
            try:
                return await self.provider.complete(request)
            except _RETRYABLE as exc:
                last_error = exc
                if attempt == self.settings.max_retries:
                    break

                suggested = getattr(exc, "retry_after_s", None)
                delay = suggested if suggested else min(2.0**attempt, 30.0)
                delay = random.uniform(0.5 * delay, delay)

                logger.warning(
                    "%s (attempt %d/%d), retrying in %.1fs",
                    exc,
                    attempt + 1,
                    self.settings.max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
            except InvalidResponseError:
                raise

        if isinstance(last_error, RateLimitError) and self._rate_limiter.enabled:
            detail = str(last_error).lower()

            if "tokens per minute" in detail or "tpm" in detail:
                raise LLMError(
                    f"Rate limited on the provider's TOKENS-per-minute quota, not its "
                    f"request quota. Pacing is currently set by requests "
                    f"({self.settings.requests_per_minute}/minute) and "
                    f"tokens ({self.settings.tokens_per_minute or 'unset'}/minute).\n\n"
                    f"A request limit cannot control this: each call carries roughly a "
                    f"thousand tokens, so a comfortable request rate can still be several "
                    f"times over the token ceiling.\n\n"
                    f"Set HIRELENS_TOKENS_PER_MINUTE to just under your provider's limit "
                    f"(Groq's free tier is 12000, so 9000 is a safe value).\n\n"
                    f"Original error: {last_error}"
                ) from last_error

            raise LLMError(
                f"Rate limited on every attempt despite pacing at "
                f"{self.settings.requests_per_minute} requests/minute. That rules out the "
                f"per-minute request quota, so this is almost certainly the provider's DAILY "
                f"free-tier limit, which resets every 24 hours.\n\n"
                f"Options: wait for the reset, switch to a local model with "
                f"HIRELENS_LLM_PROVIDER=ollama, or use a second free provider with "
                f"HIRELENS_LLM_PROVIDER=groq.\n\n"
                f"Lowering HIRELENS_REQUESTS_PER_MINUTE will not help: the pacing is already "
                f"working."
            ) from last_error

        raise LLMError(
            f"Giving up after {self.settings.max_retries + 1} attempts: {last_error}"
        ) from last_error

    async def chat(
        self,
        *,
        system: str | None = None,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages = []
        if system:
            messages.append(Message.system(system))
        messages.append(Message.user(user))

        response = await self.complete(
            CompletionRequest(
                messages=tuple(messages),
                temperature=(
                    temperature if temperature is not None else self.settings.extraction_temperature
                ),
                max_tokens=max_tokens,
            )
        )
        return response.content

    async def structured(
        self,
        model: type[TModel],
        *,
        system: str | None = None,
        user: str,
        temperature: float | None = None,
        max_repair_attempts: int = 2,
    ) -> TModel:
        schema = model.model_json_schema()
        messages: list[Message] = []
        if system:
            messages.append(Message.system(system))
        messages.append(Message.user(user))

        temp = temperature if temperature is not None else self.settings.extraction_temperature
        last_error = ""

        for attempt in range(max_repair_attempts + 1):
            request = CompletionRequest(
                messages=tuple(messages),
                temperature=temp if attempt == 0 else max(temp, 0.2),
                json_schema=schema,
            )
            response = await self.complete(request)

            try:
                return model.model_validate(response.json())
            except (ValidationError, ValueError) as exc:
                self.cache.evict(request, self.provider.model)

                last_error = str(exc)[:1500]
                if attempt == max_repair_attempts:
                    break

                logger.info(
                    "structured output failed validation (attempt %d/%d), repairing",
                    attempt + 1,
                    max_repair_attempts,
                )
                messages.extend(
                    [
                        Message.assistant(response.content),
                        Message.user(
                            "That response did not validate against the required schema.\n\n"
                            f"Validation errors:\n{last_error}\n\n"
                            "Return corrected JSON only. No explanation, no markdown fences, "
                            "no additional keys. Every required field must be present."
                        ),
                    ]
                )

        raise InvalidResponseError(
            f"Could not obtain valid {model.__name__} after "
            f"{max_repair_attempts + 1} attempts. Last error: {last_error}"
        )

    async def gather(self, requests: list[CompletionRequest]) -> list[CompletionResponse]:
        return list(await asyncio.gather(*(self.complete(r) for r in requests)))

    def usage_summary(self) -> dict[str, Any]:
        return {
            "provider": self.provider.name,
            "model": self.provider.model,
            "api_calls": self.call_count,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "cache": self.cache.stats(),
            "requests_per_minute": self.settings.requests_per_minute,
        }

    async def aclose(self) -> None:
        await self.provider.aclose()

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
