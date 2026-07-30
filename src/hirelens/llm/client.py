"""The client every other module uses: caching, retries, concurrency, schema repair.

Providers in :mod:`hirelens.llm.providers` are deliberately dumb, they shape one
request and parse one response. All the reliability behaviour lives here, in one
place, so it applies identically to whichever backend is configured.

The interesting method is :meth:`LLMClient.structured`, which implements the
repair loop. Asking a model for JSON and calling ``json.loads`` on the result is
where most LLM projects quietly break: roughly a few percent of calls come back
with a missing required field, a string where a number belongs, or a stray
trailing comma. Crashing on those makes the tool feel broken; silently accepting
them corrupts the data. The third option, showing the model its own validation
error and asking it to fix it, recovers the large majority of failures and is what
production systems actually do.
"""

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


class RateLimiter:
    """Paces outgoing requests to stay under a per-minute quota.

    A simple leaky bucket: each request waits until at least ``interval`` seconds
    have passed since the previous one started. Not a token bucket, deliberately,
    because a token bucket permits an initial burst and the burst is exactly what
    trips a per-minute quota when a run fans out twenty judge calls at once.

    This exists because retrying into a rate limit does not work. The backoff
    eventually runs out of attempts, the requirement is recorded as "judging
    failed", and that is indistinguishable in the final report from "no evidence
    found". A candidate can lose points to a quota. Pacing makes the run slower
    and correct instead.
    """

    def __init__(self, requests_per_minute: int) -> None:
        self.interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_slot = 0.0

    @property
    def enabled(self) -> bool:
        return self.interval > 0.0

    async def acquire(self) -> None:
        if not self.enabled:
            return

        async with self._lock:
            now = asyncio.get_running_loop().time()
            wait = max(0.0, self._next_slot - now)
            # Reserve the slot before releasing the lock, so concurrent callers
            # queue up behind each other rather than all reading the same "now".
            self._next_slot = max(now, self._next_slot) + self.interval

        if wait:
            await asyncio.sleep(wait)


def build_provider(settings: Settings | None = None) -> LLMProvider:
    """Instantiate the backend named in configuration.

    This is the point where credentials actually become necessary, so this is
    where they are checked.
    """
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
    """Reliability wrapper around a provider.

    Responsibilities, in the order a request meets them:

    1. cache lookup
    2. concurrency limit (a semaphore, so free-tier per-minute quotas survive a
       batch of resumes)
    3. call with exponential backoff and jitter on retryable errors
    4. cache write
    5. optional Pydantic validation with a repair round-trip
    """

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
        self._rate_limiter = RateLimiter(self.settings.requests_per_minute)
        self.call_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    # -- core ----------------------------------------------------------------

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        cached = self.cache.get(request, self.provider.model)
        if cached is not None:
            logger.debug("cache hit (%s)", self.provider.model)
            return cached

        async with self._semaphore:
            # Pace before the call, not after a 429. Cache hits above never reach
            # here, so a warm cache costs nothing in wall-clock time.
            await self._rate_limiter.acquire()
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

                # Honour Retry-After when the provider sends one, otherwise back
                # off exponentially. Full jitter, because a batch of resumes hits
                # the quota at the same instant and un-jittered retries would all
                # come back in lockstep and trip it again.
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
                # Bad key or malformed request. Retrying changes nothing.
                raise

        # A rate limit that survives every retry *while pacing is on* is almost
        # certainly a daily cap, not a per-minute one: the pacer makes a
        # per-minute breach arithmetically impossible. Saying so turns a
        # confusing wall of 429s into a clear "come back tomorrow", instead of
        # sending someone off to tune settings that cannot help.
        if isinstance(last_error, RateLimitError) and self._rate_limiter.enabled:
            raise LLMError(
                f"Rate limited on every attempt despite pacing at "
                f"{self.settings.requests_per_minute} requests/minute. That rules out the "
                f"per-minute quota, so this is almost certainly the provider's DAILY free-tier "
                f"limit, which resets every 24 hours.\n\n"
                f"Options: wait for the reset, switch to a local model with "
                f"HIRELENS_LLM_PROVIDER=ollama, or use a second free provider with "
                f"HIRELENS_LLM_PROVIDER=groq.\n\n"
                f"Lowering HIRELENS_REQUESTS_PER_MINUTE will not help: the pacing is already "
                f"working."
            ) from last_error

        raise LLMError(
            f"Giving up after {self.settings.max_retries + 1} attempts: {last_error}"
        ) from last_error

    # -- convenience ---------------------------------------------------------

    async def chat(
        self,
        *,
        system: str | None = None,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Plain text completion."""
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
        """Get a validated Pydantic object back, repairing malformed replies.

        On a validation failure we send the model its own output plus the exact
        Pydantic error and ask for a correction. In practice one repair round
        fixes the overwhelming majority of failures, because the errors are almost
        always mechanical (a missing field, a string where an int belongs) rather
        than conceptual.

        Raises :class:`InvalidResponseError` if the output is still invalid after
        ``max_repair_attempts``, so callers can count and report the failure rate
        instead of receiving silently wrong data.
        """
        schema = model.model_json_schema()
        messages: list[Message] = []
        if system:
            messages.append(Message.system(system))
        messages.append(Message.user(user))

        temp = temperature if temperature is not None else self.settings.extraction_temperature
        last_error = ""

        for attempt in range(max_repair_attempts + 1):
            response = await self.complete(
                CompletionRequest(
                    messages=tuple(messages),
                    # Nudge the temperature up on a retry: a deterministic decode
                    # that produced invalid JSON will reproduce it exactly.
                    temperature=temp if attempt == 0 else max(temp, 0.2),
                    json_schema=schema,
                )
            )

            try:
                return model.model_validate(response.json())
            except (ValidationError, ValueError) as exc:
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

    # -- fan-out -------------------------------------------------------------

    async def gather(self, requests: list[CompletionRequest]) -> list[CompletionResponse]:
        """Run many completions concurrently, bounded by the semaphore."""
        return list(await asyncio.gather(*(self.complete(r) for r in requests)))

    # -- reporting -----------------------------------------------------------

    def usage_summary(self) -> dict[str, Any]:
        """Token and cache accounting. Feeds the cost column of the eval table."""
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
