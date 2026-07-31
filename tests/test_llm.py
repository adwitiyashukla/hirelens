"""LLM layer tests: JSON recovery, caching, retries, and the repair loop.

Everything here runs against a fake provider. No network, no API key, no cost, so
these run in CI on every push.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from hirelens.config import Settings
from hirelens.llm.base import (
    CompletionRequest,
    CompletionResponse,
    InvalidResponseError,
    LLMProvider,
    Message,
    RateLimitError,
    TransientProviderError,
    Usage,
    extract_json,
)
from hirelens.llm.cache import ResponseCache
from hirelens.llm.client import LLMClient, RateLimiter, estimate_tokens
from hirelens.llm.providers import (
    _describe_schema,
    _with_schema_instruction,
    sanitize_schema_for_gemini,
)


class FakeProvider(LLMProvider):
    """Returns scripted responses and counts calls."""

    name = "fake"
    model = "fake-model"

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        item = self._responses.pop(0) if self._responses else "{}"
        if isinstance(item, Exception):
            raise item
        return CompletionResponse(
            content=item, model=self.model, usage=Usage(prompt_tokens=10, completion_tokens=5)
        )

    async def aclose(self) -> None:
        return None


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "llm_provider": "ollama",  # no credential check
        "cache_dir": tmp_path / "cache",
        "cache_enabled": True,
        "max_retries": 2,
        "max_concurrent_requests": 4,
        # No pacing: these hit a fake provider, so throttling only slows CI.
        "requests_per_minute": 0,
    }
    return Settings(**{**defaults, **overrides})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_plain_object(self) -> None:
        assert extract_json('{"score": 7}') == '{"score": 7}'

    def test_markdown_fence(self) -> None:
        assert extract_json('```json\n{"score": 7}\n```') == '{"score": 7}'

    def test_bare_fence(self) -> None:
        assert extract_json('```\n{"score": 7}\n```') == '{"score": 7}'

    def test_reasoning_block_is_stripped(self) -> None:
        raw = '<think>Let me weigh the evidence carefully.</think>\n{"score": 7}'
        assert extract_json(raw) == '{"score": 7}'

    def test_surrounding_prose(self) -> None:
        raw = 'Here is the evaluation you asked for:\n{"score": 7}\nHope that helps.'
        assert extract_json(raw) == '{"score": 7}'

    def test_array_response(self) -> None:
        assert extract_json("Results: [1, 2, 3] done") == "[1, 2, 3]"

    def test_nested_braces_survive(self) -> None:
        raw = '{"scores": {"open_source": {"score": 7}}}'
        assert extract_json(raw) == raw

    def test_unrecoverable_input_is_returned_unchanged(self) -> None:
        """So the caller gets a real JSONDecodeError with context, not an empty string."""
        assert extract_json("no json here at all") == "no json here at all"


# ---------------------------------------------------------------------------


class TestResponseCache:
    def test_miss_then_hit(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path / "c")
        request = CompletionRequest(messages=(Message.user("hello"),))
        response = CompletionResponse(content="hi", model="m")

        assert cache.get(request, "m") is None
        cache.put(request, "m", response)

        hit = cache.get(request, "m")
        assert hit is not None
        assert hit.content == "hi"
        assert hit.cached is True
        assert cache.hit_rate == 0.5

    def test_key_depends_on_temperature(self, tmp_path: Path) -> None:
        """Self-consistency sampling must not collapse into one cached answer."""
        cache = ResponseCache(tmp_path / "c")
        cold = CompletionRequest(messages=(Message.user("x"),), temperature=0.0)
        warm = CompletionRequest(messages=(Message.user("x"),), temperature=0.7)
        assert cache.key_for(cold, "m") != cache.key_for(warm, "m")

    def test_key_depends_on_model(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path / "c")
        request = CompletionRequest(messages=(Message.user("x"),))
        assert cache.key_for(request, "gemini") != cache.key_for(request, "llama")

    def test_disabled_cache_is_a_no_op(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path / "c", enabled=False)
        request = CompletionRequest(messages=(Message.user("x"),))
        cache.put(request, "m", CompletionResponse(content="y", model="m"))
        assert cache.get(request, "m") is None

    def test_corrupt_entry_is_treated_as_a_miss(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path / "c")
        request = CompletionRequest(messages=(Message.user("x"),))
        cache.put(request, "m", CompletionResponse(content="y", model="m"))

        path = cache._path_for(cache.key_for(request, "m"))
        path.write_text("{ truncated", encoding="utf-8")

        assert cache.get(request, "m") is None
        assert not path.exists()


# ---------------------------------------------------------------------------


class Assessment(BaseModel):
    score: int = Field(ge=0, le=10)
    rationale: str = Field(min_length=1)


class TestLLMClient:
    async def test_second_identical_call_is_served_from_cache(self, tmp_path: Path) -> None:
        provider = FakeProvider(['{"score": 8, "rationale": "solid"}'])
        client = LLMClient(provider, settings=make_settings(tmp_path))

        request = CompletionRequest(messages=(Message.user("evaluate"),))
        await client.complete(request)
        await client.complete(request)

        assert len(provider.calls) == 1
        assert client.cache.hits == 1

    async def test_retries_on_rate_limit_then_succeeds(self, tmp_path: Path) -> None:
        provider = FakeProvider(
            [RateLimitError("429", retry_after_s=0.01), '{"score": 5, "rationale": "ok"}']
        )
        client = LLMClient(provider, settings=make_settings(tmp_path))

        response = await client.complete(CompletionRequest(messages=(Message.user("x"),)))
        assert response.content == '{"score": 5, "rationale": "ok"}'
        assert len(provider.calls) == 2

    async def test_persistent_rate_limiting_while_pacing_is_reported_as_a_daily_cap(
        self, tmp_path: Path
    ) -> None:
        """Pacing makes a per-minute breach impossible, so the cause must be daily.

        Without this, the user sees a wall of 429s and reasonably concludes their
        settings are wrong, then spends an evening lowering a number that cannot
        possibly help.
        """
        provider = FakeProvider([RateLimitError("429", retry_after_s=0.001)] * 5)
        settings = make_settings(tmp_path, max_retries=1, requests_per_minute=6000)
        client = LLMClient(provider, settings=settings)

        with pytest.raises(Exception, match="DAILY"):
            await client.complete(CompletionRequest(messages=(Message.user("x"),)))

    async def test_gives_up_after_max_retries(self, tmp_path: Path) -> None:
        provider = FakeProvider([TransientProviderError("500")] * 5)
        client = LLMClient(provider, settings=make_settings(tmp_path, max_retries=1))

        with pytest.raises(Exception, match="Giving up"):
            await client.complete(CompletionRequest(messages=(Message.user("x"),)))
        assert len(provider.calls) == 2

    async def test_bad_credentials_are_not_retried(self, tmp_path: Path) -> None:
        """Retrying a 401 four times just delays the same error message."""
        provider = FakeProvider([InvalidResponseError("401 bad key")] * 3)
        client = LLMClient(provider, settings=make_settings(tmp_path))

        with pytest.raises(InvalidResponseError):
            await client.complete(CompletionRequest(messages=(Message.user("x"),)))
        assert len(provider.calls) == 1

    async def test_structured_output_validates(self, tmp_path: Path) -> None:
        provider = FakeProvider(['{"score": 7, "rationale": "shipped to production"}'])
        client = LLMClient(provider, settings=make_settings(tmp_path))

        result = await client.structured(Assessment, user="score this")
        assert result.score == 7

    async def test_repair_loop_recovers_from_a_validation_error(self, tmp_path: Path) -> None:
        """The behaviour that separates a demo from something usable."""
        provider = FakeProvider(
            [
                '{"score": 99, "rationale": "out of range"}',  # violates le=10
                '{"score": 9, "rationale": "corrected"}',
            ]
        )
        client = LLMClient(provider, settings=make_settings(tmp_path))

        result = await client.structured(Assessment, user="score this")
        assert result.score == 9
        assert len(provider.calls) == 2

        # The repair turn must show the model its own output and the error.
        repair_prompt = provider.calls[1].messages[-1].content
        assert "did not validate" in repair_prompt
        assert "score" in repair_prompt

    async def test_structured_raises_after_exhausting_repairs(self, tmp_path: Path) -> None:
        provider = FakeProvider(['{"score": 99, "rationale": "nope"}'] * 5)
        client = LLMClient(provider, settings=make_settings(tmp_path))

        with pytest.raises(InvalidResponseError, match="Could not obtain valid Assessment"):
            await client.structured(Assessment, user="x", max_repair_attempts=1)

    async def test_usage_summary_tracks_tokens(self, tmp_path: Path) -> None:
        provider = FakeProvider(['{"score": 1, "rationale": "a"}'])
        client = LLMClient(provider, settings=make_settings(tmp_path))
        await client.complete(CompletionRequest(messages=(Message.user("x"),)))

        summary = client.usage_summary()
        assert summary["api_calls"] == 1
        assert summary["prompt_tokens"] == 10

    async def test_gather_runs_many_requests(self, tmp_path: Path) -> None:
        provider = FakeProvider([f'{{"i": {i}}}' for i in range(6)])
        client = LLMClient(provider, settings=make_settings(tmp_path))

        requests = [CompletionRequest(messages=(Message.user(f"q{i}"),)) for i in range(6)]
        responses = await client.gather(requests)
        assert len(responses) == 6


# ---------------------------------------------------------------------------


class Nested(BaseModel):
    label: str


class Outer(BaseModel):
    name: str
    nested: Nested
    optional_note: str | None = None


class TestGeminiSchemaSanitizer:
    def test_refs_are_inlined(self) -> None:
        """Gemini rejects $ref, and Pydantic emits one for every nested model."""
        cleaned = sanitize_schema_for_gemini(Outer.model_json_schema())
        serialised = str(cleaned)
        assert "$ref" not in serialised
        assert "$defs" not in serialised
        assert cleaned["properties"]["nested"]["properties"]["label"]["type"] == "string"

    def test_unsupported_keywords_are_dropped(self) -> None:
        cleaned = sanitize_schema_for_gemini(Outer.model_json_schema())
        serialised = str(cleaned)
        for keyword in ("additionalProperties", "$schema", "anyOf"):
            assert keyword not in serialised

    def test_optional_fields_keep_their_non_null_type(self) -> None:
        cleaned = sanitize_schema_for_gemini(Outer.model_json_schema())
        assert cleaned["properties"]["optional_note"]["type"] == "string"

    def test_core_structure_survives(self) -> None:
        cleaned = sanitize_schema_for_gemini(Outer.model_json_schema())
        assert cleaned["type"] == "object"
        assert set(cleaned["properties"]) == {"name", "nested", "optional_note"}


class TestGroqSchemaInPrompt:
    """Groq cannot be sent a schema, so it has to be told one in words.

    These guard a bug that cost a full evening. Groq's endpoint accepts only
    ``{"type": "json_object"}``, meaning "valid JSON, shape unspecified". The
    model then guessed field names, the repair loop burned its attempts, and it
    converged on a sparse object that validated because the absent fields had
    defaults. A strong candidate scored 0 out of 100 while every quality metric
    read 100%.
    """

    @staticmethod
    def _schema() -> dict[str, object]:
        class Inner(BaseModel):
            label: str
            weight: float = 1.0

        class Outer(BaseModel):
            name: str
            kind: str = Field(json_schema_extra={"enum": ["must_have", "nice_to_have"]})
            items: list[Inner]

        return Outer.model_json_schema()

    def test_field_names_and_types_reach_the_prompt(self) -> None:
        described = "\n".join(_describe_schema(sanitize_schema_for_gemini(self._schema())))
        assert '"name": string' in described
        assert '"items": array of objects' in described

    def test_nested_object_fields_are_described(self) -> None:
        """The flattening of nested models was half the original failure."""
        described = "\n".join(_describe_schema(sanitize_schema_for_gemini(self._schema())))
        assert '"label"' in described
        assert '"weight"' in described

    def test_enum_values_are_listed(self) -> None:
        """Exactly the failure that produced a rubric with no must-haves."""
        described = "\n".join(_describe_schema(sanitize_schema_for_gemini(self._schema())))
        assert "must_have" in described and "nice_to_have" in described

    def test_optional_fields_are_marked(self) -> None:
        described = "\n".join(_describe_schema(sanitize_schema_for_gemini(self._schema())))
        assert "(optional)" in described

    def test_instruction_is_appended_to_the_last_user_message(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "score this"},
        ]
        updated = _with_schema_instruction(messages, self._schema())

        assert len(updated) == 2
        assert updated[0] == {"role": "system", "content": "sys"}
        assert updated[1]["content"].startswith("score this")
        assert '"name": string' in updated[1]["content"]

    def test_original_messages_are_not_mutated(self) -> None:
        messages = [{"role": "user", "content": "original"}]
        _with_schema_instruction(messages, self._schema())
        assert messages[0]["content"] == "original"

    def test_repair_turn_still_gets_the_schema(self) -> None:
        """The repair turn ends on a user message, so it must be the one patched."""
        messages = [
            {"role": "user", "content": "first ask"},
            {"role": "assistant", "content": "{bad}"},
            {"role": "user", "content": "that did not validate"},
        ]
        updated = _with_schema_instruction(messages, self._schema())

        assert updated[0]["content"] == "first ask"
        assert '"name": string' in updated[2]["content"]

    def test_word_json_is_present_for_the_response_format(self) -> None:
        """OpenAI-compatible json_object mode 400s unless the prompt says "json"."""
        updated = _with_schema_instruction([{"role": "user", "content": "x"}], self._schema())
        assert "JSON" in updated[0]["content"]

    def test_empty_schema_leaves_the_prompt_alone(self) -> None:
        messages = [{"role": "user", "content": "x"}]
        assert _with_schema_instruction(messages, {"type": "object"}) == messages


class TestTokenAwarePacing:
    """The binding quota on some providers is tokens, not requests.

    Written after a real failure. Pacing at 25 requests/minute against Groq's
    12,000 tokens/minute ceiling aims for roughly 25,000 TPM. Half the calls
    were rejected, extraction exhausted its retries, and a strong candidate came
    back with one evidence unit and was reported as a weak match. A quota
    shortfall had turned into a hiring signal, which is the exact failure class
    this project exists to prevent.
    """

    def test_estimate_counts_prompt_and_completion(self) -> None:
        request = CompletionRequest(messages=(Message.user("x" * 3600),), max_tokens=500)
        # 3600 chars / 3.6 = 1000 prompt tokens, plus the explicit completion.
        assert estimate_tokens(request) == 1500

    def test_estimate_assumes_a_completion_when_max_tokens_is_unset(self) -> None:
        """Quotas count output too, so assuming zero would under-pace badly."""
        request = CompletionRequest(messages=(Message.user("x" * 360),))
        assert estimate_tokens(request) > 100

    def test_estimate_includes_every_message(self) -> None:
        """The repair turn resends the whole conversation, so it costs more."""
        one = CompletionRequest(messages=(Message.user("x" * 360),), max_tokens=10)
        three = CompletionRequest(
            messages=(
                Message.system("x" * 360),
                Message.user("x" * 360),
                Message.assistant("x" * 360),
            ),
            max_tokens=10,
        )
        assert estimate_tokens(three) > estimate_tokens(one) * 2

    def test_a_token_only_limiter_is_enabled(self) -> None:
        assert RateLimiter(requests_per_minute=0, tokens_per_minute=9000).enabled

    def test_both_quotas_off_means_no_pacing(self) -> None:
        assert not RateLimiter(requests_per_minute=0, tokens_per_minute=0).enabled

    def test_large_requests_reserve_proportionally_more_quota(self) -> None:
        """A 2000-token call must consume twice the minute a 1000-token one does."""
        limiter = RateLimiter(requests_per_minute=0, tokens_per_minute=6000)
        # 6000 tokens per 60s is 0.01 seconds of quota per token.
        assert limiter.seconds_per_token == pytest.approx(0.01)

    async def test_token_pacing_actually_delays(self) -> None:
        """The reservation has to produce real waiting, not just bookkeeping.

        Scaled so the whole test costs a fraction of a second. A version of this
        that paced at a realistic 9,000 tokens/minute would need to sleep for
        twenty seconds to prove the same property, and a test suite nobody wants
        to run is a test suite that stops being run.
        """
        # 600k tokens/minute is 10,000 per second, so 2,000 tokens costs 0.2s.
        limiter = RateLimiter(requests_per_minute=0, tokens_per_minute=600_000)
        loop = asyncio.get_running_loop()

        started = loop.time()
        for _ in range(3):
            await limiter.acquire(estimated_tokens=2_000)
        elapsed = loop.time() - started

        # The first call reserves but does not wait, leaving two waits of 0.2s.
        assert elapsed >= 0.3

    async def test_a_token_quota_breach_names_the_right_setting(self, tmp_path: Path) -> None:
        """The previous message blamed the daily limit for every persistent 429.

        Someone following that advice would wait 24 hours for a problem that one
        setting fixes in a second.
        """
        provider = FakeProvider(
            [
                RateLimitError(
                    "Rate limit reached for model in organization on tokens per minute "
                    "(TPM): Limit 12000, Used 11172",
                    retry_after_s=0.001,
                )
            ]
            * 5
        )
        settings = make_settings(tmp_path, max_retries=1, requests_per_minute=25)
        client = LLMClient(provider, settings=settings)

        with pytest.raises(Exception, match="HIRELENS_TOKENS_PER_MINUTE"):
            await client.complete(CompletionRequest(messages=(Message.user("x"),)))

    async def test_a_request_quota_breach_still_reports_a_daily_cap(self, tmp_path: Path) -> None:
        provider = FakeProvider([RateLimitError("429 quota exceeded", retry_after_s=0.001)] * 5)
        settings = make_settings(tmp_path, max_retries=1, requests_per_minute=6000)
        client = LLMClient(provider, settings=settings)

        with pytest.raises(Exception, match="DAILY"):
            await client.complete(CompletionRequest(messages=(Message.user("x"),)))


class TestBadResponsesAreNotCached:
    """A response that fails validation must not survive in the cache.

    Found the hard way. A rate-limited run cached some malformed extraction
    responses. Every later run for that resume then replayed them, produced
    almost no evidence, and reported a strong candidate as a weak match. There
    was no error to see: the prompt was deterministic, so the failure was
    perfectly reproducible and looked like a considered judgement.
    """

    async def test_a_failing_response_is_evicted(self, tmp_path: Path) -> None:
        provider = FakeProvider(
            [
                '{"score": 99, "rationale": "out of range"}',  # violates le=10
                '{"score": 8, "rationale": "corrected"}',
            ]
        )
        client = LLMClient(provider, settings=make_settings(tmp_path))

        result = await client.structured(Assessment, user="score this")
        assert result.score == 8

        # The first, invalid response must be gone. If it were still cached, a
        # fresh client would replay it and fail forever.
        first_request = CompletionRequest(
            messages=(Message.user("score this"),),
            temperature=client.settings.extraction_temperature,
            json_schema=Assessment.model_json_schema(),
        )
        assert client.cache.get(first_request, provider.model) is None

    async def test_a_second_run_recovers_instead_of_replaying_the_failure(
        self, tmp_path: Path
    ) -> None:
        """The property that actually matters: the bug does not become permanent."""
        settings = make_settings(tmp_path)

        first = FakeProvider(['{"score": 99, "rationale": "bad"}'] * 4)
        with pytest.raises(InvalidResponseError):
            await LLMClient(first, settings=settings).structured(
                Assessment, user="score this", max_repair_attempts=1
            )

        # A new client, same cache directory, same prompt, healthy provider.
        second = FakeProvider(['{"score": 7, "rationale": "fine"}'])
        result = await LLMClient(second, settings=settings).structured(
            Assessment, user="score this"
        )

        assert result.score == 7
        assert len(second.calls) == 1, "the poisoned entry was replayed instead of refetched"

    async def test_valid_responses_are_still_cached(self, tmp_path: Path) -> None:
        """Eviction must not turn into "never cache anything"."""
        provider = FakeProvider(['{"score": 5, "rationale": "ok"}'])
        settings = make_settings(tmp_path)

        await LLMClient(provider, settings=settings).structured(Assessment, user="x")
        second = FakeProvider(['{"score": 1, "rationale": "should not be reached"}'])
        result = await LLMClient(second, settings=settings).structured(Assessment, user="x")

        assert result.score == 5
        assert second.calls == []

    def test_evicting_a_missing_entry_is_harmless(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path / "c")
        assert cache.evict(CompletionRequest(messages=(Message.user("x"),)), "m") is False

    def test_a_disabled_cache_evicts_nothing(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path / "c", enabled=False)
        assert cache.evict(CompletionRequest(messages=(Message.user("x"),)), "m") is False
