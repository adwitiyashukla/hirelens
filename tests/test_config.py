from __future__ import annotations

import pytest

from hirelens.config import MissingCredentialError, Provider, Settings


class TestProviderSelection:
    def test_active_model_follows_the_provider(self) -> None:
        settings = Settings(
            llm_provider=Provider.GROQ,
            groq_api_key="k",
            groq_model="llama-3.3-70b-versatile",
            gemini_model="gemini-2.5-flash",
        )
        assert settings.active_model == "llama-3.3-70b-versatile"

    def test_provider_formats_as_a_plain_string(self) -> None:
        assert f"{Provider.GEMINI}" == "gemini"

    def test_provider_accepts_its_string_value(self) -> None:
        assert Settings(llm_provider="ollama").llm_provider is Provider.OLLAMA  # type: ignore[arg-type]


class TestCredentials:
    def test_settings_construct_without_any_key(self) -> None:
        settings = Settings(llm_provider=Provider.GEMINI, gemini_api_key="")
        assert settings.llm_provider is Provider.GEMINI
        assert settings.has_credentials is False

    def test_validate_raises_for_missing_gemini_key(self) -> None:
        settings = Settings(llm_provider=Provider.GEMINI, gemini_api_key="")
        with pytest.raises(MissingCredentialError, match="GEMINI_API_KEY"):
            settings.validate_credentials()

    def test_error_message_points_at_both_ways_out(self) -> None:
        settings = Settings(llm_provider=Provider.GROQ, groq_api_key="")
        with pytest.raises(MissingCredentialError) as exc:
            settings.validate_credentials()
        message = str(exc.value)
        assert "console.groq.com" in message
        assert "ollama" in message.lower()

    def test_ollama_needs_no_credential(self) -> None:
        settings = Settings(llm_provider=Provider.OLLAMA)
        settings.validate_credentials()
        assert settings.has_credentials is True

    def test_present_key_passes(self) -> None:
        settings = Settings(llm_provider=Provider.GEMINI, gemini_api_key="abc123")
        settings.validate_credentials()
        assert settings.has_credentials is True


class TestBounds:
    @pytest.mark.parametrize("k", [0, 16])
    def test_self_consistency_k_is_bounded(self, k: int) -> None:
        with pytest.raises(ValueError):
            Settings(self_consistency_k=k)

    def test_temperature_is_bounded(self) -> None:
        with pytest.raises(ValueError):
            Settings(judge_temperature=5.0)

    def test_defaults_are_sensible(self) -> None:
        settings = Settings(llm_provider=Provider.OLLAMA)
        assert settings.extraction_temperature == 0.0
        assert settings.judge_temperature > 0.0
        assert settings.blind_mode is True
