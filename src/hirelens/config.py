from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from hirelens._compat import StrEnum


class MissingCredentialError(RuntimeError):
    pass


class Provider(StrEnum):
    GEMINI = "gemini"
    GROQ = "groq"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HIRELENS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    llm_provider: Provider = Provider.GEMINI

    gemini_api_key: str = ""
    gemini_model: str = "gemini-flash-latest"

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    extraction_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    judge_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    self_consistency_k: int = Field(default=5, ge=1, le=15)

    request_timeout_s: float = Field(default=90.0, gt=0)
    max_retries: int = Field(default=4, ge=0, le=10)
    max_concurrent_requests: int = Field(default=4, ge=1, le=32)

    tokens_per_minute: int = Field(default=0, ge=0, le=10_000_000)

    requests_per_minute: int = Field(default=15, ge=0, le=10_000)

    cache_enabled: bool = True
    cache_dir: Path = Path(".hirelens_cache")

    blind_mode: bool = True
    max_demographic_drift: float = Field(default=2.0, ge=0.0)

    github_token: str = ""

    log_level: str = "INFO"

    def validate_credentials(self) -> None:
        required = {
            Provider.GEMINI: ("gemini_api_key", "https://aistudio.google.com/api-keys"),
            Provider.GROQ: ("groq_api_key", "https://console.groq.com/keys"),
        }
        if self.llm_provider not in required:
            return

        field, url = required[self.llm_provider]
        if not getattr(self, field):
            raise MissingCredentialError(
                f"HIRELENS_LLM_PROVIDER is '{self.llm_provider}' but "
                f"HIRELENS_{field.upper()} is empty.\n\n"
                f"  Get a free key at {url} and add it to your .env file,\n"
                f"  or set HIRELENS_LLM_PROVIDER=ollama to run fully locally with no key."
            )

    @property
    def has_credentials(self) -> bool:
        try:
            self.validate_credentials()
        except MissingCredentialError:
            return False
        return True

    @property
    def active_model(self) -> str:
        return {
            Provider.GEMINI: self.gemini_model,
            Provider.GROQ: self.groq_model,
            Provider.OLLAMA: self.ollama_model,
        }[self.llm_provider]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
