"""Central configuration.

Every tunable lives here and is loaded from the environment or a ``.env`` file, so
nothing in the codebase reads ``os.environ`` directly. That matters more than it
sounds: the evaluation harness sweeps these values (model, temperature, k) to
produce the comparison tables in the README, and it can only do that if there is
exactly one place to change them.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from hirelens._compat import StrEnum


class MissingCredentialError(RuntimeError):
    """The selected provider needs an API key that is not configured."""


class Provider(StrEnum):
    """LLM backends we support.

    All three are usable for free. ``OLLAMA`` needs no key and no internet.
    """

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

    # ---- provider selection ------------------------------------------------
    llm_provider: Provider = Provider.GEMINI

    gemini_api_key: str = ""
    # An alias rather than a pinned version. Google retires pinned names and
    # stops serving them to newly created keys, which surfaces as a bare 404
    # with nothing in it about retirement. Defaulting to a pin means the project
    # silently stops working for anyone who clones it after the next rotation.
    gemini_model: str = "gemini-flash-latest"

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"

    # ---- embeddings (local, no API) ---------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # ---- generation behaviour ---------------------------------------------
    # Extraction is a transcription task, so we want it deterministic.
    extraction_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    # Judging is sampled on purpose: self-consistency needs variation to
    # measure. A temperature of zero would give a fake confidence band.
    judge_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    self_consistency_k: int = Field(default=5, ge=1, le=15)

    request_timeout_s: float = Field(default=90.0, gt=0)
    max_retries: int = Field(default=4, ge=0, le=10)
    max_concurrent_requests: int = Field(default=4, ge=1, le=32)

    #: Client-side pacing, in requests per minute. Free tiers enforce a
    #: per-minute quota (Gemini's is currently 20 for the flash models), and a
    #: burst of concurrent calls trips it in seconds.
    #:
    #: Retrying into a rate limit is the wrong fix: the backoff eventually
    #: exhausts its attempts and the requirement is recorded as "judging failed",
    #: which is indistinguishable in the output from "no evidence found". Pacing
    #: requests so the quota is never hit means the run is slower and correct
    #: rather than fast and wrong.
    #:
    #: Set to 0 to disable pacing entirely.
    #: Token quota, which is what Groq actually meters (12,000/minute free).
    #: Zero disables token pacing. Setting only ``requests_per_minute`` against a
    #: token-metered provider silently fails half the calls in a large run.
    tokens_per_minute: int = Field(default=0, ge=0, le=10_000_000)

    requests_per_minute: int = Field(default=15, ge=0, le=10_000)

    # ---- caching -----------------------------------------------------------
    cache_enabled: bool = True
    cache_dir: Path = Path(".hirelens_cache")

    # ---- screening policy --------------------------------------------------
    blind_mode: bool = True
    max_demographic_drift: float = Field(default=2.0, ge=0.0)

    # ---- enrichment --------------------------------------------------------
    github_token: str = ""

    # ---- misc --------------------------------------------------------------
    log_level: str = "INFO"

    # -----------------------------------------------------------------------

    def validate_credentials(self) -> None:
        """Raise if the selected provider has no usable credential.

        Deliberately *not* a Pydantic validator. Making it one meant that merely
        constructing ``Settings`` required an API key, so ``hirelens ingest``
        refused to parse a PDF despite never calling a model. Anyone cloning the
        repo hit an auth error on their first command, which is a terrible first
        impression and an obviously wrong coupling.

        Credentials are a requirement of *talking to a provider*, so the check
        belongs at the point a provider is constructed. See
        :func:`hirelens.llm.client.build_provider`.
        """
        required = {
            Provider.GEMINI: ("gemini_api_key", "https://aistudio.google.com/api-keys"),
            Provider.GROQ: ("groq_api_key", "https://console.groq.com/keys"),
        }
        if self.llm_provider not in required:
            return  # Ollama needs nothing.

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
        """Whether the selected provider is usable, without raising."""
        try:
            self.validate_credentials()
        except MissingCredentialError:
            return False
        return True

    @property
    def active_model(self) -> str:
        """The model name for the currently selected provider."""
        return {
            Provider.GEMINI: self.gemini_model,
            Provider.GROQ: self.groq_model,
            Provider.OLLAMA: self.ollama_model,
        }[self.llm_provider]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached so that importing this module a hundred times does not re-read the
    ``.env`` file a hundred times. Call ``get_settings.cache_clear()`` in tests
    that need to change the environment.
    """
    return Settings()
