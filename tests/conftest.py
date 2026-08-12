from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from hirelens.config import Settings, get_settings

_NEUTRAL_ENVIRONMENT = {
    "HIRELENS_GEMINI_API_KEY": "",
    "HIRELENS_GROQ_API_KEY": "",
    "HIRELENS_GITHUB_TOKEN": "",
    "HIRELENS_REQUESTS_PER_MINUTE": "0",
    "HIRELENS_TOKENS_PER_MINUTE": "0",
}


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path_factory) -> Iterator[None]:  # type: ignore[no-untyped-def]
    for key in list(os.environ):
        if key.startswith("HIRELENS_"):
            monkeypatch.delenv(key, raising=False)

    for key, value in _NEUTRAL_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)

    monkeypatch.setitem(Settings.model_config, "env_file", None)

    get_settings.cache_clear()
    monkeypatch.setenv("HIRELENS_CACHE_DIR", str(tmp_path_factory.mktemp("hirelens-cache")))

    yield

    get_settings.cache_clear()
