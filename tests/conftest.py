"""Test isolation from the developer's machine.

Pydantic-settings fills any field a test does not explicitly set from ``.env``
and from the process environment. That coupling is invisible until it bites,
and it bit: adding ``HIRELENS_TOKENS_PER_MINUTE=9000`` to a local ``.env`` made
the whole suite start pacing against a real token budget, and tests that had
run in milliseconds began sleeping for minutes.

The failure mode is worse than slow tests. A suite that reads local
configuration can pass on one machine and fail on another for reasons nobody
can see from the code, and it can pass in CI while being broken locally. A test
should depend on its own inputs and nothing else.

This fixture is autouse, so it applies to every test in the suite without any
test having to remember.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from hirelens.config import Settings, get_settings

#: Set during tests so any code path that reads the environment directly, rather
#: than going through Settings, is also insulated.
_NEUTRAL_ENVIRONMENT = {
    # No provider credentials: a test must never be able to reach a real API,
    # accidentally or otherwise.
    "HIRELENS_GEMINI_API_KEY": "",
    "HIRELENS_GROQ_API_KEY": "",
    "HIRELENS_GITHUB_TOKEN": "",
    # No pacing. Every test runs against a fake provider, so throttling only
    # buys wall-clock time.
    "HIRELENS_REQUESTS_PER_MINUTE": "0",
    "HIRELENS_TOKENS_PER_MINUTE": "0",
}


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path_factory) -> Iterator[None]:  # type: ignore[no-untyped-def]
    """Cut every test off from ``.env`` and from inherited HIRELENS_ variables."""
    # Anything the developer exported in their shell.
    for key in list(os.environ):
        if key.startswith("HIRELENS_"):
            monkeypatch.delenv(key, raising=False)

    for key, value in _NEUTRAL_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)

    # Stop pydantic-settings reading the repository's .env. Tests that construct
    # Settings directly get their own values and defaults, nothing else.
    monkeypatch.setitem(Settings.model_config, "env_file", None)

    # The cached global settings object would otherwise carry configuration
    # between tests, and out of a previous test's tmp_path.
    get_settings.cache_clear()
    monkeypatch.setenv("HIRELENS_CACHE_DIR", str(tmp_path_factory.mktemp("hirelens-cache")))

    yield

    get_settings.cache_clear()
