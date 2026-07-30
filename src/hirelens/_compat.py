"""Small compatibility shims.

Python 3.11 added :class:`enum.StrEnum`, which is exactly what we want for config
values: an enum whose members compare and format as plain strings, so
``f"{settings.llm_provider}"`` prints ``gemini`` rather than ``Provider.GEMINI``.

Rather than requiring 3.11 for one convenience class, we use it when available and
fall back to the equivalent ``(str, Enum)`` pattern on 3.10. Keeping the floor at
3.10 matters here because the person cloning this repo is as likely to be on a
distro Python as on the latest release, and "upgrade your interpreter" is a bad
first impression for a portfolio project.
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):  # pragma: no cover - depends on interpreter
    from enum import StrEnum
else:  # pragma: no cover - depends on interpreter
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        """Backport of :class:`enum.StrEnum` for Python 3.10."""

        def __str__(self) -> str:
            return str(self.value)

        def _generate_next_value_(  # type: ignore[override]
            name: str,  # noqa: N805 - enum protocol signature
            start: int,
            count: int,
            last_values: list[str],
        ) -> str:
            return name.lower()


__all__ = ["StrEnum"]
