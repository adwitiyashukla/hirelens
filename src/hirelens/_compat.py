from __future__ import annotations

import sys

if sys.version_info >= (3, 11):  # pragma: no cover - depends on interpreter
    from enum import StrEnum
else:  # pragma: no cover - depends on interpreter
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        def __str__(self) -> str:
            return str(self.value)

        def _generate_next_value_(  # type: ignore[override]
            name: str,
            start: int,
            count: int,
            last_values: list[str],
        ) -> str:
            return name.lower()


__all__ = ["StrEnum"]
