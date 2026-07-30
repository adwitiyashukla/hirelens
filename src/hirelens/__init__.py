"""HireLens: evidence-grounded candidate screening.

See docs/DESIGN.md for the architecture and the reasoning behind it.
"""

from hirelens.config import Provider, Settings, get_settings

__version__ = "0.1.0"
__all__ = ["Provider", "Settings", "__version__", "get_settings"]
