"""Logging utilities for structured logging.

Provides a simple configuration function to set log level and format.
"""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with a structured, concise format.

    Parameters
    ----------
    level: str
        Logging level name (e.g., "DEBUG", "INFO").
    """

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

