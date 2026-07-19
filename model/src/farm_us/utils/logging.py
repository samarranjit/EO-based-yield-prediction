"""Structured logging setup (rich if available, plain otherwise)."""

from __future__ import annotations

import logging

_CONFIGURED = False


def get_logger(name: str = "farm_us", level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        try:
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
            fmt = "%(message)s"
        except Exception:
            handler = logging.StreamHandler()
            fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        logging.basicConfig(level=level, format=fmt, handlers=[handler], datefmt="[%X]")
        _CONFIGURED = True
    return logging.getLogger(name)


class FarmError(Exception):
    """Base class for FARM-US domain errors."""


class DataContractError(FarmError):
    """Raised when a raster / manifest violates the data contract."""


class LeakageError(FarmError):
    """Raised when a split / normalization step would leak test information."""
