"""Configure loguru once at process startup."""

from __future__ import annotations

import sys

from loguru import logger


def _ensure_extra(record: dict) -> bool:
    record["extra"].setdefault("scope", "app")
    return True


def configure_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[scope]}</cyan> | "
            "{message}"
        ),
        level=level,
        filter=_ensure_extra,
    )
