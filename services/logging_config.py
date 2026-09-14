from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys


def configure_logging(service_name: str) -> None:
    """Log to stderr and, when LOG_DIR is set, to a bounded rolling file."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s pid=%(process)d %(message)s"
    )
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    log_dir_value = os.getenv("LOG_DIR", "").strip()
    if log_dir_value:
        log_dir = Path(log_dir_value)
        log_dir.mkdir(parents=True, exist_ok=True)
        max_bytes = _positive_int("LOG_MAX_BYTES", 10 * 1024 * 1024)
        backup_count = _positive_int("LOG_BACKUP_COUNT", 5)
        handlers.append(
            RotatingFileHandler(
                log_dir / f"{service_name}.log",
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        )

    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=level, handlers=handlers, force=True)


def _positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default
