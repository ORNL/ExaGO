"""Logging configuration for AgentiGrid."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def setup_logging(logs_dir: Path, verbose: bool = False) -> None:
    """Configure root ``agentigrid`` logger with console and file handlers.

    Args:
        logs_dir: Directory where log files are written.
        verbose: If *True*, the console handler uses DEBUG level.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger("agentigrid")
    root_logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on repeated calls
    if root_logger.handlers:
        return

    # --- Console handler ---
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    console.setFormatter(console_fmt)
    root_logger.addHandler(console)

    # --- File handler ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"agentigrid_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    root_logger.addHandler(file_handler)

    root_logger.debug("Logging initialised — file: %s", log_file)
