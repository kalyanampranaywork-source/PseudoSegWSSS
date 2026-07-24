"""
utils/logger.py

Centralized logging utility for the curriculum learning framework.

Features
--------
• Console logging
• File logging
• Automatic log directory creation
• Timestamped log files
• Reusable across all stages
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def get_logger(
    name: str,
    log_directory: str | Path,
    log_level: str = "INFO",
):
    """
    Create or retrieve a configured logger.

    Parameters
    ----------
    name : str
        Logger name.

    log_directory : str | Path
        Directory where log files will be stored.

    log_level : str, default="INFO"
        Logging level.

    Returns
    -------
    logging.Logger
    """

    log_directory = Path(log_directory)
    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper()))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ------------------------------------------------------------
    # Log file
    # ------------------------------------------------------------

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = (
        log_directory
        / f"{name}_{timestamp}.log"
    )

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)

    # ------------------------------------------------------------
    # Console
    # ------------------------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    # ------------------------------------------------------------
    # Register handlers
    # ------------------------------------------------------------

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.propagate = False

    logger.info("=" * 80)
    logger.info(f"Logger initialized : {name}")
    logger.info(f"Log file           : {log_file}")
    logger.info("=" * 80)

    return logger