"""
Logging Utilities

Setup and configuration for logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(name: str = 'amr_gnn',
                level: int = logging.INFO,
                log_file: Optional[str] = None,
                console: bool = True) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file (optional)
        console: Whether to add console handler

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Add file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
