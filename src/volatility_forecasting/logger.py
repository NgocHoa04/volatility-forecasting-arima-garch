"""
Logging configuration for volatility forecasting project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from .config import LOG_LEVEL, LOG_FORMAT, LOG_FILE


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Set up logger with console and file handlers.
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__)
    level : str, optional
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        If None, uses LOG_LEVEL from config
    log_file : Path, optional
        Path to log file. If None, uses LOG_FILE from config
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if level is None:
        level = LOG_LEVEL
    logger.setLevel(getattr(logging, level))
    
    if log_file is None:
        log_file = LOG_FILE
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, level))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# Global logger for package
logger = setup_logger(__name__)
