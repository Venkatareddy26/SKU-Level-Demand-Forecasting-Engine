"""
Structured logging framework for the forecasting engine.

This module provides JSON-formatted logging capabilities with support for
console and file outputs, configurable log levels, and custom fields for
metrics and model versions.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging.
    
    This formatter converts log records into JSON format suitable for
    structured log aggregation systems. It includes standard fields
    (timestamp, level, module, function, message) and supports custom
    fields like metrics and model_version.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON.
        
        Args:
            record: The log record to format
            
        Returns:
            JSON-formatted log string
        """
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'message': record.getMessage(),
        }
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add custom fields if present
        if hasattr(record, 'metrics'):
            log_data['metrics'] = record.metrics
        if hasattr(record, 'model_version'):
            log_data['model_version'] = record.model_version
            
        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Format logs as human-readable text.
    
    This formatter provides a traditional text-based log format suitable
    for console output and development environments.
    """
    
    def __init__(self):
        """Initialize the text formatter with a standard format."""
        super().__init__(
            fmt='%(asctime)s - %(levelname)s - %(module)s:%(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


def setup_logging(
    level: str = 'INFO',
    log_format: str = 'text',
    log_file: Optional[str] = None,
    console: bool = True
) -> None:
    """Configure logging based on provided settings.
    
    This function sets up the root logger with console and/or file handlers
    based on the configuration. It supports both JSON and text log formats.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format for logs ('json' or 'text')
        log_file: Path to log file (optional, enables file logging if provided)
        console: Whether to enable console logging (default: True)
        
    Example:
        >>> setup_logging(level='INFO', log_format='json', log_file='logs/app.log')
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Application started")
    """
    # Get root logger
    logger = logging.getLogger()
    
    # Set log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Choose formatter based on format setting
    if log_format.lower() == 'json':
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        # Always use JSON format for file logs for structured aggregation
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)


def setup_logging_from_config(config: Any) -> None:
    """Configure logging from a configuration object.
    
    This is a convenience function that extracts logging settings from
    a configuration object (like ConfigManager) and calls setup_logging.
    
    Args:
        config: Configuration object with get() method
        
    Example:
        >>> from config.config_manager import ConfigManager
        >>> config = ConfigManager('config/config.yaml')
        >>> setup_logging_from_config(config)
    """
    level = config.get('logging.level', 'INFO')
    log_format = config.get('logging.format', 'text')
    log_file = config.get('logging.file', None)
    console = config.get('logging.console', True)
    
    setup_logging(
        level=level,
        log_format=log_format,
        log_file=log_file,
        console=console
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.
    
    This is a convenience function that wraps logging.getLogger()
    for consistency across the codebase.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return logging.getLogger(name)
