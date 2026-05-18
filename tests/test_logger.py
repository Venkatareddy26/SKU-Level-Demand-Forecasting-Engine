"""
Unit tests for the structured logging framework.

Tests verify that the logging framework correctly formats logs in JSON and text
formats, includes required fields, and properly configures console and file handlers.
"""

import logging
import json
import tempfile
from pathlib import Path
import pytest

from utils.logger import (
    JSONFormatter,
    TextFormatter,
    setup_logging,
    get_logger
)


def _close_all_file_handlers():
    """Close all file handlers on the root logger to release file locks (Windows fix)."""
    logger = logging.getLogger()
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
    logger.handlers.clear()


class TestJSONFormatter:
    """Tests for JSONFormatter class."""
    
    def test_format_basic_message(self):
        """Test that JSONFormatter formats a basic log message correctly."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=10,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        result = formatter.format(record)
        log_data = json.loads(result)
        
        # Verify required fields are present
        assert 'timestamp' in log_data
        assert log_data['level'] == 'INFO'
        assert log_data['module'] == 'test'
        assert log_data['function'] == record.funcName
        assert log_data['message'] == 'Test message'
    
    def test_format_includes_timestamp(self):
        """Test that JSONFormatter includes ISO format timestamp."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=10,
            msg='Test',
            args=(),
            exc_info=None
        )
        
        result = formatter.format(record)
        log_data = json.loads(result)
        
        # Verify timestamp is in ISO format
        assert 'timestamp' in log_data
        # Should be parseable as ISO format (will raise if not)
        from datetime import datetime
        datetime.fromisoformat(log_data['timestamp'])
    
    def test_format_with_exception(self):
        """Test that JSONFormatter includes exception information."""
        formatter = JSONFormatter()
        
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
            
            record = logging.LogRecord(
                name='test',
                level=logging.ERROR,
                pathname='test.py',
                lineno=10,
                msg='Error occurred',
                args=(),
                exc_info=exc_info
            )
            
            result = formatter.format(record)
            log_data = json.loads(result)
            
            assert 'exception' in log_data
            assert 'ValueError' in log_data['exception']
            assert 'Test error' in log_data['exception']
    
    def test_format_with_custom_metrics(self):
        """Test that JSONFormatter includes custom metrics field."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=10,
            msg='Training complete',
            args=(),
            exc_info=None
        )
        
        # Add custom metrics
        record.metrics = {'rmse': 45.2, 'wrmsse': 0.58}
        
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert 'metrics' in log_data
        assert log_data['metrics']['rmse'] == 45.2
        assert log_data['metrics']['wrmsse'] == 0.58
    
    def test_format_with_model_version(self):
        """Test that JSONFormatter includes model_version field."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=10,
            msg='Model saved',
            args=(),
            exc_info=None
        )
        
        # Add model version
        record.model_version = 'v20240115_143022'
        
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert 'model_version' in log_data
        assert log_data['model_version'] == 'v20240115_143022'


class TestTextFormatter:
    """Tests for TextFormatter class."""
    
    def test_format_basic_message(self):
        """Test that TextFormatter formats a basic log message correctly."""
        formatter = TextFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=10,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        result = formatter.format(record)
        
        # Verify format includes key components
        assert 'INFO' in result
        assert 'test' in result
        assert 'Test message' in result


class TestSetupLogging:
    """Tests for setup_logging function."""
    
    def teardown_method(self):
        """Clean up logging handlers after each test."""
        _close_all_file_handlers()
    
    def test_setup_logging_default(self):
        """Test setup_logging with default parameters."""
        setup_logging()
        
        logger = logging.getLogger()
        
        # Verify log level is INFO
        assert logger.level == logging.INFO
        
        # Verify console handler is added
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)
    
    def test_setup_logging_with_level(self):
        """Test setup_logging with custom log level."""
        setup_logging(level='DEBUG')
        
        logger = logging.getLogger()
        assert logger.level == logging.DEBUG
    
    def test_setup_logging_json_format(self):
        """Test setup_logging with JSON format."""
        setup_logging(log_format='json')
        
        logger = logging.getLogger()
        
        # Verify JSONFormatter is used
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0].formatter, JSONFormatter)
    
    def test_setup_logging_text_format(self):
        """Test setup_logging with text format."""
        setup_logging(log_format='text')
        
        logger = logging.getLogger()
        
        # Verify TextFormatter is used
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0].formatter, TextFormatter)
    
    def test_setup_logging_with_file(self):
        """Test setup_logging with file handler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'test.log'
            
            setup_logging(log_file=str(log_file))
            
            logger = logging.getLogger()
            
            # Verify both console and file handlers are added
            assert len(logger.handlers) == 2
            
            # Verify file handler is present
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) == 1
            
            # Verify file handler uses JSONFormatter
            assert isinstance(file_handlers[0].formatter, JSONFormatter)
            
            # Close handlers before tmpdir cleanup
            _close_all_file_handlers()
    
    def test_setup_logging_no_console(self):
        """Test setup_logging with console disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'test.log'
            
            setup_logging(log_file=str(log_file), console=False)
            
            logger = logging.getLogger()
            
            # Verify only file handler is added
            assert len(logger.handlers) == 1
            assert isinstance(logger.handlers[0], logging.FileHandler)
            
            # Close handlers before tmpdir cleanup
            _close_all_file_handlers()
    
    def test_setup_logging_creates_log_directory(self):
        """Test that setup_logging creates log directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'logs' / 'subdir' / 'test.log'
            
            setup_logging(log_file=str(log_file))
            
            # Verify directory was created
            assert log_file.parent.exists()
            
            # Close handlers before tmpdir cleanup
            _close_all_file_handlers()
    
    def test_logging_output_to_file(self):
        """Test that logs are actually written to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'test.log'
            
            setup_logging(log_file=str(log_file), console=False)
            
            logger = logging.getLogger(__name__)
            logger.info("Test log message")
            
            # Force flush
            for handler in logger.handlers:
                handler.flush()
            
            # Verify log file was created and contains the message
            assert log_file.exists()
            content = log_file.read_text()
            assert 'Test log message' in content
            
            # Verify it's JSON format
            log_data = json.loads(content)
            assert log_data['message'] == 'Test log message'
            assert log_data['level'] == 'INFO'
            
            # Close handlers before tmpdir cleanup
            _close_all_file_handlers()


class TestGetLogger:
    """Tests for get_logger function."""
    
    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a Logger instance."""
        logger = get_logger(__name__)
        assert isinstance(logger, logging.Logger)
    
    def test_get_logger_with_name(self):
        """Test that get_logger returns logger with correct name."""
        logger = get_logger('test.module')
        assert logger.name == 'test.module'


class TestLoggingRequirements:
    """Tests verifying specific requirements are met."""
    
    def teardown_method(self):
        """Clean up logging handlers after each test."""
        _close_all_file_handlers()
    
    def test_requirement_6_2_includes_required_fields(self):
        """Test Requirement 6.2: Logs include timestamp, level, module, message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'test.log'
            setup_logging(log_file=str(log_file), log_format='json', console=False)
            
            logger = logging.getLogger(__name__)
            logger.info("Test message")
            
            # Force flush
            for handler in logger.handlers:
                handler.flush()
            
            # Read and parse log
            content = log_file.read_text()
            log_data = json.loads(content)
            
            # Verify required fields
            assert 'timestamp' in log_data
            assert 'level' in log_data
            assert 'module' in log_data
            assert 'message' in log_data
            
            # Close handlers before tmpdir cleanup
            _close_all_file_handlers()
    
    def test_requirement_6_3_supports_log_levels(self):
        """Test Requirement 6.3: System supports configurable log levels."""
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in levels:
            setup_logging(level=level)
            logger = logging.getLogger()
            assert logger.level == getattr(logging, level)
    
    def test_requirement_6_6_supports_console_and_file(self):
        """Test Requirement 6.6: System supports logging to console and file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'test.log'
            
            setup_logging(log_file=str(log_file), console=True)
            
            logger = logging.getLogger()
            
            # Verify both handlers present
            assert len(logger.handlers) == 2
            
            # Verify handler types
            handler_types = [type(h) for h in logger.handlers]
            assert logging.StreamHandler in handler_types
            assert logging.FileHandler in handler_types
            
            # Close handlers before tmpdir cleanup
            _close_all_file_handlers()
    
    def test_requirement_6_7_json_format_for_production(self):
        """Test Requirement 6.7: JSON format for production logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'test.log'
            
            setup_logging(log_file=str(log_file), log_format='json', console=False)
            
            logger = logging.getLogger(__name__)
            logger.info("Production log")
            
            # Force flush
            for handler in logger.handlers:
                handler.flush()
            
            # Verify JSON format
            content = log_file.read_text()
            log_data = json.loads(content)  # Should not raise
            assert log_data['message'] == 'Production log'
            
            # Close handlers before tmpdir cleanup
            _close_all_file_handlers()
