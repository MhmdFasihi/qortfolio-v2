# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Logging Framework for Qortfolio V2
Provides structured, configurable logging across the application
"""

import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import traceback


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add process/thread info
        log_entry["process_id"] = os.getpid()
        log_entry["thread_id"] = record.thread
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                'thread', 'threadName', 'processName', 'process', 'getMessage'
            }:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        return json.dumps(log_entry, default=str)


class ContextualFormatter(logging.Formatter):
    """Human-readable formatter with contextual information."""
    
    def __init__(self, include_context: bool = True):
        self.include_context = include_context
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with context."""
        # Base format
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level_colored = self._colorize_level(record.levelname)
        
        # Basic message
        base_msg = f"{timestamp} | {level_colored:8} | {record.name:20} | {record.getMessage()}"
        
        if self.include_context:
            context = f" [{record.module}.{record.funcName}:{record.lineno}]"
            base_msg += context
        
        # Add exception info if present
        if record.exc_info:
            base_msg += "\n" + "".join(traceback.format_exception(*record.exc_info))
        
        return base_msg
    
    def _colorize_level(self, level: str) -> str:
        """Add color to log levels (for terminal output)."""
        colors = {
            "DEBUG": "\033[36m",    # Cyan
            "INFO": "\033[32m",     # Green  
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",    # Red
            "CRITICAL": "\033[35m", # Magenta
        }
        reset = "\033[0m"
        
        if sys.stdout.isatty() and level in colors:
            return f"{colors[level]}{level}{reset}"
        return level


class QortfolioLogger:
    """Main logging manager for Qortfolio V2."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize logging system.
        
        Args:
            config: Logging configuration dictionary
        """
        self.config = config or {}
        self._loggers: Dict[str, logging.Logger] = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up the logging system."""
        # Get configuration
        log_level = self._get_log_level()
        log_format = self.config.get("format", "contextual")  # json or contextual
        log_file = self.config.get("file", "logs/qortfolio.log")
        console_enabled = self.config.get("console", True)
        file_enabled = self.config.get("file_enabled", True)
        
        # Create logs directory
        if file_enabled:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger("qortfolio")
        root_logger.setLevel(log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatters
        if log_format.lower() == "json":
            formatter = JSONFormatter()
        else:
            formatter = ContextualFormatter(include_context=True)
        
        # Add console handler
        if console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # Add file handler
        if file_enabled:
            # Use rotating file handler to prevent large log files
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Prevent propagation to parent loggers
        root_logger.propagate = False
        
        # Log startup message
        root_logger.info("Qortfolio V2 logging system initialized", extra={
            "log_level": logging.getLevelName(log_level),
            "log_format": log_format,
            "console_enabled": console_enabled,
            "file_enabled": file_enabled,
            "log_file": log_file if file_enabled else None
        })
    
    def _get_log_level(self) -> int:
        """Get log level from configuration or environment."""
        # Check environment variable first
        env_level = os.getenv("QORTFOLIO_LOG_LEVEL", "").upper()
        if env_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            return getattr(logging, env_level)
        
        # Check configuration
        config_level = self.config.get("level", "INFO").upper()
        return getattr(logging, config_level, logging.INFO)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger for a specific module/component.
        
        Args:
            name: Logger name (usually module name)
            
        Returns:
            Configured logger instance
        """
        full_name = f"qortfolio.{name}"
        
        if full_name not in self._loggers:
            logger = logging.getLogger(full_name)
            logger.setLevel(self._get_log_level())
            self._loggers[full_name] = logger
        
        return self._loggers[full_name]
    
    def log_function_call(self, func_name: str, args: tuple = (), kwargs: dict = None):
        """
        Log function call details (for debugging).
        
        Args:
            func_name: Name of the function being called
            args: Function arguments
            kwargs: Function keyword arguments
        """
        logger = self.get_logger("function_calls")
        
        kwargs = kwargs or {}
        
        # Sanitize arguments (remove sensitive data)
        safe_args = self._sanitize_args(args)
        safe_kwargs = self._sanitize_args(kwargs)
        
        logger.debug(f"Function call: {func_name}", extra={
            "function": func_name,
            "args": safe_args,
            "kwargs": safe_kwargs
        })
    
    def log_api_call(self, api_name: str, endpoint: str, method: str = "GET", 
                     status_code: Optional[int] = None, response_time: Optional[float] = None):
        """
        Log API call details.
        
        Args:
            api_name: Name of the API (e.g., "deribit", "yfinance")
            endpoint: API endpoint
            method: HTTP method
            status_code: Response status code
            response_time: Response time in seconds
        """
        logger = self.get_logger("api_calls")
        
        log_level = logging.INFO if status_code and status_code < 400 else logging.WARNING
        
        logger.log(log_level, f"API call: {api_name} {method} {endpoint}", extra={
            "api_name": api_name,
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time_ms": round(response_time * 1000, 2) if response_time else None
        })
    
    def log_data_collection(self, data_type: str, symbol: str, records_count: int, 
                           success: bool = True, error: Optional[str] = None):
        """
        Log data collection events.
        
        Args:
            data_type: Type of data collected (e.g., "spot_prices", "options_data")
            symbol: Asset symbol
            records_count: Number of records collected
            success: Whether collection was successful
            error: Error message if unsuccessful
        """
        logger = self.get_logger("data_collection")
        
        if success:
            logger.info(f"Data collection successful: {data_type} for {symbol}", extra={
                "data_type": data_type,
                "symbol": symbol,
                "records_count": records_count,
                "success": success
            })
        else:
            logger.error(f"Data collection failed: {data_type} for {symbol}", extra={
                "data_type": data_type,
                "symbol": symbol,
                "success": success,
                "error": error
            })
    
    def log_calculation(self, calculation_type: str, symbol: str, inputs: dict, 
                       result: Any, execution_time: Optional[float] = None):
        """
        Log financial calculations.
        
        Args:
            calculation_type: Type of calculation (e.g., "black_scholes", "greeks")
            symbol: Asset symbol
            inputs: Calculation inputs
            result: Calculation result
            execution_time: Execution time in seconds
        """
        logger = self.get_logger("calculations")
        
        # Sanitize inputs and results
        safe_inputs = self._sanitize_args(inputs)
        safe_result = self._sanitize_args(result)
        
        logger.debug(f"Calculation: {calculation_type} for {symbol}", extra={
            "calculation_type": calculation_type,
            "symbol": symbol,
            "inputs": safe_inputs,
            "result": safe_result,
            "execution_time_ms": round(execution_time * 1000, 2) if execution_time else None
        })
    
    def _sanitize_args(self, obj: Any) -> Any:
        """
        Sanitize arguments to remove sensitive data and large objects.
        
        Args:
            obj: Object to sanitize
            
        Returns:
            Sanitized object
        """
        if isinstance(obj, dict):
            return {k: self._sanitize_value(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_value(item) for item in obj]
        else:
            return self._sanitize_value(obj)
    
    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize a single value."""
        # Skip large objects
        if hasattr(value, '__len__') and len(value) > 100:
            return f"<{type(value).__name__} with {len(value)} items>"
        
        # Convert pandas/numpy objects to basic types
        if hasattr(value, 'to_dict'):
            return "<pandas_object>"
        if hasattr(value, 'tolist'):
            return "<numpy_array>"
        
        # Ensure JSON serializable
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)


# Global logging instance
_logging_instance: Optional[QortfolioLogger] = None


def setup_logging(config: Optional[Dict[str, Any]] = None):
    """
    Set up the global logging system.
    
    Args:
        config: Logging configuration
    """
    global _logging_instance
    
    # Get configuration from environment if not provided
    if config is None:
        config = {
            "level": os.getenv("QORTFOLIO_LOG_LEVEL", "INFO"),
            "format": os.getenv("QORTFOLIO_LOG_FORMAT", "contextual"),
            "file": os.getenv("QORTFOLIO_LOG_FILE", "logs/qortfolio.log"),
            "console": True,
            "file_enabled": True
        }
    
    _logging_instance = QortfolioLogger(config)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.
    
    Args:
        name: Module name
        
    Returns:
        Configured logger
    """
    global _logging_instance
    if _logging_instance is None:
        setup_logging()
    return _logging_instance.get_logger(name)


# Convenience functions
def log_api_call(api_name: str, endpoint: str, **kwargs):
    """Log an API call."""
    global _logging_instance
    if _logging_instance is None:
        setup_logging()
    _logging_instance.log_api_call(api_name, endpoint, **kwargs)


def log_data_collection(data_type: str, symbol: str, records_count: int, **kwargs):
    """Log data collection."""
    global _logging_instance
    if _logging_instance is None:
        setup_logging()
    _logging_instance.log_data_collection(data_type, symbol, records_count, **kwargs)


if __name__ == "__main__":
    # Test the logging system
    setup_logging()
    
    logger = get_logger("test")
    
    print("🔧 Testing Logging System")
    print("=" * 30)
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("System initialized successfully")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test structured logging
    logger.info("Processing options data", extra={
        "symbol": "BTC",
        "options_count": 150,
        "processing_time": 2.3
    })
    
    # Test API call logging
    log_api_call("deribit", "/public/ticker", method="GET", status_code=200, response_time=0.5)
    
    # Test data collection logging
    log_data_collection("options_data", "BTC", 150, success=True)
    
    print("✅ Logging system test completed!")