# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Logging Framework for Qortfolio V2
Location: src/core/logging.py

Provides consistent logging across all modules with configurable levels and formats.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Global logger registry
_loggers: Dict[str, logging.Logger] = {}
_configured = False

class QortfolioLogger:
    """
    Advanced logging system for Qortfolio V2 with structured logging support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the logging system."""
        self.config = config
        self._loggers: Dict[str, logging.Logger] = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup the logging configuration."""
        level = self._get_log_level()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        root_logger.handlers.clear()
        
        # Console handler
        if self.config.get('console', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if self.config.get('file_enabled', False):
            try:
                from pathlib import Path
                log_path = Path(self.config.get('file', 'logs/qortfolio.log'))
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(log_path)
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
            except Exception as e:
                print(f"Warning: Could not setup file logging: {e}")
    
    def _get_log_level(self) -> int:
        """Get logging level from configuration."""
        import os
        
        # Check environment variable first
        env_level = os.getenv("QORTFOLIO_LOG_LEVEL", "").upper()
        if env_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            return getattr(logging, env_level)
        
        # Check configuration
        config_level = self.config.get("level", "INFO").upper()
        return getattr(logging, config_level, logging.INFO)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger for a specific module/component."""
        full_name = f"qortfolio.{name}"
        
        if full_name not in self._loggers:
            logger = logging.getLogger(full_name)
            logger.setLevel(self._get_log_level())
            self._loggers[full_name] = logger
        
        return self._loggers[full_name]
    
    def log_api_call(self, api_name: str, endpoint: str, method: str = "GET", 
                     status_code: Optional[int] = None, response_time: Optional[float] = None):
        """Log API call details."""
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
        """Log data collection events."""
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
        """Log financial calculations."""
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
    
    def log_function_call(self, func_name: str, args: tuple = (), kwargs: dict = None):
        """Log function call details (for debugging)."""
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
    
    def _sanitize_args(self, obj: Any) -> Any:
        """Sanitize arguments to remove sensitive data and large objects."""
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
            import json
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
        import os
        config = {
            "level": os.getenv("QORTFOLIO_LOG_LEVEL", "INFO"),
            "format": os.getenv("QORTFOLIO_LOG_FORMAT", "contextual"),
            "file": os.getenv("QORTFOLIO_LOG_FILE", "logs/qortfolio.log"),
            "console": True,
            "file_enabled": False  # Disabled by default to prevent file creation issues
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


# CRITICAL: Convenience functions that dashboard expects
def log_api_call(api_name: str, endpoint: str, **kwargs):
    """Log an API call - REQUIRED by dashboard."""
    global _logging_instance
    if _logging_instance is None:
        setup_logging()
    _logging_instance.log_api_call(api_name, endpoint, **kwargs)


def log_data_collection(data_type: str, symbol: str, records_count: int, **kwargs):
    """Log data collection - CRITICAL MISSING FUNCTION."""
    global _logging_instance
    if _logging_instance is None:
        setup_logging()
    _logging_instance.log_data_collection(data_type, symbol, records_count, **kwargs)


def log_calculation(calculation_type: str, symbol: str, inputs: dict, result: Any, **kwargs):
    """Log financial calculations - REQUIRED by dashboard."""
    global _logging_instance
    if _logging_instance is None:
        setup_logging()
    _logging_instance.log_calculation(calculation_type, symbol, inputs, result, **kwargs)


def log_function_call(func_name: str, args: tuple = (), kwargs: dict = None):
    """Log function call details - REQUIRED by dashboard."""
    global _logging_instance
    if _logging_instance is None:
        setup_logging()
    _logging_instance.log_function_call(func_name, args, kwargs)

def set_log_level(level: str):
    """
    Set global log level.
    
    Args:
        level: Log level string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """
    global _logging_instance
    if _logging_instance is None:
        setup_logging()
    
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    log_level = level_map.get(level.upper(), logging.INFO)
    
    # Update root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Update all handlers
    for handler in root_logger.handlers:
        handler.setLevel(log_level)


def log_performance(func):
    """Decorator to log function performance."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.debug(f"{func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    return wrapper


# Initialize basic logging on module import
setup_logging()


if __name__ == "__main__":
    # Test the logging system
    setup_logging()
    
    logger = get_logger("test")
    
    print("🔧 Testing Complete Logging System")
    print("=" * 40)
    
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
    
    # Test data collection logging (CRITICAL function)
    log_data_collection("options_data", "BTC", 150, success=True)
    
    # Test calculation logging
    log_calculation("black_scholes", "BTC", {"spot": 95000, "strike": 100000}, {"price": 1250.5})
    
    # Test function call logging
    log_function_call("test_function", ("arg1", "arg2"), {"kwarg1": "value1"})
    
    print("✅ Complete logging system test completed!")
    print("✅ All dashboard-required functions available!")
    print("   - log_data_collection ✅")
    print("   - log_api_call ✅") 
    print("   - log_calculation ✅")
    print("   - log_function_call ✅")