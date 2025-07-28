# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Base Data Collector for Qortfolio V2
Abstract base class for all data collection implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import time
import requests
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass

from core.config import get_config
from core.logging import get_logger


@dataclass
class CollectionResult:
    """Result of a data collection operation."""
    success: bool
    data: Optional[pd.DataFrame]
    error: Optional[str]
    records_count: int
    response_time: float
    timestamp: datetime
    source: str


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, requests_per_second: int = 10):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Maximum requests per second
        """
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()


class BaseDataCollector(ABC):
    """
    Abstract base class for all data collectors.
    
    Provides common functionality for:
    - Rate limiting
    - Error handling
    - Logging
    - Data validation
    - Configuration management
    """
    
    def __init__(self, name: str):
        """
        Initialize base collector.
        
        Args:
            name: Name of the collector (e.g., "deribit", "yfinance")
        """
        self.name = name
        self.config = get_config()
        self.logger = get_logger(f"data.{name}")
        
        # Initialize rate limiter from config
        rate_limit = self._get_rate_limit_config()
        self.rate_limiter = RateLimiter(rate_limit)
        
        # HTTP session for connection pooling
        self.session = requests.Session()
        self._configure_session()
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_records": 0,
            "average_response_time": 0.0,
            "last_update": None
        }
        
        self.logger.info(f"Initialized {name} data collector", extra={
            "collector": name,
            "rate_limit": rate_limit
        })
    
    def _get_rate_limit_config(self) -> int:
        """Get rate limit configuration for this collector."""
        # Default to 5 requests per second
        return self.config.get(f"{self.name}_api.rate_limits.requests_per_second", 5)
    
    def _configure_session(self):
        """Configure HTTP session with timeouts and headers."""
        # Get HTTP configuration
        timeout = self.config.get("http_client.timeout", 30)
        user_agent = self.config.get("http_client.headers.User-Agent", "Qortfolio-V2/0.1.0")
        
        # Set session defaults
        self.session.timeout = timeout
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate"
        })
    
    @abstractmethod
    def collect_data(self, symbol: str, **kwargs) -> CollectionResult:
        """
        Collect data for a specific symbol.
        
        Args:
            symbol: Asset symbol (e.g., "BTC", "ETH")
            **kwargs: Additional parameters specific to the collector
            
        Returns:
            Collection result with data or error information
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate collected data.
        
        Args:
            data: Collected data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        pass
    
    def _make_request(self, url: str, params: Optional[Dict] = None, 
                     method: str = "GET") -> requests.Response:
        """
        Make HTTP request with rate limiting and error handling.
        
        Args:
            url: Request URL
            params: Request parameters
            method: HTTP method
            
        Returns:
            Response object
            
        Raises:
            requests.RequestException: For request failures
        """
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        start_time = time.time()
        
        try:
            # Make request
            if method.upper() == "GET":
                response = self.session.get(url, params=params)
            elif method.upper() == "POST":
                response = self.session.post(url, json=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response_time = time.time() - start_time
            
            # Update statistics
            self.stats["total_requests"] += 1
            
            if response.status_code == 200:
                self.stats["successful_requests"] += 1
            else:
                self.stats["failed_requests"] += 1
            
            # Update average response time
            total_requests = self.stats["total_requests"]
            current_avg = self.stats["average_response_time"]
            self.stats["average_response_time"] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
            
            # Log API call
            from core.logging import log_api_call
            log_api_call(
                api_name=self.name,
                endpoint=url,
                method=method,
                status_code=response.status_code,
                response_time=response_time
            )
            
            # Raise for bad status codes
            response.raise_for_status()
            
            return response
            
        except requests.RequestException as e:
            response_time = time.time() - start_time
            self.stats["failed_requests"] += 1
            
            self.logger.error(f"Request failed: {url}", extra={
                "url": url,
                "params": params,
                "method": method,
                "error": str(e),
                "response_time": response_time
            })
            raise
    
    def _validate_symbol(self, symbol: str) -> bool:
        """
        Validate that symbol is supported.
        
        Args:
            symbol: Asset symbol to validate
            
        Returns:
            True if symbol is supported
        """
        # Check against configuration
        supported_symbols = [crypto.symbol for crypto in self.config.enabled_cryptocurrencies]
        return symbol.upper() in [s.upper() for s in supported_symbols]
    
    def _create_error_result(self, error: str, symbol: str) -> CollectionResult:
        """
        Create error result.
        
        Args:
            error: Error message
            symbol: Symbol that failed
            
        Returns:
            Error collection result
        """
        return CollectionResult(
            success=False,
            data=None,
            error=error,
            records_count=0,
            response_time=0.0,
            timestamp=datetime.now(),
            source=self.name
        )
    
    def _create_success_result(self, data: pd.DataFrame, response_time: float) -> CollectionResult:
        """
        Create success result.
        
        Args:
            data: Collected data
            response_time: Request response time
            
        Returns:
            Success collection result
        """
        records_count = len(data) if data is not None else 0
        self.stats["total_records"] += records_count
        self.stats["last_update"] = datetime.now()
        
        return CollectionResult(
            success=True,
            data=data,
            error=None,
            records_count=records_count,
            response_time=response_time,
            timestamp=datetime.now(),
            source=self.name
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collector statistics."""
        return {
            "collector": self.name,
            **self.stats,
            "success_rate": (
                self.stats["successful_requests"] / max(self.stats["total_requests"], 1)
            ),
            "average_records_per_request": (
                self.stats["total_records"] / max(self.stats["successful_requests"], 1)
            )
        }
    
    def reset_statistics(self):
        """Reset collector statistics."""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_records": 0,
            "average_response_time": 0.0,
            "last_update": None
        }
        self.logger.info(f"Statistics reset for {self.name} collector")
    
    def __del__(self):
        """Cleanup when collector is destroyed."""
        if hasattr(self, 'session'):
            self.session.close()


class DataCollectionError(Exception):
    """Custom exception for data collection errors."""
    
    def __init__(self, message: str, collector: str, symbol: str = None):
        """
        Initialize data collection error.
        
        Args:
            message: Error message
            collector: Name of the collector that failed
            symbol: Symbol that failed (optional)
        """
        self.collector = collector
        self.symbol = symbol
        super().__init__(message)
    
    def __str__(self):
        base_msg = f"[{self.collector}] {super().__str__()}"
        if self.symbol:
            base_msg += f" (symbol: {self.symbol})"
        return base_msg


def validate_dataframe_structure(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate DataFrame has required structure.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if DataFrame is valid
    """
    if df is None or df.empty:
        return False
    
    # Check required columns exist
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        return False
    
    # Check for any data
    return len(df) > 0


def clean_numeric_data(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """
    Clean numeric data in DataFrame.
    
    Args:
        df: DataFrame to clean
        numeric_columns: List of columns that should be numeric
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    for col in numeric_columns:
        if col in df_clean.columns:
            # Convert to numeric, replacing errors with NaN
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Remove infinite values
            df_clean[col] = df_clean[col].replace([float('inf'), float('-inf')], None)
    
    # Remove rows with all NaN values in numeric columns
    df_clean = df_clean.dropna(subset=numeric_columns, how='all')
    
    return df_clean