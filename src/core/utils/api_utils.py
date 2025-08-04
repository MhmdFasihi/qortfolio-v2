# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
API Helper Utilities for Qortfolio V2
Location: src/core/utils/api_utils.py

Professional API utilities for handling HTTP requests, rate limiting,
error handling, and response validation for financial data APIs.
"""

import time
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta
import logging
from functools import wraps
from dataclasses import dataclass
import hashlib

from ..exceptions import (
    APIError,
    APIConnectionError,
    APIRateLimitError,
    APIResponseError,
    DataValidationError
)

logger = logging.getLogger(__name__)

# ==================== CONSTANTS ====================

DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 1.0
DEFAULT_RATE_LIMIT = 10  # requests per second

# HTTP status codes
STATUS_CODES = {
    'OK': 200,
    'BAD_REQUEST': 400,
    'UNAUTHORIZED': 401,
    'FORBIDDEN': 403,
    'NOT_FOUND': 404,
    'TOO_MANY_REQUESTS': 429,
    'INTERNAL_SERVER_ERROR': 500,
    'BAD_GATEWAY': 502,
    'SERVICE_UNAVAILABLE': 503,
    'GATEWAY_TIMEOUT': 504
}

# Retry status codes
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]


# ==================== DATA CLASSES ====================

@dataclass
class APIConfig:
    """Configuration for API client."""
    base_url: str
    timeout: int = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR
    rate_limit: float = DEFAULT_RATE_LIMIT
    headers: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {
                'User-Agent': 'Qortfolio-V2/1.0',
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }


@dataclass
class APIResponse:
    """Standardized API response wrapper."""
    success: bool
    data: Any
    status_code: int
    response_time: float
    error: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    cached: bool = False
    
    @property
    def is_success(self) -> bool:
        """Check if response was successful."""
        return self.success and 200 <= self.status_code < 300
    
    @property
    def is_rate_limited(self) -> bool:
        """Check if response indicates rate limiting."""
        return self.status_code == STATUS_CODES['TOO_MANY_REQUESTS']


# ==================== RATE LIMITING ====================

class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""
    
    def __init__(self, rate: float, burst_size: Optional[int] = None):
        """
        Initialize rate limiter.
        
        Args:
            rate: Requests per second
            burst_size: Maximum burst size (defaults to rate)
        """
        self.rate = rate
        self.burst_size = burst_size or max(1, int(rate))
        self.tokens = self.burst_size
        self.last_update = time.time()
    
    def wait_if_needed(self) -> float:
        """
        Wait if necessary to respect rate limit.
        
        Returns:
            Time waited in seconds
        """
        now = time.time()
        time_passed = now - self.last_update
        
        # Add tokens based on time passed
        self.tokens = min(self.burst_size, self.tokens + time_passed * self.rate)
        self.last_update = now
        
        if self.tokens >= 1:
            self.tokens -= 1
            return 0.0
        else:
            # Need to wait
            wait_time = (1 - self.tokens) / self.rate
            time.sleep(wait_time)
            self.tokens = 0
            return wait_time


# ==================== CACHING ====================

class APICache:
    """Simple in-memory cache for API responses."""
    
    def __init__(self, default_ttl: int = 300):
        """
        Initialize cache.
        
        Args:
            default_ttl: Default time-to-live in seconds
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
    
    def _generate_key(self, url: str, params: Optional[Dict] = None) -> str:
        """Generate cache key from URL and parameters."""
        key_data = f"{url}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, url: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Get cached response if available and not expired."""
        key = self._generate_key(url, params)
        
        if key in self.cache:
            entry = self.cache[key]
            if time.time() < entry['expires_at']:
                logger.debug(f"Cache hit for {url}")
                return entry['data']
            else:
                # Expired, remove
                del self.cache[key]
                logger.debug(f"Cache expired for {url}")
        
        return None
    
    def set(self, url: str, data: Any, params: Optional[Dict] = None, 
            ttl: Optional[int] = None) -> None:
        """Cache response data.""" 
        key = self._generate_key(url, params)
        expires_at = time.time() + (ttl or self.default_ttl)
        
        self.cache[key] = {
            'data': data,
            'expires_at': expires_at,
            'created_at': time.time()
        }
        
        logger.debug(f"Cached response for {url} (TTL: {ttl or self.default_ttl}s)")
    
    def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        logger.info("API cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = time.time()
        valid_entries = sum(1 for entry in self.cache.values() if now < entry['expires_at'])
        
        return {
            'total_entries': len(self.cache),
            'valid_entries': valid_entries,
            'expired_entries': len(self.cache) - valid_entries,
            'cache_size_mb': len(str(self.cache)) / (1024 * 1024)
        }


# ==================== API CLIENT ====================

class APIClient:
    """Enhanced API client with rate limiting, caching, and error handling."""
    
    def __init__(self, config: APIConfig, enable_caching: bool = True):
        """
        Initialize API client.
        
        Args:
            config: API configuration
            enable_caching: Whether to enable response caching
        """
        self.config = config
        self.rate_limiter = RateLimiter(config.rate_limit)
        self.cache = APICache() if enable_caching else None
        
        # Configure session with retries
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=config.max_retries,
            backoff_factor=config.backoff_factor,
            status_forcelist=RETRY_STATUS_CODES,
            allowed_methods=["GET", "POST"],
            raise_on_status=False
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update(config.headers)
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'rate_limit_waits': 0,
            'total_wait_time': 0.0
        }
    
    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint."""
        base = self.config.base_url.rstrip('/')
        endpoint = endpoint.lstrip('/')
        return f"{base}/{endpoint}"
    
    def _handle_response(self, response: requests.Response, 
                        start_time: float) -> APIResponse:
        """Handle and validate API response."""
        response_time = time.time() - start_time
        self.stats['total_requests'] += 1
        
        # Check for rate limiting
        if response.status_code == STATUS_CODES['TOO_MANY_REQUESTS']:
            self.stats['failed_requests'] += 1
            retry_after = response.headers.get('Retry-After')
            
            raise APIRateLimitError(
                api_name="API",
                retry_after=int(retry_after) if retry_after else None
            )
        
        # Handle other errors
        if not response.ok:
            self.stats['failed_requests'] += 1
            
            try:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', str(response.reason))
            except:
                error_msg = str(response.reason)
            
            if response.status_code >= 500:
                raise APIConnectionError(
                    api_name="API",
                    endpoint=response.url,
                    status_code=response.status_code
                )
            else:
                raise APIResponseError(
                    api_name="API",
                    message=error_msg,
                    response_data=response.text
                )
        
        # Parse response
        try:
            data = response.json()
        except json.JSONDecodeError:
            data = response.text
        
        self.stats['successful_requests'] += 1
        
        return APIResponse(
            success=True,
            data=data,
            status_code=response.status_code,
            response_time=response_time,
            headers=dict(response.headers)
        )
    
    def request(self, method: str, endpoint: str, params: Optional[Dict] = None,
               data: Optional[Dict] = None, timeout: Optional[int] = None,
               cache_ttl: Optional[int] = None, skip_cache: bool = False) -> APIResponse:
        """
        Make API request with full error handling and caching.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body data
            timeout: Request timeout
            cache_ttl: Cache time-to-live (GET requests only)
            skip_cache: Skip cache lookup
        
        Returns:
            APIResponse object
        """
        url = self._build_url(endpoint)
        method = method.upper()
        
        # Check cache for GET requests
        if method == 'GET' and self.cache and not skip_cache:
            cached_response = self.cache.get(url, params)
            if cached_response is not None:
                self.stats['cache_hits'] += 1
                return APIResponse(
                    success=True,
                    data=cached_response,
                    status_code=200,
                    response_time=0.0,
                    cached=True
                )
        
        # Rate limiting
        wait_time = self.rate_limiter.wait_if_needed()
        if wait_time > 0:
            self.stats['rate_limit_waits'] += 1
            self.stats['total_wait_time'] += wait_time
        
        # Make request
        start_time = time.time()
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                timeout=timeout or self.config.timeout
            )
            
            api_response = self._handle_response(response, start_time)
            
            # Cache successful GET responses
            if (method == 'GET' and self.cache and api_response.is_success 
                and not skip_cache):
                self.cache.set(url, api_response.data, params, cache_ttl)
            
            return api_response
            
        except requests.RequestException as e:
            self.stats['failed_requests'] += 1
            raise APIConnectionError(
                api_name="API",
                endpoint=url,
                status_code=getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            ) from e
    
    def get(self, endpoint: str, params: Optional[Dict] = None, **kwargs) -> APIResponse:
        """Make GET request."""
        return self.request('GET', endpoint, params=params, **kwargs)
    
    def post(self, endpoint: str, data: Optional[Dict] = None, **kwargs) -> APIResponse:
        """Make POST request."""
        return self.request('POST', endpoint, data=data, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        stats = self.stats.copy()
        
        if stats['total_requests'] > 0:
            stats['success_rate'] = stats['successful_requests'] / stats['total_requests']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_requests']
        else:
            stats['success_rate'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        if self.cache:
            stats['cache_stats'] = self.cache.get_stats()
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear response cache."""
        if self.cache:
            self.cache.clear()
    
    def __del__(self):
        """Cleanup when client is destroyed."""
        if hasattr(self, 'session'):
            self.session.close()


# ==================== SPECIALIZED API CLIENTS ====================

class DeribitAPIClient(APIClient):
    """Specialized client for Deribit API."""
    
    def __init__(self, test_mode: bool = True, rate_limit: float = 20):
        """
        Initialize Deribit API client.
        
        Args:
            test_mode: Use test API endpoints
            rate_limit: Requests per second
        """
        base_url = "https://test.deribit.com/api/v2" if test_mode else "https://www.deribit.com/api/v2"
        
        config = APIConfig(
            base_url=base_url,
            rate_limit=rate_limit,
            headers={
                'User-Agent': 'Qortfolio-V2/1.0',
                'Accept': 'application/json'
            }
        )
        
        super().__init__(config)
        self.test_mode = test_mode
    
    def get_instruments(self, currency: str = "BTC", kind: str = "option", 
                       expired: bool = False) -> APIResponse:
        """Get available instruments."""
        params = {
            'currency': currency.upper(),
            'kind': kind,
            'expired': expired
        }
        
        return self.get('/public/get_instruments', params=params, cache_ttl=1800)
    
    def get_ticker(self, instrument_name: str) -> APIResponse:
        """Get ticker information for instrument."""
        params = {'instrument_name': instrument_name}
        return self.get('/public/ticker', params=params, cache_ttl=60)
    
    def get_order_book(self, instrument_name: str, depth: int = 5) -> APIResponse:
        """Get order book for instrument."""
        params = {
            'instrument_name': instrument_name,
            'depth': depth
        }
        return self.get('/public/get_order_book', params=params, cache_ttl=30)


class YFinanceAPIWrapper:
    """Wrapper for yfinance with error handling."""
    
    def __init__(self, rate_limit: float = 5):
        """
        Initialize yfinance wrapper.
        
        Args:
            rate_limit: Requests per second
        """
        self.rate_limiter = RateLimiter(rate_limit)
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0
        }
    
    def get_historical_data(self, symbol: str, period: str = "1y", 
                          interval: str = "1d") -> APIResponse:
        """
        Get historical data using yfinance.
        
        Args:
            symbol: Ticker symbol (e.g., 'BTC-USD')
            period: Data period
            interval: Data interval
        
        Returns:
            APIResponse with historical data
        """
        import yfinance as yf
        
        self.rate_limiter.wait_if_needed()
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval, auto_adjust=True)
            
            if data.empty:
                self.stats['failed_requests'] += 1
                return APIResponse(
                    success=False,
                    data=None,
                    status_code=404,
                    response_time=time.time() - start_time,
                    error=f"No data found for {symbol}"
                )
            
            self.stats['successful_requests'] += 1
            return APIResponse(
                success=True,
                data=data,
                status_code=200,
                response_time=time.time() - start_time
            )
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            return APIResponse(
                success=False,
                data=None,
                status_code=500,
                response_time=time.time() - start_time,
                error=str(e)
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get wrapper statistics."""
        stats = self.stats.copy()
        
        if stats['total_requests'] > 0:
            stats['success_rate'] = stats['successful_requests'] / stats['total_requests']
        else:
            stats['success_rate'] = 0.0
        
        return stats


# ==================== UTILITY FUNCTIONS ====================

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, 
                    backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """
    Decorator to retry function on failure.
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Exception types to catch
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        raise
                    
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None  # Should never reach here
        return wrapper
    return decorator


def validate_api_key(api_key: str, min_length: int = 16) -> bool:
    """
    Validate API key format.
    
    Args:
        api_key: API key to validate
        min_length: Minimum required length
    
    Returns:
        True if API key appears valid
    """
    if not isinstance(api_key, str):
        return False
    
    if len(api_key) < min_length:
        return False
    
    # Check for reasonable characters
    if not api_key.replace('-', '').replace('_', '').isalnum():
        return False
    
    return True


def build_query_string(params: Dict[str, Any]) -> str:
    """
    Build query string from parameters.
    
    Args:
        params: Dictionary of parameters
    
    Returns:
        URL-encoded query string
    """
    import urllib.parse
    
    # Filter out None values
    filtered_params = {k: v for k, v in params.items() if v is not None}
    
    return urllib.parse.urlencode(filtered_params)


def parse_api_error(response: requests.Response) -> str:
    """
    Parse error message from API response.
    
    Args:
        response: HTTP response object
    
    Returns:
        Parsed error message
    """
    try:
        error_data = response.json()
        
        # Try common error message patterns
        if isinstance(error_data, dict):
            for key in ['error', 'message', 'detail', 'error_description']:
                if key in error_data:
                    error_value = error_data[key]
                    if isinstance(error_value, dict) and 'message' in error_value:
                        return error_value['message']
                    elif isinstance(error_value, str):
                        return error_value
        
        return str(error_data)
        
    except:
        return f"HTTP {response.status_code}: {response.reason}"


# ==================== EXPORTS ====================

__all__ = [
    # Data classes
    'APIConfig',
    'APIResponse',
    
    # Core classes
    'RateLimiter',
    'APICache',
    'APIClient',
    
    # Specialized clients
    'DeribitAPIClient',
    'YFinanceAPIWrapper',
    
    # Utility functions
    'retry_on_failure',
    'validate_api_key',
    'build_query_string',
    'parse_api_error',
    
    # Constants
    'STATUS_CODES',
    'DEFAULT_TIMEOUT',
    'DEFAULT_MAX_RETRIES',
    'DEFAULT_RATE_LIMIT'
]