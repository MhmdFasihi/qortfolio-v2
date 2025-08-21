# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
API utility functions for external service integration.
Handles rate limiting, retries, and error handling.
"""

import asyncio
import time
from typing import Any, Dict, Optional, Callable
from functools import wraps
import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
from ..exceptions import APIConnectionError, RateLimitError

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, calls_per_second: float = 1.0):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
    
    async def wait_if_needed(self):
        """Wait if necessary to respect rate limit."""
        current_time = time.time()
        time_since_last = current_time - self.last_call
        
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_call = time.time()

def rate_limit(calls_per_second: float = 1.0):
    """Decorator for rate limiting."""
    limiter = RateLimiter(calls_per_second)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            await limiter.wait_if_needed()
            return await func(*args, **kwargs)
        return wrapper
    return decorator

class APIUtils:
    """Utility functions for API interactions."""
    
    @staticmethod
    def create_session(
        max_retries: int = 3,
        backoff_factor: float = 0.3,
        timeout: int = 30
    ) -> requests.Session:
        """
        Create requests session with retry strategy.
        
        Args:
            max_retries: Maximum number of retries
            backoff_factor: Backoff factor for retries
            timeout: Request timeout in seconds
        """
        session = requests.Session()
        
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        session.timeout = timeout
        
        return session
    
    @staticmethod
    async def async_get(
        url: str,
        headers: Optional[Dict] = None,
        timeout: int = 30,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Async GET request with retry logic.
        
        Args:
            url: URL to request
            headers: Request headers
            timeout: Timeout in seconds
            max_retries: Maximum retries
        """
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url, 
                        headers=headers, 
                        timeout=aiohttp.ClientTimeout(total=timeout)
                    ) as response:
                        if response.status == 429:
                            raise RateLimitError("Rate limit exceeded")
                        response.raise_for_status()
                        return await response.json()
                        
            except aiohttp.ClientError as e:
                logger.warning(f"API request failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise APIConnectionError(f"Failed after {max_retries} attempts: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    @staticmethod
    def handle_api_error(response: requests.Response) -> None:
        """Handle API error responses."""
        if response.status_code == 429:
            raise RateLimitError(
                f"Rate limit exceeded. Retry after: {response.headers.get('Retry-After', 'unknown')}"
            )
        elif response.status_code >= 500:
            raise APIConnectionError(f"Server error: {response.status_code}")
        elif response.status_code >= 400:
            raise APIConnectionError(
                f"Client error: {response.status_code} - {response.text}"
            )

if __name__ == "__main__":
    # Test API utilities
    print("API Utils initialized")
    session = APIUtils.create_session()
    print(f"✅ Session created with retry strategy")
