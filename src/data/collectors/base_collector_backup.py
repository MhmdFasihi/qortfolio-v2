# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Base data collector class for Qortfolio V2.
Provides common functionality for all data collectors including
rate limiting, retry logic, caching, and error handling.
"""

import asyncio
import time
import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from functools import wraps
import logging
from dataclasses import dataclass
import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import our modules
from ...core.config import config
from ...core.exceptions import (
    DataCollectionError,
    APIConnectionError,
    RateLimitError,
    ValidationError
)
from ...core.database.connection import db_connection

logger = logging.getLogger(__name__)

@dataclass
class CollectorStats:
    """Statistics for data collector performance."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_latency_ms: float = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency in milliseconds."""
        if self.successful_requests == 0:
            return 0
        return self.total_latency_ms / self.successful_requests
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests == 0:
            return 0
        return (self.cache_hits / total_cache_requests) * 100

class RateLimiter:
    """Rate limiter implementation with token bucket algorithm."""
    
    def __init__(self, calls_per_second: float = 1.0):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_second: Maximum API calls per second
        """
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """Acquire permission to make an API call."""
        async with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            
            if time_since_last_call < self.min_interval:
                sleep_time = self.min_interval - time_since_last_call
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
            
            self.last_call_time = time.time()

def with_retry(max_retries: int = 3, backoff_factor: float = 1.0):
    """
    Decorator for retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (APIConnectionError, RateLimitError) as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        wait_time = backoff_factor * (2 ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed")
                        
            raise last_exception or DataCollectionError("Unknown error in retry logic")
        
        return wrapper
    return decorator

class BaseDataCollector(ABC):
    """
    Abstract base class for all data collectors.
    Provides common functionality like rate limiting, caching, and error handling.
    """
    
    def __init__(
        self,
        name: str,
        rate_limit: Optional[float] = None,
        cache_ttl: Optional[int] = None,
        enable_cache: bool = True
    ):
        """
        Initialize base data collector.
        
        Args:
            name: Name of the collector
            rate_limit: API calls per second (None for no limit)
            cache_ttl: Cache time-to-live in seconds
            enable_cache: Whether to enable caching
        """
        self.name = name
        self.stats = CollectorStats()
        
        # Rate limiting
        if rate_limit:
            self.rate_limiter = RateLimiter(rate_limit)
        else:
            self.rate_limiter = None
        
        # Caching
        self.enable_cache = enable_cache and config.app_settings.get("enable_caching", False)
        self.cache_ttl = cache_ttl or config.app_settings.get("cache_ttl_seconds", 300)
        self.cache: Dict[str, Dict[str, Any]] = {}
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        self._sync_session: Optional[requests.Session] = None
        
        # Database connection
        self.db_connection = db_connection
        
        logger.info(
            f"Initialized {name} collector "
            f"(rate_limit={rate_limit}, cache={enable_cache}, ttl={cache_ttl}s)"
        )
    
    # === Abstract Methods (Must be implemented by subclasses) ===
    
    @abstractmethod
    async def fetch_data(self, **kwargs) -> Any:
        """
        Fetch data from the source.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    async def validate_data(self, data: Any) -> bool:
        """
        Validate fetched data.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    async def process_data(self, raw_data: Any) -> Any:
        """
        Process raw data into desired format.
        Must be implemented by subclasses.
        """
        pass
    
    # === Session Management ===
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create async HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": f"Qortfolio-V2/{self.name}"}
            )
        return self._session
    
    def get_sync_session(self) -> requests.Session:
        """Get or create sync HTTP session with retry strategy."""
        if self._sync_session is None:
            self._sync_session = requests.Session()
            
            # Configure retry strategy
            retry_strategy = Retry(
                total=3,
                backoff_factor=0.3,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._sync_session.mount("http://", adapter)
            self._sync_session.mount("https://", adapter)
            
            self._sync_session.headers.update({
                "User-Agent": f"Qortfolio-V2/{self.name}"
            })
            
        return self._sync_session
    
    async def close(self):
        """Close all sessions and connections."""
        if self._session:
            await self._session.close()
        if self._sync_session:
            self._sync_session.close()
        logger.info(f"Closed {self.name} collector sessions")
    
    # === Cache Management ===
    
    def _generate_cache_key(self, **kwargs) -> str:
        """Generate unique cache key from parameters."""
        # Sort kwargs for consistent key generation
        sorted_params = json.dumps(kwargs, sort_keys=True)
        hash_obj = hashlib.md5(sorted_params.encode())
        return f"{self.name}:{hash_obj.hexdigest()}"
    
    async def get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if available and not expired."""
        if not self.enable_cache:
            return None
        
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            
            # Check if cache is still valid
            if datetime.utcnow() < cached["expires_at"]:
                self.stats.cache_hits += 1
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached["data"]
            else:
                # Remove expired cache
                del self.cache[cache_key]
        
        self.stats.cache_misses += 1
        return None
    
    async def set_cached_data(self, cache_key: str, data: Any):
        """Store data in cache with TTL."""
        if not self.enable_cache:
            return
        
        self.cache[cache_key] = {
            "data": data,
            "expires_at": datetime.utcnow() + timedelta(seconds=self.cache_ttl),
            "created_at": datetime.utcnow()
        }
        logger.debug(f"Cached data for key: {cache_key}")
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        logger.info(f"Cleared cache for {self.name} collector")
    
    # === Rate Limiting ===
    
    async def _apply_rate_limit(self):
        """Apply rate limiting if configured."""
        if self.rate_limiter:
            await self.rate_limiter.acquire()
    
    # === Main Data Collection Method ===
    
    @with_retry(max_retries=3, backoff_factor=1.0)
    async def collect(self, use_cache: bool = True, **kwargs) -> Any:
        """
        Main method to collect data with all features.
        
        Args:
            use_cache: Whether to use cache for this request
            **kwargs: Parameters to pass to fetch_data
            
        Returns:
            Processed data
            
        Raises:
            DataCollectionError: If data collection fails
        """
        start_time = time.time()
        self.stats.total_requests += 1
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(**kwargs)
            if use_cache and self.enable_cache:
                cached_data = await self.get_cached_data(cache_key)
                if cached_data is not None:
                    return cached_data
            
            # Apply rate limiting
            await self._apply_rate_limit()
            
            # Fetch fresh data
            logger.debug(f"Fetching data for {self.name} with params: {kwargs}")
            raw_data = await self.fetch_data(**kwargs)
            
            # Validate data
            if not await self.validate_data(raw_data):
                raise ValidationError(f"Data validation failed for {self.name}")
            
            # Process data
            processed_data = await self.process_data(raw_data)
            
            # Cache processed data
            if use_cache and self.enable_cache:
                await self.set_cached_data(cache_key, processed_data)
            
            # Update statistics
            self.stats.successful_requests += 1
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats.total_latency_ms += elapsed_ms
            
            logger.info(
                f"Successfully collected data from {self.name} "
                f"(elapsed: {elapsed_ms:.1f}ms)"
            )
            
            return processed_data
            
        except Exception as e:
            self.stats.failed_requests += 1
            logger.error(f"Failed to collect data from {self.name}: {e}")
            raise DataCollectionError(f"Collection failed for {self.name}: {e}")
    
    # === Statistics and Monitoring ===
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        return {
            "name": self.name,
            "total_requests": self.stats.total_requests,
            "successful_requests": self.stats.successful_requests,
            "failed_requests": self.stats.failed_requests,
            "rate_limited_requests": self.stats.rate_limited_requests,
            "success_rate": f"{self.stats.success_rate:.1f}%",
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "cache_hit_rate": f"{self.stats.cache_hit_rate:.1f}%",
            "average_latency_ms": f"{self.stats.average_latency_ms:.1f}"
        }
    
    def reset_stats(self):
        """Reset collector statistics."""
        self.stats = CollectorStats()
        logger.info(f"Reset statistics for {self.name} collector")
    
    # === Database Operations ===
    
    async def store_data(self, collection_name: str, data: Union[Dict, List[Dict]]):
        """
        Store collected data in MongoDB.
        
        Args:
            collection_name: Name of the MongoDB collection
            data: Data to store (single document or list)
        """
        try:
            db = await self.db_connection.get_database_async()
            collection = db[collection_name]
            
            if isinstance(data, list):
                if data:  # Only insert if list is not empty
                    result = await collection.insert_many(data)
                    logger.info(
                        f"Stored {len(result.inserted_ids)} documents "
                        f"in {collection_name}"
                    )
            else:
                result = await collection.insert_one(data)
                logger.info(f"Stored document in {collection_name}: {result.inserted_id}")
                
        except Exception as e:
            logger.error(f"Failed to store data in {collection_name}: {e}")
            raise DatabaseOperationError(f"Storage failed: {e}")
    
    def __str__(self) -> str:
        """String representation of collector."""
        return f"{self.__class__.__name__}(name={self.name})"
    
    def __repr__(self) -> str:
        """Detailed representation of collector."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"cache={self.enable_cache}, "
            f"stats={self.stats.total_requests} requests"
            f")"
        )

if __name__ == "__main__":
    # This is an abstract class, cannot be instantiated directly
    print("✅ Base Data Collector module loaded successfully")
    print("This is an abstract base class for all data collectors.")
    print("\nFeatures provided:")
    print("  - Rate limiting with token bucket algorithm")
    print("  - Retry logic with exponential backoff")
    print("  - In-memory caching with TTL")
    print("  - Session management (async and sync)")
    print("  - Statistics tracking")
    print("  - Database storage integration")
