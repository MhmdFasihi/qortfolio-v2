# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Data Collection Manager for Qortfolio V2
Unified interface for all data collection operations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union
import time
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .crypto_collector import CryptoCollector
from .deribit_collector import DeribitCollector
from .base_collector import CollectionResult
from core.config import get_config
from core.logging import get_logger


@dataclass
class MarketData:
    """Combined market data from multiple sources."""
    spot_prices: Optional[pd.DataFrame]
    options_data: Optional[pd.DataFrame]
    historical_data: Optional[pd.DataFrame]
    collection_timestamp: datetime
    symbols: List[str]
    sources: List[str]


class SimpleCache:
    """Simple in-memory cache for data collection results."""
    
    def __init__(self, default_ttl: int = 300):
        """
        Initialize cache.
        
        Args:
            default_ttl: Default TTL in seconds
        """
        self.cache: Dict[str, Dict] = {}
        self.default_ttl = default_ttl
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self.cache:
                item = self.cache[key]
                
                # Check if expired
                if datetime.now() < item['expires']:
                    return item['data']
                else:
                    # Remove expired item
                    del self.cache[key]
            
            return None
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache."""
        if ttl is None:
            ttl = self.default_ttl
        
        with self._lock:
            self.cache[key] = {
                'data': data,
                'expires': datetime.now() + timedelta(seconds=ttl),
                'created': datetime.now()
            }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        with self._lock:
            now = datetime.now()
            expired_keys = [
                key for key, item in self.cache.items()
                if now >= item['expires']
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_items = len(self.cache)
            expired_count = sum(
                1 for item in self.cache.values()
                if datetime.now() >= item['expires']
            )
            
            return {
                'total_items': total_items,
                'active_items': total_items - expired_count,
                'expired_items': expired_count
            }


class DataManager:
    """
    Unified data collection manager.
    
    Coordinates data collection from multiple sources:
    - yfinance (historical crypto prices)
    - Deribit (options data, spot prices)
    
    Provides caching, error handling, and unified API.
    """
    
    def __init__(self):
        """Initialize data manager."""
        self.config = get_config()
        self.logger = get_logger("data_manager")
        
        # Initialize collectors
        self.crypto_collector = CryptoCollector()
        self.deribit_collector = DeribitCollector()
        
        # Initialize cache
        cache_enabled = self.config.get('caching.enabled', True)
        if cache_enabled:
            default_ttl = self.config.get('caching.ttl.spot_prices', 300)
            self.cache = SimpleCache(default_ttl)
        else:
            self.cache = None
        
        # Thread pool for concurrent data collection
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="data_collector")
        
        self.logger.info("DataManager initialized", extra={
            "cache_enabled": cache_enabled,
            "collectors": ["crypto_collector", "deribit_collector"]
        })
    
    def get_market_data(self, symbols: List[str], include_options: bool = True, 
                       include_historical: bool = False, **kwargs) -> MarketData:
        """
        Get comprehensive market data for multiple symbols.
        
        Args:
            symbols: List of crypto symbols (e.g., ["BTC", "ETH"])
            include_options: Include options data
            include_historical: Include historical price data
            **kwargs: Additional parameters for data collection
            
        Returns:
            MarketData object with collected data
        """
        start_time = time.time()
        
        self.logger.info(f"Collecting market data for {len(symbols)} symbols", extra={
            "symbols": symbols,
            "include_options": include_options,
            "include_historical": include_historical
        })
        
        # Initialize results
        spot_prices_data = []
        options_data_list = []
        historical_data_list = []
        sources_used = set()
        
        # Collect data concurrently
        futures = []
        
        # Submit spot price collection tasks
        for symbol in symbols:
            future = self.executor.submit(self._get_spot_price_with_cache, symbol)
            futures.append(('spot', symbol, future))
        
        # Submit options data collection tasks (if requested)
        if include_options:
            for symbol in symbols:
                if self._symbol_has_options(symbol):
                    future = self.executor.submit(self._get_options_data_with_cache, symbol, **kwargs)
                    futures.append(('options', symbol, future))
        
        # Submit historical data collection tasks (if requested)
        if include_historical:
            for symbol in symbols:
                period = kwargs.get('period', '30d')
                interval = kwargs.get('interval', '1d')
                future = self.executor.submit(self._get_historical_data_with_cache, symbol, period, interval)
                futures.append(('historical', symbol, future))
        
        # Collect results
        for data_type, symbol, future in futures:
            try:
                result = future.result(timeout=30)  # 30 second timeout
                
                if result and hasattr(result, 'success') and result.success:
                    sources_used.add(result.source)
                    
                    if data_type == 'spot':
                        if result.data is not None and not result.data.empty:
                            spot_data = {
                                'symbol': symbol,
                                'price': result.data,
                                'timestamp': result.timestamp,
                                'source': result.source
                            }
                            spot_prices_data.append(spot_data)
                    
                    elif data_type == 'options':
                        if result.data is not None and not result.data.empty:
                            options_data_list.append(result.data)
                    
                    elif data_type == 'historical':
                        if result.data is not None and not result.data.empty:
                            historical_data_list.append(result.data)
                
            except Exception as e:
                self.logger.warning(f"Failed to collect {data_type} data for {symbol}: {e}")
                continue
        
        # Combine results into DataFrames
        spot_prices_df = self._combine_spot_prices(spot_prices_data) if spot_prices_data else None
        options_df = self._combine_options_data(options_data_list) if options_data_list else None
        historical_df = self._combine_historical_data(historical_data_list) if historical_data_list else None
        
        collection_time = time.time() - start_time
        
        self.logger.info(f"Market data collection completed", extra={
            "symbols": symbols,
            "collection_time": collection_time,
            "spot_records": len(spot_prices_df) if spot_prices_df is not None else 0,
            "options_records": len(options_df) if options_df is not None else 0,
            "historical_records": len(historical_df) if historical_df is not None else 0,
            "sources": list(sources_used)
        })
        
        return MarketData(
            spot_prices=spot_prices_df,
            options_data=options_df,
            historical_data=historical_df,
            collection_timestamp=datetime.now(),
            symbols=symbols,
            sources=list(sources_used)
        )
    
    def get_spot_price(self, symbol: str, use_cache: bool = True) -> Optional[float]:
        """
        Get current spot price for a symbol.
        
        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH")
            use_cache: Use cached data if available
            
        Returns:
            Current spot price or None if failed
        """
        # Try cache first
        if use_cache and self.cache:
            cache_key = f"spot_price:{symbol}"
            cached_price = self.cache.get(cache_key)
            if cached_price is not None:
                return cached_price
        
        # Try Deribit first (if symbol supported)
        price = None
        if self._symbol_has_options(symbol):
            price = self.deribit_collector.get_spot_price(symbol)
            if price is not None:
                # Cache the result
                if self.cache:
                    ttl = self.config.get_cache_ttl('spot_prices')
                    self.cache.set(f"spot_price:{symbol}", price, ttl)
                return price
        
        # Fallback to yfinance
        price = self.crypto_collector.get_current_price(symbol)
        if price is not None and self.cache:
            ttl = self.config.get_cache_ttl('spot_prices')
            self.cache.set(f"spot_price:{symbol}", price, ttl)
        
        return price
    
    def get_options_chain(self, symbol: str, expiry_date: Optional[str] = None, 
                         use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Get options chain for a symbol.
        
        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH")
            expiry_date: Specific expiry date or None for all
            use_cache: Use cached data if available
            
        Returns:
            DataFrame with options chain or None if failed
        """
        if not self._symbol_has_options(symbol):
            self.logger.warning(f"Symbol {symbol} does not have options on Deribit")
            return None
        
        # Try cache first
        if use_cache and self.cache:
            cache_key = f"options_chain:{symbol}:{expiry_date or 'all'}"
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Collect options data
        result = self.deribit_collector.collect_data(symbol, kind="option", expired=False)
        
        if result.success and result.data is not None:
            data = result.data
            
            # Filter by expiry if specified
            if expiry_date and 'expiry_date' in data.columns:
                try:
                    target_date = pd.to_datetime(expiry_date).date()
                    data = data[data['expiry_date'] == target_date]
                except Exception as e:
                    self.logger.error(f"Invalid expiry date format '{expiry_date}': {e}")
                    return None
            
            # Cache the result
            if self.cache:
                ttl = self.config.get_cache_ttl('options_data')
                cache_key = f"options_chain:{symbol}:{expiry_date or 'all'}"
                self.cache.set(cache_key, data, ttl)
            
            return data
        
        return None
    
    def get_historical_data(self, symbol: str, period: str = "1y", interval: str = "1d", 
                          use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Get historical price data for a symbol.
        
        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH")
            period: Data period (e.g., "1y", "6mo", "3mo")
            interval: Data interval (e.g., "1d", "1h", "15m")
            use_cache: Use cached data if available
            
        Returns:
            DataFrame with historical data or None if failed
        """
        # Try cache first
        if use_cache and self.cache:
            cache_key = f"historical:{symbol}:{period}:{interval}"
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Collect historical data
        result = self.crypto_collector.collect_data(symbol, period=period, interval=interval)
        
        if result.success and result.data is not None:
            # Cache the result
            if self.cache:
                ttl = self.config.get_cache_ttl('historical_data')
                cache_key = f"historical:{symbol}:{period}:{interval}"
                self.cache.set(cache_key, result.data, ttl)
            
            return result.data
        
        return None
    
    def get_multi_symbol_data(self, symbols: List[str], data_type: str = "spot", 
                            **kwargs) -> Dict[str, Any]:
        """
        Get data for multiple symbols efficiently.
        
        Args:
            symbols: List of crypto symbols
            data_type: Type of data ("spot", "options", "historical")
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping symbols to their data
        """
        results = {}
        
        if data_type == "spot":
            # Use concurrent collection for spot prices
            futures = []
            for symbol in symbols:
                future = self.executor.submit(self.get_spot_price, symbol, **kwargs)
                futures.append((symbol, future))
            
            for symbol, future in futures:
                try:
                    price = future.result(timeout=10)
                    if price is not None:
                        results[symbol] = price
                except Exception as e:
                    self.logger.warning(f"Failed to get spot price for {symbol}: {e}")
        
        elif data_type == "historical":
            # Use batch collection from crypto collector
            crypto_results = self.crypto_collector.collect_multiple_symbols(symbols, **kwargs)
            for symbol, result in crypto_results.items():
                if result.success:
                    results[symbol] = result.data
        
        elif data_type == "options":
            # Collect options for symbols that support them
            for symbol in symbols:
                if self._symbol_has_options(symbol):
                    options_data = self.get_options_chain(symbol, **kwargs)
                    if options_data is not None:
                        results[symbol] = options_data
        
        return results
    
    def _get_spot_price_with_cache(self, symbol: str) -> CollectionResult:
        """Get spot price and return as CollectionResult."""
        price = self.get_spot_price(symbol, use_cache=True)
        
        if price is not None:
            # Return as CollectionResult for consistency
            return CollectionResult(
                success=True,
                data=price,
                error=None,
                records_count=1,
                response_time=0.0,
                timestamp=datetime.now(),
                source="deribit" if self._symbol_has_options(symbol) else "yfinance"
            )
        else:
            return CollectionResult(
                success=False,
                data=None,
                error=f"Failed to get spot price for {symbol}",
                records_count=0,
                response_time=0.0,
                timestamp=datetime.now(),
                source="unknown"
            )
    
    def _get_options_data_with_cache(self, symbol: str, **kwargs) -> CollectionResult:
        """Get options data with caching."""
        return self.deribit_collector.collect_data(symbol, kind="option", **kwargs)
    
    def _get_historical_data_with_cache(self, symbol: str, period: str, interval: str) -> CollectionResult:
        """Get historical data with caching."""
        return self.crypto_collector.collect_data(symbol, period=period, interval=interval)
    
    def _symbol_has_options(self, symbol: str) -> bool:
        """Check if symbol has options available on Deribit."""
        return symbol.upper() in [s.upper() for s in self.config.deribit_currencies]
    
    def _combine_spot_prices(self, spot_data_list: List[Dict]) -> pd.DataFrame:
        """Combine spot price data into DataFrame."""
        if not spot_data_list:
            return pd.DataFrame()
        
        df_data = []
        for item in spot_data_list:
            df_data.append({
                'symbol': item['symbol'],
                'price': item['price'],
                'timestamp': item['timestamp'],
                'source': item['source']
            })
        
        return pd.DataFrame(df_data)
    
    def _combine_options_data(self, options_data_list: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple options DataFrames."""
        if not options_data_list:
            return pd.DataFrame()
        
        return pd.concat(options_data_list, ignore_index=True)
    
    def _combine_historical_data(self, historical_data_list: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple historical DataFrames."""
        if not historical_data_list:
            return pd.DataFrame()
        
        return pd.concat(historical_data_list, ignore_index=True)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        else:
            return {"cache_enabled": False}
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        if self.cache:
            self.cache.clear()
            self.logger.info("Cache cleared")
    
    def get_collector_stats(self) -> Dict[str, Any]:
        """Get statistics from all collectors."""
        return {
            "crypto_collector": self.crypto_collector.get_statistics(),
            "deribit_collector": self.deribit_collector.get_statistics(),
            "cache": self.get_cache_stats()
        }
    
    def __del__(self):
        """Cleanup when manager is destroyed."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# Global data manager instance
_data_manager_instance: Optional[DataManager] = None


def get_data_manager() -> DataManager:
    """Get the global data manager instance."""
    global _data_manager_instance
    if _data_manager_instance is None:
        _data_manager_instance = DataManager()
    return _data_manager_instance


# Convenience functions
def get_current_prices(symbols: List[str]) -> Dict[str, float]:
    """Get current prices for multiple symbols."""
    dm = get_data_manager()
    return dm.get_multi_symbol_data(symbols, data_type="spot")


def get_options_data(symbol: str, expiry_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Get options chain for a symbol."""
    dm = get_data_manager()
    return dm.get_options_chain(symbol, expiry_date)


def get_crypto_history(symbol: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """Get historical data for a crypto symbol."""
    dm = get_data_manager()
    return dm.get_historical_data(symbol, period, interval)


if __name__ == "__main__":
    # Test the data manager
    print("🧪 Testing Data Collection Manager")
    print("=" * 40)
    
    dm = DataManager()
    
    # Test spot prices
    print("Testing spot prices...")
    symbols = ["BTC", "ETH"]
    prices = get_current_prices(symbols)
    print(f"✅ Current prices: {prices}")
    
    # Test market data collection
    print("\nTesting comprehensive market data...")
    market_data = dm.get_market_data(
        symbols=["BTC"],
        include_options=True,
        include_historical=True,
        period="5d",
        interval="1d"
    )
    
    print(f"📊 Market data collected:")
    print(f"  Spot prices: {len(market_data.spot_prices) if market_data.spot_prices is not None else 0}")
    print(f"  Options: {len(market_data.options_data) if market_data.options_data is not None else 0}")
    print(f"  Historical: {len(market_data.historical_data) if market_data.historical_data is not None else 0}")
    print(f"  Sources: {market_data.sources}")
    
    # Test statistics
    print("\nData manager statistics:")
    stats = dm.get_collector_stats()
    for collector, collector_stats in stats.items():
        print(f"  {collector}:")
        for key, value in collector_stats.items():
            print(f"    {key}: {value}")
    
    print("\n🎉 Data manager test completed!")