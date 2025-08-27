# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Cryptocurrency data collector using yfinance.
Fetches real historical price data and stores in MongoDB.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import logging

from .base_collector import BaseDataCollector
from src.core.config import config
from src.core.exceptions import (
    DataCollectionError,
    ValidationError,
    InvalidTickerError
)
from src.core.utils.validation import ValidationUtils
from src.core.database.models import PriceData

logger = logging.getLogger(__name__)

class CryptoCollector(BaseDataCollector):
    """
    Collector for cryptocurrency historical data using yfinance.
    Fetches OHLCV data and stores in MongoDB.
    """
    
    def __init__(
        self,
        rate_limit: float = 2.0,  # 2 requests per second for yfinance
        cache_ttl: int = 300,  # 5 minutes cache
        enable_cache: bool = True
    ):
        """
        Initialize crypto data collector.
        
        Args:
            rate_limit: Maximum requests per second to yfinance
            cache_ttl: Cache time-to-live in seconds
            enable_cache: Whether to enable caching
        """
        super().__init__(
            name="CryptoCollector",
            rate_limit=rate_limit or config.api.yfinance_rate_limit,
            cache_ttl=cache_ttl,
            enable_cache=enable_cache
        )
        
        # Validate crypto sectors are loaded
        if not config.crypto_sectors:
            raise ImportError("Crypto sectors configuration is required")
        
        self.crypto_sectors = config.crypto_sectors
        
    def _get_yfinance_ticker(self, symbol: str) -> str:
        """
        Convert crypto symbol to yfinance ticker.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            yfinance ticker (e.g., 'BTC-USD', 'ETH-USD')
        """
        return self.crypto_sectors.get_yfinance_ticker(symbol)
    
    def _validate_ticker(self, symbol: str) -> bool:
        """
        Validate if ticker exists in our configuration.
        
        Args:
            symbol: Ticker to validate
            
        Returns:
            True if valid, raises exception otherwise
        """
        is_valid, message = self.crypto_sectors.validate_ticker(symbol)
        if not is_valid:
            # Check if it's a major crypto not in sectors
            if symbol in ['BTC', 'ETH']:
                return True
            raise InvalidTickerError(message)
        return True
    
    async def fetch_data(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch historical price data from yfinance.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Validate ticker
            self._validate_ticker(symbol)
            
            # Get yfinance ticker
            yf_ticker = self._get_yfinance_ticker(symbol)
            
            logger.info(f"Fetching data for {symbol} ({yf_ticker}) - period: {period}, interval: {interval}")
            
            # Create ticker object
            ticker = yf.Ticker(yf_ticker)
            
            # Fetch historical data
            if start_date and end_date:
                # Use specific date range
                hist_data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,
                    prepost=True
                )
            else:
                # Use period
                hist_data = ticker.history(
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    prepost=True
                )
            
            if hist_data.empty:
                raise DataCollectionError(f"No data returned for {symbol}")
            
            # Add symbol column
            hist_data['Symbol'] = symbol
            hist_data['YFTicker'] = yf_ticker
            
            # Get additional info if available
            try:
                info = ticker.info
                if info:
                    hist_data.attrs['info'] = {
                        'name': info.get('shortName', symbol),
                        'market_cap': info.get('marketCap', 0),
                        'volume_24h': info.get('volume24Hr', 0),
                        'price_change_24h': info.get('regularMarketChangePercent', 0),
                        'sector': self.crypto_sectors.get_ticker_sector(symbol)
                    }
            except Exception as e:
                logger.warning(f"Could not fetch additional info for {symbol}: {e}")
                hist_data.attrs['info'] = {'sector': self.crypto_sectors.get_ticker_sector(symbol)}
            
            return hist_data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            raise DataCollectionError(f"Failed to fetch {symbol}: {e}")
    
    async def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate fetched price data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if data is None or data.empty:
                logger.error("Empty data received")
                return False
            
            # Check required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    logger.error(f"Missing required column: {col}")
                    return False
            
            # Validate price values
            if (data['Low'] > data['High']).any():
                logger.error("Invalid data: Low > High")
                return False
            
            if ((data['Open'] > data['High']) | (data['Open'] < data['Low'])).any():
                logger.error("Invalid data: Open outside of High-Low range")
                return False
            
            if ((data['Close'] > data['High']) | (data['Close'] < data['Low'])).any():
                logger.error("Invalid data: Close outside of High-Low range")
                return False
            
            # Check for negative values
            if (data[['Open', 'High', 'Low', 'Close']] < 0).any().any():
                logger.error("Invalid data: Negative prices")
                return False
            
            # Check for NaN values in critical columns
            if data[required_columns].isna().any().any():
                logger.warning("Data contains NaN values, will clean")
                # We'll handle NaN in process_data
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False
    
    async def process_data(self, raw_data: pd.DataFrame) -> List[Dict]:
        """
        Process raw price data for storage.
        
        Args:
            raw_data: DataFrame with raw price data
            
        Returns:
            List of processed price documents
        """
        try:
            # Clean data
            data = raw_data.copy()
            
            # Forward fill NaN values (common in crypto markets during low volume)
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Drop any remaining rows with NaN
            data = data.dropna()
            
            # Process each row
            processed_data = []
            symbol = data['Symbol'].iloc[0] if 'Symbol' in data.columns else 'UNKNOWN'
            sector = data.attrs.get('info', {}).get('sector', 'Unknown')
            
            for timestamp, row in data.iterrows():
                price_doc = {
                    'symbol': symbol,
                    'timestamp': timestamp.to_pydatetime(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume']),
                    'source': 'yfinance',
                    'sector': sector,
                    'metadata': {
                        'yf_ticker': row.get('YFTicker', f"{symbol}-USD"),
                        'collected_at': datetime.utcnow()
                    }
                }
                
                # Validate using our PriceData model
                try:
                    price_data = PriceData(
                        symbol=price_doc['symbol'],
                        open=price_doc['open'],
                        high=price_doc['high'],
                        low=price_doc['low'],
                        close=price_doc['close'],
                        volume=price_doc['volume'],
                        timestamp=price_doc['timestamp'],
                        source=price_doc['source']
                    )
                    processed_data.append(price_data.to_dict())
                except Exception as e:
                    logger.warning(f"Skipping invalid row: {e}")
                    continue
            
            logger.info(f"Processed {len(processed_data)} price records for {symbol}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Data processing error: {e}")
            raise DataCollectionError(f"Failed to process data: {e}")
    
    # === Convenience Methods ===
    
    async def get_current_price(self, symbol: str) -> Dict[str, float]:
        """
        Get current price for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            Dictionary with current price information
        """
        try:
            # Fetch last 1 day of data
            data = await self.collect(
                symbol=symbol,
                period="1d",
                interval="1m",
                use_cache=False  # Always get fresh price
            )
            
            if not data:
                raise DataCollectionError(f"No price data for {symbol}")
            
            # Get the latest record
            latest = data[-1]
            
            return {
                'symbol': symbol,
                'price': latest['close'],
                'open': latest['open'],
                'high': latest['high'],
                'low': latest['low'],
                'volume': latest['volume'],
                'timestamp': latest['timestamp'],
                'change_24h': ((latest['close'] - latest['open']) / latest['open']) * 100
            }
            
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            raise
    
    async def get_multiple_symbols(
        self,
        symbols: List[str],
        period: str = "1mo",
        interval: str = "1d"
    ) -> Dict[str, List[Dict]]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of cryptocurrency symbols
            period: Time period for historical data
            interval: Data interval
            
        Returns:
            Dictionary mapping symbols to their data
        """
        results = {}
        
        for symbol in symbols:
            try:
                data = await self.collect(
                    symbol=symbol,
                    period=period,
                    interval=interval
                )
                results[symbol] = data
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                results[symbol] = []
        
        return results
    
    async def get_sector_data(
        self,
        sector: str,
        period: str = "1mo",
        interval: str = "1d",
        limit: Optional[int] = None
    ) -> Dict[str, List[Dict]]:
        """
        Fetch data for all cryptocurrencies in a sector.
        
        Args:
            sector: Sector name (e.g., 'DeFi', 'Infrastructure', 'AI')
            period: Time period for historical data
            interval: Data interval
            limit: Maximum number of symbols to fetch (None for all)
            
        Returns:
            Dictionary mapping symbols to their data
        """
        # Get sector tickers
        tickers = self.crypto_sectors.get_sector_tickers(sector)
        
        if not tickers:
            raise ValidationError(f"Invalid sector: {sector}")
        
        # Apply limit if specified
        if limit:
            tickers = tickers[:limit]
        
        logger.info(f"Fetching data for {len(tickers)} symbols in {sector} sector")
        
        return await self.get_multiple_symbols(tickers, period, interval)
    
    async def store_historical_data(
        self,
        symbol: str,
        period: str = "3mo",
        interval: str = "1d"
    ) -> int:
        """
        Fetch and store historical data in MongoDB.
        
        Args:
            symbol: Cryptocurrency symbol
            period: Time period for historical data
            interval: Data interval
            
        Returns:
            Number of records stored
        """
        try:
            # Fetch data
            data = await self.collect(
                symbol=symbol,
                period=period,
                interval=interval,
                use_cache=False  # Don't cache when storing
            )
            
            if not data:
                logger.warning(f"No data to store for {symbol}")
                return 0
            
            # Store in MongoDB
            await self.store_data("price_data", data)
            
            return len(data)
            
        except Exception as e:
            logger.error(f"Failed to store historical data for {symbol}: {e}")
            raise

# === Module testing ===

async def test_crypto_collector():
    """Test the crypto collector functionality."""
    collector = CryptoCollector()
    
    print("\n📊 Testing Crypto Collector")
    print("=" * 50)
    
    try:
        # Test 1: Fetch BTC data
        print("\n1️⃣ Fetching BTC historical data...")
        btc_data = await collector.collect(
            symbol="BTC",
            period="5d",
            interval="1d"
        )
        print(f"   ✅ Fetched {len(btc_data)} BTC records")
        if btc_data:
            latest = btc_data[-1]
            print(f"   Latest: ${latest['close']:.2f} on {latest['timestamp']}")
        
        # Test 2: Get current price
        print("\n2️⃣ Getting current ETH price...")
        eth_price = await collector.get_current_price("ETH")
        print(f"   ✅ ETH Price: ${eth_price['price']:.2f}")
        print(f"   24h Change: {eth_price['change_24h']:.2f}%")
        
        # Test 3: Fetch multiple symbols
        print("\n3️⃣ Fetching multiple symbols...")
        symbols = ["BTC", "ETH", "SOL"]
        multi_data = await collector.get_multiple_symbols(symbols, period="1d")
        for symbol, data in multi_data.items():
            if data:
                print(f"   ✅ {symbol}: {len(data)} records")
        
        # Test 4: Get sector data
        print("\n4️⃣ Fetching DeFi sector data (limited to 3)...")
        defi_data = await collector.get_sector_data("DeFi", period="1d", limit=3)
        for symbol, data in defi_data.items():
            if data:
                print(f"   ✅ {symbol}: {len(data)} records")
        
        # Show statistics
        print("\n📊 Collector Statistics:")
        stats = collector.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
    finally:
        await collector.close()

if __name__ == "__main__":
    # Run tests
    asyncio.run(test_crypto_collector())
