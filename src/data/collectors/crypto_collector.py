# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Cryptocurrency Data Collector using yfinance
Collects historical price data and real-time quotes for cryptocurrencies
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
import time

from .base_collector import BaseDataCollector, CollectionResult, DataCollectionError
from .base_collector import validate_dataframe_structure, clean_numeric_data
from core.logging import log_data_collection


class CryptoCollector(BaseDataCollector):
    """
    Cryptocurrency data collector using yfinance.
    
    Provides:
    - Historical price data (OHLCV)
    - Real-time quotes
    - Multiple timeframes
    - Automatic symbol mapping from configuration
    """
    
    def __init__(self):
        """Initialize crypto data collector."""
        super().__init__("yfinance")
        
        # Required columns for price data
        self.required_price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.numeric_price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        self.logger.info("CryptoCollector initialized for yfinance data")
    
    def collect_data(self, symbol: str, **kwargs) -> CollectionResult:
        """
        Collect cryptocurrency data.
        
        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH")
            **kwargs: Additional parameters:
                - period: Data period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
                - interval: Data interval ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
                - start: Start date (datetime or string)
                - end: End date (datetime or string)
                - include_dividends: Include dividend data
                
        Returns:
            Collection result with historical price data
        """
        start_time = time.time()
        
        # Validate symbol
        if not self._validate_symbol(symbol):
            error = f"Symbol '{symbol}' not supported or not enabled in configuration"
            self.logger.warning(error, extra={"symbol": symbol})
            return self._create_error_result(error, symbol)
        
        # Get yfinance ticker from configuration
        ticker_symbol = self.config.get_yfinance_ticker(symbol)
        if not ticker_symbol:
            error = f"No yfinance ticker found for symbol '{symbol}'"
            self.logger.error(error, extra={"symbol": symbol})
            return self._create_error_result(error, symbol)
        
        try:
            # Extract parameters
            period = kwargs.get('period', self.config.get('yfinance_api.defaults.period', '1y'))
            interval = kwargs.get('interval', self.config.get('yfinance_api.defaults.interval', '1d'))
            start = kwargs.get('start')
            end = kwargs.get('end')
            # include_dividends = kwargs.get('include_dividends', False)  # Removed - API issue
            
            self.logger.debug(f"Collecting data for {ticker_symbol}", extra={
                "symbol": symbol,
                "ticker": ticker_symbol,
                "period": period,
                "interval": interval,
                "start": start,
                "end": end
            })
            
            # Create yfinance ticker object
            ticker = yf.Ticker(ticker_symbol)
            
            # Collect historical data
            if start and end:
                # Use date range
                data = ticker.history(
                    start=start,
                    end=end,
                    interval=interval,
                    # include_dividends removed - caused API errors
                    auto_adjust=self.config.get('yfinance_api.defaults.auto_adjust', True),
                    back_adjust=self.config.get('yfinance_api.defaults.back_adjust', False)
                )
            else:
                # Use period
                data = ticker.history(
                    period=period,
                    interval=interval,
                    # include_dividends removed - caused API errors
                    auto_adjust=self.config.get('yfinance_api.defaults.auto_adjust', True),
                    back_adjust=self.config.get('yfinance_api.defaults.back_adjust', False)
                )
            
            response_time = time.time() - start_time
            
            # Validate and clean data
            if not self.validate_data(data):
                error = f"Invalid data received for {ticker_symbol}"
                self.logger.error(error, extra={
                    "symbol": symbol,
                    "ticker": ticker_symbol,
                    "data_shape": data.shape if data is not None else None
                })
                return self._create_error_result(error, symbol)
            
            # Clean and process data
            processed_data = self._process_price_data(data, symbol, ticker_symbol)
            
            # Log successful collection
            log_data_collection(
                data_type="historical_prices",
                symbol=symbol,
                records_count=len(processed_data),
                success=True
            )
            
            self.logger.info(f"Successfully collected data for {symbol}", extra={
                "symbol": symbol,
                "ticker": ticker_symbol,
                "records": len(processed_data),
                "period": period,
                "interval": interval,
                "response_time": response_time
            })
            
            return self._create_success_result(processed_data, response_time)
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"Failed to collect data for {symbol}: {str(e)}"
            
            self.logger.error(error_msg, extra={
                "symbol": symbol,
                "ticker": ticker_symbol,
                "error": str(e),
                "response_time": response_time
            })
            
            log_data_collection(
                data_type="historical_prices",
                symbol=symbol,
                records_count=0,
                success=False,
                error=error_msg
            )
            
            return self._create_error_result(error_msg, symbol)
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a cryptocurrency.
        
        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH")
            
        Returns:
            Current price or None if failed
        """
        try:
            # Get ticker
            ticker_symbol = self.config.get_yfinance_ticker(symbol)
            if not ticker_symbol:
                return None
            
            ticker = yf.Ticker(ticker_symbol)
            
            # Get current data (last 1 day)
            data = ticker.history(period="1d", interval="1m")
            
            if data.empty:
                return None
            
            # Return latest close price
            return float(data['Close'].iloc[-1])
            
        except Exception as e:
            self.logger.error(f"Failed to get current price for {symbol}: {e}")
            return None
    
    def get_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a cryptocurrency.
        
        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH")
            
        Returns:
            Info dictionary or None if failed
        """
        try:
            ticker_symbol = self.config.get_yfinance_ticker(symbol)
            if not ticker_symbol:
                return None
            
            ticker = yf.Ticker(ticker_symbol)
            return ticker.info
            
        except Exception as e:
            self.logger.error(f"Failed to get info for {symbol}: {e}")
            return None
    
    def collect_multiple_symbols(self, symbols: List[str], **kwargs) -> Dict[str, CollectionResult]:
        """
        Collect data for multiple symbols efficiently.
        
        Args:
            symbols: List of crypto symbols
            **kwargs: Parameters for data collection
            
        Returns:
            Dictionary mapping symbols to collection results
        """
        results = {}
        
        # Get valid symbols and their tickers
        valid_symbols = []
        ticker_mapping = {}
        
        for symbol in symbols:
            if self._validate_symbol(symbol):
                ticker = self.config.get_yfinance_ticker(symbol)
                if ticker:
                    valid_symbols.append(symbol)
                    ticker_mapping[symbol] = ticker
        
        if not valid_symbols:
            self.logger.warning("No valid symbols provided for batch collection")
            return results
        
        try:
            # Use yfinance download for efficient batch collection
            tickers_str = " ".join(ticker_mapping.values())
            
            period = kwargs.get('period', '1y')
            interval = kwargs.get('interval', '1d')
            
            self.logger.info(f"Batch collecting data for {len(valid_symbols)} symbols", extra={
                "symbols": valid_symbols,
                "period": period,
                "interval": interval
            })
            
            # Batch download
            data = yf.download(
                tickers=tickers_str,
                period=period,
                interval=interval,
                group_by='ticker',
                auto_adjust=True,
                prepost=True,
                threads=True
            )
            
            # Process results for each symbol
            for symbol in valid_symbols:
                ticker = ticker_mapping[symbol]
                
                try:
                    if len(valid_symbols) == 1:
                        # Single symbol - data is directly the DataFrame
                        symbol_data = data
                    else:
                        # Multiple symbols - data is MultiIndex
                        symbol_data = data[ticker] if ticker in data.columns.levels[0] else pd.DataFrame()
                    
                    if self.validate_data(symbol_data):
                        processed_data = self._process_price_data(symbol_data, symbol, ticker)
                        results[symbol] = self._create_success_result(processed_data, 0.0)
                        
                        log_data_collection(
                            data_type="batch_historical_prices",
                            symbol=symbol,
                            records_count=len(processed_data),
                            success=True
                        )
                    else:
                        error = f"Invalid data for {symbol}"
                        results[symbol] = self._create_error_result(error, symbol)
                        
                except Exception as e:
                    error = f"Failed to process data for {symbol}: {str(e)}"
                    results[symbol] = self._create_error_result(error, symbol)
            
        except Exception as e:
            error_msg = f"Batch collection failed: {str(e)}"
            self.logger.error(error_msg, extra={"symbols": valid_symbols})
            
            # Return error results for all symbols
            for symbol in valid_symbols:
                results[symbol] = self._create_error_result(error_msg, symbol)
        
        return results
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate collected price data.
        
        Args:
            data: Price data to validate
            
        Returns:
            True if data is valid
        """
        return validate_dataframe_structure(data, self.required_price_columns)
    
    def _process_price_data(self, data: pd.DataFrame, symbol: str, ticker: str) -> pd.DataFrame:
        """
        Process and clean price data.
        
        Args:
            data: Raw price data from yfinance
            symbol: Original symbol
            ticker: yfinance ticker
            
        Returns:
            Processed DataFrame
        """
        # Clean numeric data
        processed = clean_numeric_data(data, self.numeric_price_columns)
        
        # Add metadata columns
        processed['Symbol'] = symbol
        processed['Ticker'] = ticker
        processed['Source'] = 'yfinance'
        processed['CollectionTimestamp'] = datetime.now()
        
        # Calculate additional metrics
        if 'Close' in processed.columns:
            # Daily returns
            processed['DailyReturn'] = processed['Close'].pct_change()
            
            # Simple moving averages
            processed['SMA_20'] = processed['Close'].rolling(window=20).mean()
            processed['SMA_50'] = processed['Close'].rolling(window=50).mean()
            
            # Volatility (20-day rolling)
            processed['Volatility_20d'] = processed['DailyReturn'].rolling(window=20).std() * np.sqrt(252)
        
        # Reset index to make datetime a column
        processed = processed.reset_index()
        if 'Datetime' in processed.columns:
            processed = processed.rename(columns={'Datetime': 'Timestamp'})
        elif 'Date' in processed.columns:
            processed = processed.rename(columns={'Date': 'Timestamp'})
        
        # Sort by timestamp
        if 'Timestamp' in processed.columns:
            processed = processed.sort_values('Timestamp')
        
        return processed
    
    def get_supported_intervals(self) -> List[str]:
        """Get list of supported data intervals."""
        return ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    
    def get_supported_periods(self) -> List[str]:
        """Get list of supported data periods."""
        return ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]


# Convenience functions
def get_crypto_data(symbol: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Convenience function to get crypto data.
    
    Args:
        symbol: Crypto symbol (e.g., "BTC", "ETH")
        period: Data period
        interval: Data interval
        
    Returns:
        DataFrame with price data or None if failed
    """
    collector = CryptoCollector()
    result = collector.collect_data(symbol, period=period, interval=interval)
    
    if result.success:
        return result.data
    else:
        return None


def get_current_crypto_prices(symbols: List[str]) -> Dict[str, float]:
    """
    Get current prices for multiple cryptocurrencies.
    
    Args:
        symbols: List of crypto symbols
        
    Returns:
        Dictionary mapping symbols to current prices
    """
    collector = CryptoCollector()
    prices = {}
    
    for symbol in symbols:
        price = collector.get_current_price(symbol)
        if price is not None:
            prices[symbol] = price
    
    return prices


if __name__ == "__main__":
    # Test the crypto collector
    print("🧪 Testing Crypto Data Collector")
    print("=" * 40)
    
    collector = CryptoCollector()
    
    # Test single symbol collection
    print("Testing BTC data collection...")
    result = collector.collect_data("BTC", period="5d", interval="1d")
    
    if result.success:
        print(f"✅ Successfully collected {result.records_count} records")
        print(f"📊 Data shape: {result.data.shape}")
        print(f"⏱️ Response time: {result.response_time:.2f}s")
        print(f"📋 Columns: {list(result.data.columns)}")
    else:
        print(f"❌ Collection failed: {result.error}")
    
    # Test current price
    print("\nTesting current price...")
    current_price = collector.get_current_price("BTC")
    if current_price:
        print(f"✅ Current BTC price: ${current_price:,.2f}")
    else:
        print("❌ Failed to get current price")
    
    # Test statistics
    print("\nCollector statistics:")
    stats = collector.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n🎉 Crypto collector test completed!")