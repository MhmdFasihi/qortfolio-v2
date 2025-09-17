# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Yahoo Finance integration for real crypto price data.
Supports 24/7 crypto markets with 365-day trading calendar.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CryptoAsset:
    """Crypto asset configuration"""
    symbol: str
    yahoo_symbol: str
    name: str
    category: str = "crypto"

class YFinanceProvider:
    """
    Real-time Yahoo Finance integration for crypto price data.
    Handles 24/7 crypto markets with proper 365-day trading calendar.
    """

    def __init__(self):
        # Crypto asset mappings (Yahoo Finance symbols)
        self.crypto_assets = {
            'BTC': CryptoAsset('BTC', 'BTC-USD', 'Bitcoin'),
            'ETH': CryptoAsset('ETH', 'ETH-USD', 'Ethereum'),
            'SOL': CryptoAsset('SOL', 'SOL-USD', 'Solana'),
            'AVAX': CryptoAsset('AVAX', 'AVAX-USD', 'Avalanche'),
            'ADA': CryptoAsset('ADA', 'ADA-USD', 'Cardano'),
            'DOT': CryptoAsset('DOT', 'DOT-USD', 'Polkadot'),
            'LINK': CryptoAsset('LINK', 'LINK-USD', 'Chainlink'),
            'UNI': CryptoAsset('UNI', 'UNI-USD', 'Uniswap'),
            'AAVE': CryptoAsset('AAVE', 'AAVE-USD', 'Aave'),
            'MATIC': CryptoAsset('MATIC', 'MATIC-USD', 'Polygon'),
            'ATOM': CryptoAsset('ATOM', 'ATOM-USD', 'Cosmos'),
            'FTM': CryptoAsset('FTM', 'FTM-USD', 'Fantom'),
            'NEAR': CryptoAsset('NEAR', 'NEAR-USD', 'Near Protocol'),
            'LTC': CryptoAsset('LTC', 'LTC-USD', 'Litecoin'),
            'BCH': CryptoAsset('BCH', 'BCH-USD', 'Bitcoin Cash'),
            'XRP': CryptoAsset('XRP', 'XRP-USD', 'XRP'),
            'DOGE': CryptoAsset('DOGE', 'DOGE-USD', 'Dogecoin'),
            'USDT': CryptoAsset('USDT', 'USDT-USD', 'Tether USD'),
            # Traditional assets
            'GC': CryptoAsset('GC', 'GC=F', 'Gold Futures', 'commodity'),
            'SPY': CryptoAsset('SPY', 'SPY', 'SPDR S&P 500 ETF', 'equity'),
        }

        # 24/7 crypto trading - 365 days per year (excluding leap year adjustments)
        self.trading_days_per_year = 365

    def get_supported_assets(self) -> List[str]:
        """Get list of supported crypto assets"""
        return list(self.crypto_assets.keys())

    def _get_yahoo_symbol(self, symbol: str) -> str:
        """Get Yahoo Finance symbol for crypto asset"""
        # Handle symbols that are already in yahoo format
        if symbol in ['GC=F', 'SPY']:
            return symbol

        # Remove -USD suffix if present for lookup
        clean_symbol = symbol.replace('-USD', '')
        asset = self.crypto_assets.get(clean_symbol.upper())

        if asset:
            return asset.yahoo_symbol
        else:
            # For unknown symbols, return as-is if already formatted, otherwise add -USD
            return symbol if ('-USD' in symbol or '=F' in symbol) else f"{symbol.upper()}-USD"

    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """
        Get current price for a crypto asset

        Args:
            symbol: Crypto symbol (BTC, ETH, etc.)

        Returns:
            Dictionary with current price data or None if error
        """
        try:
            yahoo_symbol = self._get_yahoo_symbol(symbol)
            ticker = yf.Ticker(yahoo_symbol)

            # Get current info
            info = ticker.info

            if not info or 'regularMarketPrice' not in info:
                logger.warning(f"No current price data for {symbol}")
                return None

            result = {
                'symbol': symbol.upper(),
                'price_usd': float(info.get('regularMarketPrice', 0)),
                'currency': 'USD',
                'market_cap': info.get('marketCap'),
                'volume_24h': info.get('volume24Hr', info.get('regularMarketVolume', 0)),
                'change_24h': info.get('regularMarketChangePercent', 0) / 100 if info.get('regularMarketChangePercent') else 0,
                'high_24h': info.get('dayHigh', info.get('regularMarketDayHigh', 0)),
                'low_24h': info.get('dayLow', info.get('regularMarketDayLow', 0)),
                'timestamp': datetime.utcnow(),
                'source': 'yfinance'
            }

            logger.debug(f"Current price for {symbol}: ${result['price_usd']:,.2f}")
            return result

        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    def get_historical_data(self,
                          symbol: str,
                          period: str = "1y",
                          interval: str = "1d") -> pd.DataFrame:
        """
        Get historical price data for crypto asset

        Args:
            symbol: Crypto symbol (BTC, ETH, etc.)
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with historical OHLCV data
        """
        try:
            yahoo_symbol = self._get_yahoo_symbol(symbol)
            ticker = yf.Ticker(yahoo_symbol)

            # Download historical data
            hist_data = ticker.history(period=period, interval=interval)

            if hist_data.empty:
                logger.warning(f"No historical data for {symbol}")
                return pd.DataFrame()

            # Clean and format data
            hist_data = hist_data.reset_index()

            # Rename columns to match our schema
            column_mapping = {
                'Date': 'timestamp',
                'Datetime': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }

            hist_data = hist_data.rename(columns=column_mapping)

            # Add metadata columns
            hist_data['symbol'] = symbol.upper()
            hist_data['price_usd'] = hist_data['close']
            hist_data['source'] = 'yfinance'

            # Handle timezone issues
            if 'timestamp' in hist_data.columns:
                hist_data['timestamp'] = pd.to_datetime(hist_data['timestamp'])
                if hist_data['timestamp'].dt.tz is not None:
                    hist_data['timestamp'] = hist_data['timestamp'].dt.tz_localize(None)

            logger.info(f"Retrieved {len(hist_data)} data points for {symbol} ({period}, {interval})")
            return hist_data

        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()

    def get_multiple_current_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get current prices for multiple crypto assets

        Args:
            symbols: List of crypto symbols

        Returns:
            Dictionary mapping symbols to price data
        """
        prices = {}

        for symbol in symbols:
            price_data = self.get_current_price(symbol)
            if price_data:
                prices[symbol.upper()] = price_data

        logger.info(f"Retrieved current prices for {len(prices)}/{len(symbols)} assets")
        return prices

    def get_multiple_historical_data(self,
                                   symbols: List[str],
                                   period: str = "1y",
                                   interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple crypto assets

        Args:
            symbols: List of crypto symbols
            period: Time period for all assets
            interval: Data interval for all assets

        Returns:
            Dictionary mapping symbols to historical DataFrames
        """
        historical_data = {}

        for symbol in symbols:
            data = self.get_historical_data(symbol, period, interval)
            if not data.empty:
                historical_data[symbol.upper()] = data

        logger.info(f"Retrieved historical data for {len(historical_data)}/{len(symbols)} assets")
        return historical_data

    def calculate_returns(self,
                         historical_data: pd.DataFrame,
                         return_type: str = "simple") -> pd.Series:
        """
        Calculate returns from historical price data

        Args:
            historical_data: DataFrame with historical prices
            return_type: Type of returns ("simple" or "log")

        Returns:
            Series with calculated returns
        """
        try:
            if 'price_usd' in historical_data.columns:
                prices = historical_data['price_usd']
            elif 'close' in historical_data.columns:
                prices = historical_data['close']
            else:
                logger.error("No price column found in historical data")
                return pd.Series()

            if return_type == "log":
                returns = np.log(prices / prices.shift(1)).dropna()
            else:  # simple returns
                returns = prices.pct_change().dropna()

            return returns

        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return pd.Series()

    def get_crypto_market_summary(self) -> Dict[str, Any]:
        """
        Get market summary for all supported crypto assets

        Returns:
            Dictionary with market summary data
        """
        try:
            logger.info("Fetching crypto market summary...")

            current_prices = self.get_multiple_current_prices(list(self.crypto_assets.keys()))

            market_summary = {
                'timestamp': datetime.utcnow(),
                'total_assets': len(self.crypto_assets),
                'assets_with_data': len(current_prices),
                'market_data': current_prices,
                'trading_calendar': {
                    'trading_days_per_year': self.trading_days_per_year,
                    'market_hours': '24/7',
                    'market_type': 'crypto'
                }
            }

            # Calculate aggregate metrics if we have price data
            if current_prices:
                total_market_cap = sum(
                    asset_data.get('market_cap', 0) or 0
                    for asset_data in current_prices.values()
                )

                market_summary['aggregate_metrics'] = {
                    'total_market_cap': total_market_cap,
                    'avg_24h_change': np.mean([
                        asset_data.get('change_24h', 0)
                        for asset_data in current_prices.values()
                    ])
                }

            logger.info(f"Market summary: {market_summary['assets_with_data']} assets with live data")
            return market_summary

        except Exception as e:
            logger.error(f"Error getting market summary: {e}")
            return {}

    def get_returns_matrix(self,
                          symbols: List[str],
                          period: str = "1y",
                          interval: str = "1d") -> pd.DataFrame:
        """
        Get returns matrix for multiple assets (for correlation analysis)

        Args:
            symbols: List of crypto symbols
            period: Time period
            interval: Data interval

        Returns:
            DataFrame with returns for all assets (assets as columns)
        """
        try:
            historical_data = self.get_multiple_historical_data(symbols, period, interval)

            returns_dict = {}

            for symbol, data in historical_data.items():
                returns = self.calculate_returns(data)
                if not returns.empty:
                    returns.index = data['timestamp'].iloc[1:]  # Align with returns
                    returns_dict[symbol] = returns

            if not returns_dict:
                return pd.DataFrame()

            # Combine into matrix
            returns_df = pd.DataFrame(returns_dict)
            returns_df = returns_df.dropna()  # Remove rows with missing data

            logger.info(f"Returns matrix: {returns_df.shape[0]} days, {returns_df.shape[1]} assets")
            return returns_df

        except Exception as e:
            logger.error(f"Error creating returns matrix: {e}")
            return pd.DataFrame()

# Convenience functions for easy usage
def get_btc_price() -> Optional[float]:
    """Get current BTC price"""
    provider = YFinanceProvider()
    price_data = provider.get_current_price('BTC')
    return price_data['price_usd'] if price_data else None

def get_crypto_prices(symbols: List[str]) -> Dict[str, float]:
    """Get current prices for multiple cryptos"""
    provider = YFinanceProvider()
    price_data = provider.get_multiple_current_prices(symbols)
    return {symbol: data['price_usd'] for symbol, data in price_data.items()}

def get_crypto_returns_matrix(symbols: List[str], days: int = 365) -> pd.DataFrame:
    """Get returns matrix for crypto assets"""
    provider = YFinanceProvider()
    period = f"{days}d" if days <= 730 else "max"
    return provider.get_returns_matrix(symbols, period=period)

if __name__ == "__main__":
    # Example usage
    print("📊 Testing Yahoo Finance crypto data integration...")

    provider = YFinanceProvider()

    # Test current prices
    print("\n💰 Current Crypto Prices:")
    test_symbols = ['BTC', 'ETH', 'SOL', 'AVAX']
    current_prices = provider.get_multiple_current_prices(test_symbols)

    for symbol, data in current_prices.items():
        price = data['price_usd']
        change = data['change_24h'] * 100
        print(f"  {symbol}: ${price:,.2f} ({change:+.2f}%)")

    # Test historical data
    print(f"\n📈 Historical Data Test:")
    btc_data = provider.get_historical_data('BTC', period='30d', interval='1d')

    if not btc_data.empty:
        print(f"  Retrieved {len(btc_data)} days of BTC data")
        print(f"  Date range: {btc_data['timestamp'].min()} to {btc_data['timestamp'].max()}")

        # Calculate returns
        btc_returns = provider.calculate_returns(btc_data)
        print(f"  BTC 30d volatility: {btc_returns.std() * np.sqrt(365):.1%}")

    # Test returns matrix
    print(f"\n📊 Returns Matrix Test:")
    returns_matrix = provider.get_returns_matrix(test_symbols, period='90d')

    if not returns_matrix.empty:
        print(f"  Returns matrix shape: {returns_matrix.shape}")
        print(f"  Correlation matrix:")
        corr_matrix = returns_matrix.corr()
        print(corr_matrix.round(3))

    # Market summary
    print(f"\n🌍 Crypto Market Summary:")
    market_summary = provider.get_crypto_market_summary()

    if market_summary:
        print(f"  Assets tracked: {market_summary['total_assets']}")
        print(f"  Assets with live data: {market_summary['assets_with_data']}")
        print(f"  Trading days per year: {market_summary['trading_calendar']['trading_days_per_year']}")

        if 'aggregate_metrics' in market_summary:
            total_mcap = market_summary['aggregate_metrics']['total_market_cap']
            avg_change = market_summary['aggregate_metrics']['avg_24h_change']
            print(f"  Total market cap: ${total_mcap:,.0f}")
            print(f"  Average 24h change: {avg_change:.2%}")

    print(f"\n🎉 Yahoo Finance integration test completed!")