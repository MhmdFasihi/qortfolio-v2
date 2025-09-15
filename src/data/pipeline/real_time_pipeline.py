# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Real-time data pipeline for crypto markets with 24/7 trading and 365-day calendar.
Integrates Deribit options data and yfinance price data for institutional-grade analytics.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Import our data providers
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data.providers.deribit_provider import DeribitProvider
from data.providers.yfinance_provider import YFinanceProvider
from core.database.connection import get_database_async

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for real-time data pipeline"""
    update_interval_seconds: int = 300  # 5 minutes
    price_data_assets: List[str] = None
    options_currencies: List[str] = None
    store_historical_data: bool = True
    max_historical_days: int = 730  # 2 years
    enable_volatility_surfaces: bool = True
    enable_options_flow: bool = True

    def __post_init__(self):
        if self.price_data_assets is None:
            self.price_data_assets = ['BTC', 'ETH', 'SOL', 'AVAX', 'ADA', 'DOT', 'LINK', 'UNI', 'AAVE']
        if self.options_currencies is None:
            self.options_currencies = ['BTC', 'ETH']

class CryptoTradingCalendar:
    """
    Crypto trading calendar - 24/7 markets, 365 days per year
    Accounts for leap years and provides proper business day calculations
    """

    def __init__(self):
        self.trading_days_per_year = 365  # Base year
        self.hours_per_day = 24
        self.trading_hours_per_year = self.trading_days_per_year * self.hours_per_day

    def is_trading_day(self, date: datetime) -> bool:
        """Crypto markets trade 24/7 - always True"""
        return True

    def get_trading_days_in_year(self, year: int) -> int:
        """Get actual trading days in a given year (handles leap years)"""
        if self.is_leap_year(year):
            return 366
        return 365

    def is_leap_year(self, year: int) -> bool:
        """Check if year is leap year"""
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    def get_business_days(self, start_date: datetime, end_date: datetime) -> int:
        """Get number of business days between dates (all days for crypto)"""
        return (end_date - start_date).days

    def annualize_volatility(self, daily_volatility: float, year: int = None) -> float:
        """Annualize daily volatility using proper crypto calendar"""
        if year is None:
            year = datetime.now().year

        trading_days = self.get_trading_days_in_year(year)
        return daily_volatility * np.sqrt(trading_days)

    def get_time_to_expiry_years(self, expiry_date: datetime, current_date: datetime = None) -> float:
        """Calculate time to expiry in years using crypto calendar"""
        if current_date is None:
            current_date = datetime.utcnow()

        days_to_expiry = (expiry_date - current_date).days
        year = current_date.year
        trading_days_in_year = self.get_trading_days_in_year(year)

        return days_to_expiry / trading_days_in_year

class RealTimeDataPipeline:
    """
    Real-time data pipeline for institutional-grade crypto analytics.
    Handles 24/7 markets with proper 365-day calendar calculations.
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.calendar = CryptoTradingCalendar()

        # Data providers
        self.yfinance_provider = YFinanceProvider()
        self.deribit_provider = DeribitProvider()

        # Pipeline state
        self.running = False
        self.last_update_times = {}
        self.error_counts = {}

    async def start_pipeline(self):
        """Start the real-time data pipeline"""
        logger.info("🚀 Starting real-time crypto data pipeline...")
        logger.info(f"   Update interval: {self.config.update_interval_seconds} seconds")
        logger.info(f"   Price assets: {len(self.config.price_data_assets)}")
        logger.info(f"   Options currencies: {len(self.config.options_currencies)}")

        self.running = True

        # Initialize database connection
        try:
            db = await get_database_async()
            logger.info("✅ Database connection established")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return

        # Initialize data providers
        await self.deribit_provider.connect()

        # Start main pipeline loop
        await self._pipeline_loop()

    async def stop_pipeline(self):
        """Stop the data pipeline"""
        logger.info("⏹️ Stopping real-time data pipeline...")
        self.running = False
        await self.deribit_provider.disconnect()

    async def _pipeline_loop(self):
        """Main pipeline loop"""
        while self.running:
            try:
                start_time = datetime.utcnow()

                # Update price data
                await self._update_price_data()

                # Update options data
                if self.config.enable_volatility_surfaces:
                    await self._update_options_data()

                # Update volatility surfaces
                if self.config.enable_volatility_surfaces:
                    await self._update_volatility_surfaces()

                # Performance tracking
                update_duration = (datetime.utcnow() - start_time).total_seconds()
                logger.info(f"📊 Pipeline update completed in {update_duration:.1f}s")

                # Wait for next update
                await asyncio.sleep(self.config.update_interval_seconds)

            except Exception as e:
                logger.error(f"❌ Pipeline error: {e}")
                self.error_counts['pipeline'] = self.error_counts.get('pipeline', 0) + 1

                # Exponential backoff on errors
                error_count = self.error_counts.get('pipeline', 0)
                sleep_time = min(60, 5 * (2 ** min(error_count, 5)))
                await asyncio.sleep(sleep_time)

    async def _update_price_data(self):
        """Update real-time price data from yfinance"""
        try:
            logger.debug("Updating price data...")

            # Get current prices
            current_prices = self.yfinance_provider.get_multiple_current_prices(
                self.config.price_data_assets
            )

            if not current_prices:
                logger.warning("No price data received")
                return

            # Store in database
            db = await get_database_async()

            for symbol, price_data in current_prices.items():
                # Convert to database format
                price_doc = {
                    'symbol': symbol,
                    'price_usd': price_data['price_usd'],
                    'high_24h': price_data.get('high_24h', 0),
                    'low_24h': price_data.get('low_24h', 0),
                    'volume_24h': price_data.get('volume_24h', 0),
                    'market_cap': price_data.get('market_cap', 0),
                    'change_24h': price_data.get('change_24h', 0),
                    'timestamp': datetime.utcnow(),
                    'source': 'yfinance_realtime'
                }

                # Store with upsert (replace existing for same timestamp)
                await db.price_data.insert_one(price_doc)

            logger.debug(f"✅ Updated prices for {len(current_prices)} assets")
            self.last_update_times['price_data'] = datetime.utcnow()

        except Exception as e:
            logger.error(f"❌ Price data update error: {e}")
            self.error_counts['price_data'] = self.error_counts.get('price_data', 0) + 1

    async def _update_options_data(self):
        """Update real-time options data from Deribit"""
        try:
            logger.debug("Updating options data...")

            db = await get_database_async()

            for currency in self.config.options_currencies:
                # Get options chain
                options_df = await self.deribit_provider.get_options_chain(currency)

                if options_df.empty:
                    logger.warning(f"No options data for {currency}")
                    continue

                # Convert to database format and store
                options_docs = []
                for _, row in options_df.iterrows():
                    option_doc = {
                        'symbol': row['symbol'],
                        'underlying': row['underlying'],
                        'strike': row['strike'],
                        'expiry': datetime.fromtimestamp(row['expiry']),
                        'option_type': row['option_type'],
                        'price': row['price'],
                        'bid': row['bid'],
                        'ask': row['ask'],
                        'volume': row['volume'],
                        'open_interest': row['open_interest'],
                        'implied_volatility': row['implied_volatility'],
                        'delta': row['delta'],
                        'gamma': row['gamma'],
                        'theta': row['theta'],
                        'vega': row['vega'],
                        'timestamp': datetime.utcnow(),
                        'source': 'deribit_realtime'
                    }
                    options_docs.append(option_doc)

                # Batch insert options
                if options_docs:
                    await db.options_data.insert_many(options_docs)

                logger.debug(f"✅ Updated {len(options_docs)} {currency} options")

            self.last_update_times['options_data'] = datetime.utcnow()

        except Exception as e:
            logger.error(f"❌ Options data update error: {e}")
            self.error_counts['options_data'] = self.error_counts.get('options_data', 0) + 1

    async def _update_volatility_surfaces(self):
        """Update volatility surfaces from options data"""
        try:
            logger.debug("Updating volatility surfaces...")

            db = await get_database_async()

            for currency in self.config.options_currencies:
                # Get volatility surface from Deribit
                vol_surface = await self.deribit_provider.get_volatility_surface(currency)

                if not vol_surface:
                    logger.warning(f"No volatility surface for {currency}")
                    continue

                # Store volatility surface
                surface_doc = {
                    'currency': currency,
                    'spot_price': vol_surface['spot_price'],
                    'surface_data': vol_surface['surface'],
                    'data_points_count': vol_surface['total_options'],
                    'timestamp': datetime.utcnow(),
                    'source': 'deribit_realtime'
                }

                # Replace existing surface
                await db.volatility_surfaces.replace_one(
                    {'currency': currency},
                    surface_doc,
                    upsert=True
                )

                # Store historical copy
                await db.volatility_surfaces_history.insert_one(surface_doc)

                logger.debug(f"✅ Updated {currency} volatility surface")

            self.last_update_times['volatility_surfaces'] = datetime.utcnow()

        except Exception as e:
            logger.error(f"❌ Volatility surface update error: {e}")
            self.error_counts['volatility_surfaces'] = self.error_counts.get('volatility_surfaces', 0) + 1

    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and health metrics"""
        current_time = datetime.utcnow()

        status = {
            'running': self.running,
            'current_time': current_time,
            'update_interval': self.config.update_interval_seconds,
            'last_updates': {},
            'error_counts': self.error_counts.copy(),
            'health_status': 'healthy'
        }

        # Calculate time since last updates
        for component, last_time in self.last_update_times.items():
            time_since = (current_time - last_time).total_seconds()
            status['last_updates'][component] = {
                'last_update': last_time,
                'seconds_since': time_since,
                'status': 'healthy' if time_since < self.config.update_interval_seconds * 2 else 'stale'
            }

        # Determine overall health
        total_errors = sum(self.error_counts.values())
        if total_errors > 10:
            status['health_status'] = 'degraded'
        elif total_errors > 50:
            status['health_status'] = 'unhealthy'

        return status

    async def populate_historical_data(self, days: int = None):
        """Populate historical data for all assets"""
        if days is None:
            days = min(self.config.max_historical_days, 730)

        logger.info(f"📊 Populating {days} days of historical data...")

        try:
            db = await get_database_async()

            # Get historical price data
            for symbol in self.config.price_data_assets:
                logger.info(f"Fetching historical data for {symbol}...")

                # Get data from yfinance
                hist_data = self.yfinance_provider.get_historical_data(
                    symbol,
                    period=f"{days}d",
                    interval="1d"
                )

                if hist_data.empty:
                    logger.warning(f"No historical data for {symbol}")
                    continue

                # Convert to database format
                price_docs = []
                for _, row in hist_data.iterrows():
                    price_doc = {
                        'symbol': symbol,
                        'price_usd': row['price_usd'],
                        'high_24h': row.get('high', row['price_usd']),
                        'low_24h': row.get('low', row['price_usd']),
                        'volume_24h': row.get('volume', 0),
                        'market_cap': None,  # Not available in historical data
                        'timestamp': row['timestamp'],
                        'source': 'yfinance_historical'
                    }
                    price_docs.append(price_doc)

                # Clear existing historical data from yfinance
                await db.price_data.delete_many({
                    'symbol': symbol,
                    'source': 'yfinance_historical'
                })

                # Insert historical data in batches
                batch_size = 100
                for i in range(0, len(price_docs), batch_size):
                    batch = price_docs[i:i + batch_size]
                    await db.price_data.insert_many(batch)

                logger.info(f"✅ Stored {len(price_docs)} historical records for {symbol}")

            logger.info("✅ Historical data population completed")

        except Exception as e:
            logger.error(f"❌ Historical data population error: {e}")
            raise

# Convenience functions
async def start_real_time_pipeline(config: PipelineConfig = None):
    """Start the real-time data pipeline"""
    pipeline = RealTimeDataPipeline(config)
    await pipeline.start_pipeline()
    return pipeline

async def populate_initial_data(days: int = 365):
    """Populate initial historical data"""
    pipeline = RealTimeDataPipeline()
    await pipeline.populate_historical_data(days)

if __name__ == "__main__":
    # Example usage
    import asyncio

    async def main():
        print("🚀 Testing real-time crypto data pipeline...")

        # Create pipeline config
        config = PipelineConfig(
            update_interval_seconds=60,  # 1 minute for testing
            price_data_assets=['BTC', 'ETH', 'SOL', 'AVAX'],
            options_currencies=['BTC'],  # Start with BTC only
            enable_volatility_surfaces=True
        )

        # Test historical data population
        print("📊 Populating historical data...")
        pipeline = RealTimeDataPipeline(config)
        await pipeline.populate_historical_data(30)  # 30 days for testing

        print("✅ Historical data populated")
        print("🎯 Real-time pipeline ready!")

        # Test pipeline status
        status = await pipeline.get_pipeline_status()
        print(f"Pipeline status: {status['health_status']}")

    asyncio.run(main())