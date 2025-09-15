# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Sample data population script for development and testing.
Generates realistic price data and portfolio information for the risk dashboard.
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging

# Add src to Python path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

# Also add project root to path
project_root = os.path.dirname(__file__) + '/..'
sys.path.insert(0, project_root)

from src.core.database.connection import get_database_async
from src.core.database.operations import DatabaseOperations

logger = logging.getLogger(__name__)

class SampleDataPopulator:
    """Populate database with realistic sample data for testing"""

    def __init__(self):
        self.assets = ['BTC', 'ETH', 'SOL', 'AVAX']
        self.db_ops = None

    async def initialize(self):
        """Initialize database operations"""
        self.db_ops = DatabaseOperations()
        logger.info("Sample data populator initialized")

    def generate_realistic_price_data(self,
                                    asset: str,
                                    days: int = 365,
                                    initial_price: float = None) -> List[Dict]:
        """Generate realistic price data with volatility and trends"""

        # Initial prices (approximate recent values)
        initial_prices = {
            'BTC': 45000.0,
            'ETH': 3200.0,
            'SOL': 120.0,
            'AVAX': 40.0
        }

        if initial_price is None:
            initial_price = initial_prices.get(asset, 100.0)

        # Asset-specific parameters
        params = {
            'BTC': {'daily_vol': 0.04, 'drift': 0.0008, 'jump_prob': 0.02},
            'ETH': {'daily_vol': 0.045, 'drift': 0.0010, 'jump_prob': 0.025},
            'SOL': {'daily_vol': 0.06, 'drift': 0.0012, 'jump_prob': 0.03},
            'AVAX': {'daily_vol': 0.065, 'drift': 0.0015, 'jump_prob': 0.035}
        }

        asset_params = params.get(asset, params['BTC'])

        # Generate price path
        np.random.seed(hash(asset) % 2**32)  # Consistent but different seed per asset

        dates = []
        prices = []
        current_price = initial_price

        # End at current date, start going backwards
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        for i in range(days):
            date = start_date + timedelta(days=i)

            # Generate daily return
            # Normal market movement
            daily_return = np.random.normal(
                asset_params['drift'],
                asset_params['daily_vol']
            )

            # Add occasional jumps (crashes or pumps)
            if np.random.random() < asset_params['jump_prob']:
                jump_size = np.random.normal(0, 0.15)  # 15% average jump
                daily_return += jump_size
                logger.debug(f"{asset} jump on {date.date()}: {jump_size:.2%}")

            # Update price
            current_price *= (1 + daily_return)
            current_price = max(current_price, initial_price * 0.1)  # Floor at 10% of initial

            dates.append(date)
            prices.append(current_price)

        # Create price data documents
        price_data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Add some intraday variation (high/low/volume)
            daily_range = price * np.random.uniform(0.02, 0.08)
            high = price + daily_range * np.random.uniform(0.3, 0.7)
            low = price - daily_range * np.random.uniform(0.3, 0.7)
            volume = np.random.lognormal(15, 1.5)  # Realistic volume distribution

            price_data.append({
                'symbol': asset,
                'price_usd': float(price),
                'high_24h': float(high),
                'low_24h': float(low),
                'volume_24h': float(volume),
                'market_cap': float(price * np.random.uniform(1e9, 1e12)),  # Mock market cap
                'timestamp': date,
                'source': 'sample_data'
            })

        logger.info(f"Generated {len(price_data)} price points for {asset}")
        return price_data

    async def populate_price_data(self, days: int = 365):
        """Populate database with sample price data"""
        logger.info(f"Populating price data for {len(self.assets)} assets over {days} days")

        for asset in self.assets:
            logger.info(f"Generating price data for {asset}")

            # Generate price data
            price_data = self.generate_realistic_price_data(asset, days)

            # Store in database
            try:
                db = await get_database_async()

                # Clear existing sample data for this asset
                await db.price_data.delete_many({
                    'symbol': asset,
                    'source': 'sample_data'
                })

                # Insert new data in batches
                batch_size = 100
                for i in range(0, len(price_data), batch_size):
                    batch = price_data[i:i + batch_size]
                    await db.price_data.insert_many(batch)

                logger.info(f"✅ Inserted {len(price_data)} price records for {asset}")

            except Exception as e:
                logger.error(f"❌ Error inserting price data for {asset}: {e}")

    async def populate_portfolio_data(self):
        """Create sample portfolio data"""
        logger.info("Creating sample portfolio data")

        sample_portfolios = [
            {
                'portfolio_id': 'sample_portfolio_1',
                'user_id': 'demo_user',
                'name': 'Balanced Crypto Portfolio',
                'description': 'A balanced cryptocurrency portfolio for testing',
                'assets': self.assets,
                'weights': {'BTC': 0.4, 'ETH': 0.3, 'SOL': 0.2, 'AVAX': 0.1},
                'total_value': 100000.0,
                'cash_position': 5000.0,
                'currency': 'USD',
                'created_at': datetime.utcnow() - timedelta(days=30),
                'last_updated': datetime.utcnow()
            },
            {
                'portfolio_id': 'sample_portfolio_2',
                'user_id': 'demo_user',
                'name': 'Growth Portfolio',
                'description': 'High growth potential altcoin portfolio',
                'assets': self.assets,
                'weights': {'BTC': 0.2, 'ETH': 0.3, 'SOL': 0.3, 'AVAX': 0.2},
                'total_value': 50000.0,
                'cash_position': 2500.0,
                'currency': 'USD',
                'created_at': datetime.utcnow() - timedelta(days=15),
                'last_updated': datetime.utcnow()
            }
        ]

        try:
            db = await get_database_async()

            # Clear existing sample portfolios
            await db.portfolio_data.delete_many({'user_id': 'demo_user'})

            # Insert sample portfolios
            for portfolio in sample_portfolios:
                portfolio['timestamp'] = datetime.utcnow()
                await db.portfolio_data.insert_one(portfolio)
                logger.info(f"✅ Created portfolio: {portfolio['name']}")

        except Exception as e:
            logger.error(f"❌ Error creating portfolio data: {e}")

    async def populate_options_data(self):
        """Create sample options data for volatility surfaces"""
        logger.info("Creating sample options data")

        try:
            db = await get_database_async()

            # Clear existing sample options data
            await db.options_data.delete_many({'source': 'sample_data'})

            options_data = []

            for underlying in ['BTC', 'ETH']:
                # Get current price
                current_price_doc = await db.price_data.find_one(
                    {'symbol': underlying},
                    sort=[('timestamp', -1)]
                )

                if not current_price_doc:
                    continue

                current_price = current_price_doc['price_usd']

                # Generate options chain
                expiry_dates = [
                    datetime.utcnow() + timedelta(days=7),   # Weekly
                    datetime.utcnow() + timedelta(days=30),  # Monthly
                    datetime.utcnow() + timedelta(days=90),  # Quarterly
                ]

                for expiry in expiry_dates:
                    # Strike prices around current price
                    strikes = np.arange(
                        current_price * 0.8,
                        current_price * 1.2,
                        current_price * 0.05
                    )

                    for strike in strikes:
                        for option_type in ['call', 'put']:
                            # Simple Black-Scholes-like pricing for sample data
                            moneyness = strike / current_price
                            time_to_expiry = (expiry - datetime.utcnow()).days / 365.0

                            # Mock implied volatility (higher for OTM options)
                            base_iv = 0.8 if underlying == 'BTC' else 1.0
                            iv = base_iv * (1 + abs(moneyness - 1) * 2) * np.sqrt(time_to_expiry)

                            # Mock option price
                            intrinsic = max(0, current_price - strike if option_type == 'call'
                                          else max(0, strike - current_price))
                            time_value = current_price * 0.1 * iv * np.sqrt(time_to_expiry)
                            option_price = intrinsic + time_value

                            options_data.append({
                                'underlying': underlying,
                                'symbol': f"{underlying}-{expiry.strftime('%d%b%y')}-{int(strike)}-{option_type[0].upper()}",
                                'option_type': option_type,
                                'strike': float(strike),
                                'expiry': expiry,
                                'price': float(option_price),
                                'implied_volatility': float(iv),
                                'delta': float(0.5),  # Simplified
                                'gamma': float(0.01),
                                'theta': float(-0.05),
                                'vega': float(0.1),
                                'volume': int(np.random.exponential(100)),
                                'open_interest': int(np.random.exponential(500)),
                                'timestamp': datetime.utcnow(),
                                'source': 'sample_data'
                            })

            # Insert options data in batches
            if options_data:
                batch_size = 100
                total_inserted = 0
                for i in range(0, len(options_data), batch_size):
                    batch = options_data[i:i + batch_size]
                    await db.options_data.insert_many(batch)
                    total_inserted += len(batch)

                logger.info(f"✅ Inserted {total_inserted} options records")

        except Exception as e:
            logger.error(f"❌ Error creating options data: {e}")

    async def run_full_population(self, days: int = 365):
        """Run complete data population"""
        logger.info("🚀 Starting complete sample data population")

        await self.initialize()

        # Populate price data (this is the most important for risk calculations)
        await self.populate_price_data(days)

        # Populate portfolio data
        await self.populate_portfolio_data()

        # Populate options data
        await self.populate_options_data()

        logger.info("✅ Sample data population completed successfully!")

        # Print summary
        try:
            db = await get_database_async()

            for asset in self.assets:
                count = await db.price_data.count_documents({'symbol': asset})
                logger.info(f"📊 {asset}: {count} price records")

            portfolio_count = await db.portfolio_data.count_documents({'user_id': 'demo_user'})
            logger.info(f"📋 Created {portfolio_count} sample portfolios")

            options_count = await db.options_data.count_documents({'source': 'sample_data'})
            logger.info(f"📈 Created {options_count} options records")

        except Exception as e:
            logger.error(f"Error getting summary: {e}")

async def main():
    """Main function to run data population"""
    import argparse

    parser = argparse.ArgumentParser(description="Populate sample data for Qortfolio")
    parser.add_argument('--days', type=int, default=365,
                       help='Number of days of price history to generate (default: 365)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    try:
        populator = SampleDataPopulator()
        await populator.run_full_population(args.days)
        print("\n🎉 Sample data population completed!")
        print("🔗 You can now test the risk dashboard with realistic data")

    except Exception as e:
        logger.error(f"❌ Data population failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())