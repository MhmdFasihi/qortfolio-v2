# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Populate database with real market data from yfinance.
Replaces sample data with actual crypto market data for production use.
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import yfinance as yf
import pandas as pd

# Import from the correct location
try:
    from src.core.database.connection import get_database_async
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.core.database.connection import get_database_async

logger = logging.getLogger(__name__)

class RealMarketDataPopulator:
    """Populate database with real market data"""

    def __init__(self):
        self.crypto_assets = {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD',
            'SOL': 'SOL-USD',
            'AVAX': 'AVAX-USD',
            'ADA': 'ADA-USD',
            'DOT': 'DOT-USD',
            'LINK': 'LINK-USD',
            'UNI': 'UNI-USD',
            'AAVE': 'AAVE-USD'
        }

    async def populate_real_price_data(self, days: int = 365):
        """Populate with real price data from yfinance"""
        logger.info(f"🚀 Populating {days} days of REAL market data...")

        try:
            db = await get_database_async()

            for symbol, yahoo_symbol in self.crypto_assets.items():
                print(f"📊 Fetching real data for {symbol}...")

                # Get real data from Yahoo Finance
                ticker = yf.Ticker(yahoo_symbol)
                hist_data = ticker.history(period=f"{days}d", interval="1d")

                if hist_data.empty:
                    print(f"⚠️  No data available for {symbol}")
                    continue

                # Prepare data for database
                price_docs = []
                for date, row in hist_data.iterrows():
                    price_doc = {
                        'symbol': symbol,
                        'price_usd': float(row['Close']),
                        'high_24h': float(row['High']),
                        'low_24h': float(row['Low']),
                        'volume_24h': float(row['Volume']),
                        'market_cap': None,  # Would need additional API call
                        'timestamp': date.tz_localize(None) if date.tz else date,
                        'source': 'yfinance_real'
                    }
                    price_docs.append(price_doc)

                # Clear existing sample/test data
                await db.price_data.delete_many({
                    'symbol': symbol,
                    'source': {'$in': ['sample_data', 'yfinance_historical']}
                })

                # Insert real market data
                if price_docs:
                    batch_size = 100
                    for i in range(0, len(price_docs), batch_size):
                        batch = price_docs[i:i + batch_size]
                        await db.price_data.insert_many(batch)

                print(f"✅ Inserted {len(price_docs)} real market records for {symbol}")

            print(f"🎉 Real market data population completed!")
            return True

        except Exception as e:
            print(f"❌ Error populating real data: {e}")
            return False

    async def populate_real_portfolios(self):
        """Create portfolios with real market data"""
        try:
            db = await get_database_async()

            # Get current real prices for portfolio values
            current_prices = {}
            for symbol, yahoo_symbol in self.crypto_assets.items():
                try:
                    ticker = yf.Ticker(yahoo_symbol)
                    current_price = ticker.history(period="1d")['Close'].iloc[-1]
                    current_prices[symbol] = float(current_price)
                except:
                    current_prices[symbol] = 0

            real_portfolios = [
                {
                    'portfolio_id': 'real_balanced_portfolio',
                    'user_id': 'real_user',
                    'name': 'Balanced Crypto Portfolio (Real Data)',
                    'description': 'Balanced crypto portfolio using real market data',
                    'assets': ['BTC', 'ETH', 'SOL', 'AVAX'],
                    'weights': {'BTC': 0.4, 'ETH': 0.3, 'SOL': 0.2, 'AVAX': 0.1},
                    'total_value': sum([
                        current_prices.get('BTC', 0) * 0.4,
                        current_prices.get('ETH', 0) * 0.3,
                        current_prices.get('SOL', 0) * 0.2,
                        current_prices.get('AVAX', 0) * 0.1
                    ]) * 100,  # Scale to reasonable portfolio size
                    'cash_position': 5000.0,
                    'currency': 'USD',
                    'timestamp': datetime.utcnow(),
                    'data_source': 'real_market'
                },
                {
                    'portfolio_id': 'real_defi_portfolio',
                    'user_id': 'real_user',
                    'name': 'DeFi Portfolio (Real Data)',
                    'description': 'DeFi-focused portfolio with real market data',
                    'assets': ['ETH', 'UNI', 'AAVE', 'LINK'],
                    'weights': {'ETH': 0.4, 'UNI': 0.25, 'AAVE': 0.25, 'LINK': 0.1},
                    'total_value': sum([
                        current_prices.get('ETH', 0) * 0.4,
                        current_prices.get('UNI', 0) * 0.25,
                        current_prices.get('AAVE', 0) * 0.25,
                        current_prices.get('LINK', 0) * 0.1
                    ]) * 75,  # Scale to reasonable size
                    'cash_position': 2500.0,
                    'currency': 'USD',
                    'timestamp': datetime.utcnow(),
                    'data_source': 'real_market'
                }
            ]

            # Clear existing demo portfolios
            await db.portfolio_data.delete_many({'user_id': {'$in': ['demo_user', 'real_user']}})

            # Insert real portfolios
            for portfolio in real_portfolios:
                await db.portfolio_data.insert_one(portfolio)
                print(f"✅ Created portfolio: {portfolio['name']}")

            return True

        except Exception as e:
            print(f"❌ Error creating real portfolios: {e}")
            return False

    async def test_real_data_analytics(self):
        """Test that risk analytics work with real data"""
        try:
            from src.analytics.risk.portfolio_risk import PortfolioRiskAnalyzer

            db = await get_database_async()
            risk_analyzer = PortfolioRiskAnalyzer(db)

            # Test with real portfolio
            portfolio = await db.portfolio_data.find_one({'data_source': 'real_market'})
            if not portfolio:
                print("❌ No real portfolios found")
                return False

            portfolio_id = portfolio['portfolio_id']
            print(f"🧮 Testing risk analytics with real portfolio: {portfolio['name']}")

            # Calculate risk metrics with real data
            risk_metrics = await risk_analyzer.calculate_portfolio_metrics(
                portfolio_id=portfolio_id,
                lookback_days=90
            )

            if risk_metrics:
                print("🎉 SUCCESS! Risk analytics working with REAL market data!")
                print("📊 Real Market Risk Metrics:")

                key_metrics = [
                    ('Sharpe Ratio', 'sharpe_ratio'),
                    ('Sortino Ratio', 'sortino_ratio'),
                    ('Max Drawdown', 'max_drawdown'),
                    ('Portfolio Volatility', 'portfolio_volatility')
                ]

                for display_name, key in key_metrics:
                    if key in risk_metrics and isinstance(risk_metrics[key], (float, int)):
                        value = risk_metrics[key]
                        if 'ratio' in key.lower():
                            print(f"   {display_name}: {value:.3f}")
                        else:
                            print(f"   {display_name}: {value:.2%}")

                return True
            else:
                print("❌ Risk metrics calculation failed")
                return False

        except Exception as e:
            print(f"❌ Error testing analytics: {e}")
            return False

    async def run_full_real_data_setup(self, days: int = 365):
        """Complete setup with real market data"""
        print("🚀 SETTING UP REAL CRYPTO MARKET DATA")
        print("=" * 50)

        # Step 1: Populate real price data
        print("📊 Step 1: Populating real price data...")
        if not await self.populate_real_price_data(days):
            return False

        # Step 2: Create real portfolios
        print("\n📋 Step 2: Creating portfolios with real data...")
        if not await self.populate_real_portfolios():
            return False

        # Step 3: Test analytics with real data
        print("\n🧮 Step 3: Testing analytics with real market data...")
        if not await self.test_real_data_analytics():
            return False

        print("\n" + "=" * 50)
        print("🎉 REAL MARKET DATA SETUP COMPLETE!")
        print("✅ Dashboard now uses live crypto market data")
        print("✅ Risk analytics working with real prices")
        print("✅ 24/7 crypto market data (365-day calendar)")
        print("🚀 Production-ready institutional analytics!")

        return True

async def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    populator = RealMarketDataPopulator()
    success = await populator.run_full_real_data_setup(days=365)

    if success:
        print("\n🎯 Your risk dashboard now has REAL market data!")
    else:
        print("\n❌ Setup encountered issues - check logs above")

if __name__ == "__main__":
    asyncio.run(main())