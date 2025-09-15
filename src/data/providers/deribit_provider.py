# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Deribit API integration for real crypto options and derivatives data.
Provides live options chains, implied volatility, Greeks, and order book data.
"""

import asyncio
import aiohttp
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DeribitCredentials:
    """Deribit API credentials"""
    client_id: str = ""
    client_secret: str = ""
    testnet: bool = True  # Use testnet by default

class DeribitProvider:
    """
    Real-time Deribit API integration for crypto options and derivatives data.
    Supports live options chains, volatility surfaces, Greeks, and order book data.
    """

    def __init__(self, credentials: DeribitCredentials = None):
        self.credentials = credentials or DeribitCredentials()
        self.base_url = ("https://test.deribit.com/api/v2" if self.credentials.testnet
                        else "https://www.deribit.com/api/v2")
        self.session: Optional[aiohttp.ClientSession] = None
        self.access_token: Optional[str] = None

        # Supported instruments
        self.currencies = ['BTC', 'ETH']

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()

    async def connect(self):
        """Initialize connection to Deribit API"""
        self.session = aiohttp.ClientSession()

        # Authenticate if credentials provided
        if self.credentials.client_id and self.credentials.client_secret:
            await self._authenticate()

        logger.info(f"Connected to Deribit {'testnet' if self.credentials.testnet else 'mainnet'}")

    async def disconnect(self):
        """Close connection"""
        if self.session:
            await self.session.close()
            self.session = None

    async def _authenticate(self):
        """Authenticate with Deribit API"""
        try:
            data = {
                "jsonrpc": "2.0",
                "method": "public/auth",
                "params": {
                    "grant_type": "client_credentials",
                    "client_id": self.credentials.client_id,
                    "client_secret": self.credentials.client_secret
                },
                "id": 1
            }

            async with self.session.post(f"{self.base_url}", json=data) as response:
                result = await response.json()

                if "result" in result:
                    self.access_token = result["result"]["access_token"]
                    logger.info("✅ Deribit authentication successful")
                else:
                    logger.error(f"❌ Deribit authentication failed: {result}")

        except Exception as e:
            logger.error(f"❌ Deribit authentication error: {e}")

    async def _make_request(self, method: str, params: Dict = None) -> Optional[Dict]:
        """Make API request to Deribit"""
        try:
            data = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params or {},
                "id": 1
            }

            headers = {}
            if self.access_token:
                headers["Authorization"] = f"Bearer {self.access_token}"

            async with self.session.post(self.base_url, json=data, headers=headers) as response:
                result = await response.json()

                if "result" in result:
                    return result["result"]
                else:
                    logger.error(f"API error for {method}: {result}")
                    return None

        except Exception as e:
            logger.error(f"Request error for {method}: {e}")
            return None

    async def get_instruments(self, currency: str = "BTC", kind: str = "option") -> List[Dict]:
        """
        Get available instruments for a currency

        Args:
            currency: Currency (BTC, ETH)
            kind: Instrument kind (option, future, perpetual)

        Returns:
            List of instrument data
        """
        try:
            params = {
                "currency": currency,
                "kind": kind,
                "expired": False
            }

            result = await self._make_request("public/get_instruments", params)

            if result:
                logger.info(f"Retrieved {len(result)} {currency} {kind} instruments")
                return result

            return []

        except Exception as e:
            logger.error(f"Error getting instruments: {e}")
            return []

    async def get_options_chain(self, currency: str = "BTC") -> pd.DataFrame:
        """
        Get complete options chain for a currency

        Args:
            currency: Currency (BTC, ETH)

        Returns:
            DataFrame with options chain data
        """
        try:
            logger.info(f"Fetching {currency} options chain from Deribit...")

            instruments = await self.get_instruments(currency, "option")

            if not instruments:
                logger.warning(f"No {currency} options found")
                return pd.DataFrame()

            options_data = []

            # Get market data for each option
            for instrument in instruments:
                instrument_name = instrument["instrument_name"]

                # Get ticker data (price, volume, Greeks)
                ticker_data = await self._make_request("public/ticker", {
                    "instrument_name": instrument_name
                })

                if ticker_data:
                    option_info = {
                        "symbol": instrument_name,
                        "underlying": currency,
                        "strike": instrument["strike"],
                        "expiry": instrument["expiration_timestamp"] / 1000,  # Convert to seconds
                        "option_type": instrument["option_type"],
                        "price": ticker_data.get("last_price", 0),
                        "bid": ticker_data.get("best_bid_price", 0),
                        "ask": ticker_data.get("best_ask_price", 0),
                        "volume": ticker_data.get("stats", {}).get("volume", 0),
                        "open_interest": ticker_data.get("open_interest", 0),
                        "implied_volatility": ticker_data.get("mark_iv", 0),
                        "delta": ticker_data.get("greeks", {}).get("delta", 0),
                        "gamma": ticker_data.get("greeks", {}).get("gamma", 0),
                        "theta": ticker_data.get("greeks", {}).get("theta", 0),
                        "vega": ticker_data.get("greeks", {}).get("vega", 0),
                        "timestamp": datetime.utcnow(),
                        "source": "deribit"
                    }

                    options_data.append(option_info)

                # Rate limiting
                await asyncio.sleep(0.1)

            df = pd.DataFrame(options_data)

            if not df.empty:
                # Convert expiry to datetime
                df['expiry_date'] = pd.to_datetime(df['expiry'], unit='s')
                df['time_to_expiry'] = (df['expiry_date'] - datetime.utcnow()).dt.days

                logger.info(f"✅ Retrieved {len(df)} {currency} options from Deribit")

            return df

        except Exception as e:
            logger.error(f"Error getting options chain: {e}")
            return pd.DataFrame()

    async def get_volatility_surface(self, currency: str = "BTC") -> Dict[str, Any]:
        """
        Calculate volatility surface from options data

        Args:
            currency: Currency (BTC, ETH)

        Returns:
            Dictionary with volatility surface data
        """
        try:
            logger.info(f"Calculating {currency} volatility surface...")

            options_df = await self.get_options_chain(currency)

            if options_df.empty:
                return {}

            # Get current underlying price
            ticker_data = await self._make_request("public/ticker", {
                "instrument_name": f"{currency}-PERPETUAL"
            })

            if not ticker_data:
                logger.warning(f"Could not get {currency} spot price")
                return {}

            spot_price = ticker_data.get("last_price", 0)

            # Calculate moneyness for options
            options_df['moneyness'] = options_df['strike'] / spot_price

            # Filter for liquid options
            liquid_options = options_df[
                (options_df['volume'] > 0) &
                (options_df['implied_volatility'] > 0) &
                (options_df['time_to_expiry'] > 0)
            ]

            if liquid_options.empty:
                logger.warning(f"No liquid options found for {currency}")
                return {}

            # Create volatility surface grid
            surface_data = {}

            for expiry_days in [7, 30, 90, 180]:  # Standard expiry buckets
                expiry_options = liquid_options[
                    abs(liquid_options['time_to_expiry'] - expiry_days) <= 7
                ]

                if not expiry_options.empty:
                    surface_data[f"{expiry_days}d"] = {
                        "strikes": expiry_options['strike'].tolist(),
                        "moneyness": expiry_options['moneyness'].tolist(),
                        "implied_vol": expiry_options['implied_volatility'].tolist(),
                        "volume": expiry_options['volume'].tolist()
                    }

            result = {
                "currency": currency,
                "spot_price": spot_price,
                "surface": surface_data,
                "total_options": len(liquid_options),
                "timestamp": datetime.utcnow(),
                "source": "deribit"
            }

            logger.info(f"✅ {currency} volatility surface calculated with {len(surface_data)} expiry buckets")
            return result

        except Exception as e:
            logger.error(f"Error calculating volatility surface: {e}")
            return {}

    async def get_order_book(self, instrument_name: str, depth: int = 10) -> Dict[str, Any]:
        """
        Get order book for an instrument

        Args:
            instrument_name: Deribit instrument name
            depth: Order book depth

        Returns:
            Order book data
        """
        try:
            params = {
                "instrument_name": instrument_name,
                "depth": depth
            }

            result = await self._make_request("public/get_order_book", params)

            if result:
                return {
                    "instrument": instrument_name,
                    "bids": result.get("bids", []),
                    "asks": result.get("asks", []),
                    "timestamp": datetime.utcnow(),
                    "source": "deribit"
                }

            return {}

        except Exception as e:
            logger.error(f"Error getting order book for {instrument_name}: {e}")
            return {}

    async def get_historical_volatility(self, currency: str = "BTC", days: int = 30) -> List[Dict]:
        """
        Get historical volatility data

        Args:
            currency: Currency (BTC, ETH)
            days: Number of days to fetch

        Returns:
            List of historical volatility data points
        """
        try:
            # Get historical data using volatility index
            params = {
                "currency": currency,
                "start_timestamp": int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000),
                "end_timestamp": int(datetime.utcnow().timestamp() * 1000),
                "resolution": "1D"
            }

            # Note: This endpoint might require authentication
            result = await self._make_request("public/get_historical_volatility", params)

            if result:
                volatility_data = []
                for point in result:
                    volatility_data.append({
                        "currency": currency,
                        "timestamp": datetime.fromtimestamp(point[0] / 1000),
                        "volatility": point[1],
                        "source": "deribit"
                    })

                logger.info(f"Retrieved {len(volatility_data)} historical volatility points for {currency}")
                return volatility_data

            return []

        except Exception as e:
            logger.error(f"Error getting historical volatility: {e}")
            return []

    async def get_current_price(self, currency: str = "BTC") -> Optional[float]:
        """
        Get current spot price for a currency

        Args:
            currency: Currency (BTC, ETH)

        Returns:
            Current price or None if error
        """
        try:
            ticker_data = await self._make_request("public/ticker", {
                "instrument_name": f"{currency}-PERPETUAL"
            })

            if ticker_data:
                return ticker_data.get("last_price")

            return None

        except Exception as e:
            logger.error(f"Error getting current price for {currency}: {e}")
            return None

# Convenience functions for easy usage
async def get_btc_options_chain() -> pd.DataFrame:
    """Get BTC options chain from Deribit"""
    async with DeribitProvider() as deribit:
        return await deribit.get_options_chain("BTC")

async def get_eth_options_chain() -> pd.DataFrame:
    """Get ETH options chain from Deribit"""
    async with DeribitProvider() as deribit:
        return await deribit.get_options_chain("ETH")

async def get_btc_volatility_surface() -> Dict[str, Any]:
    """Get BTC volatility surface from Deribit"""
    async with DeribitProvider() as deribit:
        return await deribit.get_volatility_surface("BTC")

if __name__ == "__main__":
    # Example usage
    import asyncio

    async def main():
        print("🔗 Testing Deribit API integration...")

        async with DeribitProvider() as deribit:
            # Test current prices
            btc_price = await deribit.get_current_price("BTC")
            eth_price = await deribit.get_current_price("ETH")

            print(f"💰 Current Prices:")
            print(f"  BTC: ${btc_price:,.2f}" if btc_price else "  BTC: Data unavailable")
            print(f"  ETH: ${eth_price:,.2f}" if eth_price else "  ETH: Data unavailable")

            # Test options chain
            print(f"\n📊 Testing options chain retrieval...")
            btc_options = await deribit.get_options_chain("BTC")

            if not btc_options.empty:
                print(f"✅ Retrieved {len(btc_options)} BTC options")
                print(f"  Expiry range: {btc_options['time_to_expiry'].min()}-{btc_options['time_to_expiry'].max()} days")
                print(f"  Strike range: ${btc_options['strike'].min():,.0f}-${btc_options['strike'].max():,.0f}")
            else:
                print("⚠️  No BTC options data retrieved")

            # Test volatility surface
            print(f"\n📈 Testing volatility surface...")
            vol_surface = await deribit.get_volatility_surface("BTC")

            if vol_surface:
                print(f"✅ BTC volatility surface calculated")
                print(f"  Spot price: ${vol_surface['spot_price']:,.2f}")
                print(f"  Expiry buckets: {list(vol_surface['surface'].keys())}")
                print(f"  Total liquid options: {vol_surface['total_options']}")
            else:
                print("⚠️  No volatility surface data available")

        print(f"\n🎉 Deribit API integration test completed!")

    asyncio.run(main())