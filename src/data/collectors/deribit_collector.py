# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Deribit options data collector.
Fetches real-time options chain data via WebSocket and REST API.
Handles coin-based premiums correctly.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import aiohttp
import websockets
import pandas as pd
import numpy as np
import logging

from .base_collector import BaseDataCollector
from src.core.config import config
from src.core.exceptions import (
    DataCollectionError,
    APIConnectionError,
    ValidationError
)
from src.core.utils.time_utils import TimeUtils
from src.core.database.models import OptionsData, OptionType

logger = logging.getLogger(__name__)

@dataclass
class DeribitInstrument:
    """Deribit instrument information."""
    instrument_name: str
    underlying: str
    expiry: datetime
    strike: float
    option_type: str  # 'call' or 'put'
    min_trade_amount: float
    tick_size: float
    is_active: bool
    
    @classmethod
    def from_api_data(cls, data: Dict) -> 'DeribitInstrument':
        """Create instrument from Deribit API data."""
        # Parse instrument name (e.g., "BTC-31JAN25-100000-C")
        parts = data['instrument_name'].split('-')
        
        # Parse expiry date
        expiry_str = parts[1]  # e.g., "31JAN25"
        expiry = datetime.strptime(expiry_str, "%d%b%y")
        
        return cls(
            instrument_name=data['instrument_name'],
            underlying=parts[0],  # BTC or ETH
            expiry=expiry,
            strike=float(parts[2]),
            option_type='call' if parts[3] == 'C' else 'put',
            min_trade_amount=data.get('min_trade_amount', 0.1),
            tick_size=data.get('tick_size', 0.0001),
            is_active=data.get('is_active', True)
        )

class DeribitCollector(BaseDataCollector):
    """
    Collector for Deribit options data.
    Supports both REST API and WebSocket connections.
    """
    
    def __init__(
        self,
        testnet: bool = True,
        rate_limit: float = 10.0,  # 10 requests per second for Deribit
        cache_ttl: int = 60,  # 1 minute cache for options
        enable_cache: bool = True
    ):
        """
        Initialize Deribit collector.
        
        Args:
            testnet: Whether to use testnet (True) or mainnet (False)
            rate_limit: Maximum requests per second
            cache_ttl: Cache time-to-live in seconds
            enable_cache: Whether to enable caching
        """
        super().__init__(
            name="DeribitCollector",
            rate_limit=rate_limit or config.api.deribit_rate_limit,
            cache_ttl=cache_ttl,
            enable_cache=enable_cache
        )
        
        # API configuration
        self.testnet = testnet or (config.app_settings['environment'] != 'production')
        self.base_url = "https://test.deribit.com" if self.testnet else "https://www.deribit.com"
        self.ws_url = "wss://test.deribit.com/ws/api/v2" if self.testnet else "wss://www.deribit.com/ws/api/v2"
        
        # Authentication
        self.client_id = config.api.deribit_client_id
        self.client_secret = config.api.deribit_client_secret
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = None
        
        # WebSocket connection
        self.ws_connection = None
        self.ws_subscriptions: Set[str] = set()
        self.ws_callbacks: Dict[str, Any] = {}
        
        # Supported currencies
        self.supported_currencies = ["BTC", "ETH"]
        
        logger.info(f"Initialized Deribit collector ({'testnet' if self.testnet else 'mainnet'})")
    
    # === Authentication ===
    
    async def authenticate(self) -> bool:
        """
        Authenticate with Deribit API.
        
        Returns:
            True if authentication successful
        """
        try:
            if not self.client_id or not self.client_secret:
                logger.warning("No Deribit credentials provided, using public endpoints only")
                return False
            
            session = await self.get_session()
            
            auth_data = {
                "jsonrpc": "2.0",
                "method": "public/auth",
                "params": {
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret
                },
                "id": 1
            }
            
            async with session.post(f"{self.base_url}/api/v2/public/auth", json=auth_data) as response:
                result = await response.json()
                
                if 'result' in result:
                    self.access_token = result['result']['access_token']
                    self.refresh_token = result['result']['refresh_token']
                    expires_in = result['result']['expires_in']
                    self.token_expiry = time.time() + expires_in - 60  # Refresh 1 minute early
                    
                    logger.info("✅ Deribit authentication successful")
                    return True
                else:
                    logger.error(f"Authentication failed: {result}")
                    return False
                    
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    async def ensure_authenticated(self) -> bool:
        """Ensure we have valid authentication."""
        if not self.access_token or time.time() >= self.token_expiry:
            return await self.authenticate()
        return True
    
    # === REST API Methods ===
    
    async def fetch_data(
        self,
        currency: str = "BTC",
        kind: str = "option",
        expired: bool = False
    ) -> List[Dict]:
        """
        Fetch instruments from Deribit REST API.
        
        Args:
            currency: Currency (BTC or ETH)
            kind: Instrument kind (option, future, spot, etc.)
            expired: Include expired instruments
            
        Returns:
            List of instrument data
        """
        try:
            if currency not in self.supported_currencies:
                raise ValidationError(f"Unsupported currency: {currency}")
            
            session = await self.get_session()
            
            params = {
                "currency": currency,
                "kind": kind,
                "expired": "false" if not expired else "true"
            }
            
            url = f"{self.base_url}/api/v2/public/get_instruments"
            
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                if 'result' in data:
                    logger.info(f"Fetched {len(data['result'])} {currency} {kind} instruments")
                    return data['result']
                else:
                    error_msg = data.get('error', {}).get('message', 'Unknown error')
                    raise APIConnectionError(f"Deribit API error: {error_msg}")
                    
        except Exception as e:
            logger.error(f"Failed to fetch instruments: {e}")
            raise DataCollectionError(f"Failed to fetch Deribit data: {e}")
    
    async def get_order_book(
        self,
        instrument_name: str,
        depth: int = 10
    ) -> Dict:
        """
        Get order book for an instrument.
        
        Args:
            instrument_name: Instrument name (e.g., "BTC-31JAN25-100000-C")
            depth: Order book depth
            
        Returns:
            Order book data
        """
        try:
            session = await self.get_session()
            
            params = {
                "instrument_name": instrument_name,
                "depth": depth
            }
            
            url = f"{self.base_url}/api/v2/public/get_order_book"
            
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                if 'result' in data:
                    return data['result']
                else:
                    raise APIConnectionError(f"Failed to get order book: {data}")
                    
        except Exception as e:
            logger.error(f"Failed to get order book for {instrument_name}: {e}")
            raise
    
    async def get_ticker(self, instrument_name: str) -> Dict:
        """
        Get ticker data for an instrument.
        
        Args:
            instrument_name: Instrument name
            
        Returns:
            Ticker data with prices, volumes, Greeks
        """
        try:
            session = await self.get_session()
            
            params = {"instrument_name": instrument_name}
            url = f"{self.base_url}/api/v2/public/ticker"
            
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                if 'result' in data:
                    return data['result']
                else:
                    raise APIConnectionError(f"Failed to get ticker: {data}")
                    
        except Exception as e:
            logger.error(f"Failed to get ticker for {instrument_name}: {e}")
            raise
    
    async def get_index_price(self, currency: str) -> float:
        """
        Get current index price for currency.
        
        Args:
            currency: Currency (BTC or ETH)
            
        Returns:
            Current index price
        """
        try:
            session = await self.get_session()
            
            params = {"index_name": f"{currency.lower()}_usd"}
            url = f"{self.base_url}/api/v2/public/get_index_price"
            
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                if 'result' in data:
                    return data['result']['index_price']
                else:
                    raise APIConnectionError(f"Failed to get index price: {data}")
                    
        except Exception as e:
            logger.error(f"Failed to get index price for {currency}: {e}")
            raise
    
    # === Data Validation ===
    
    async def validate_data(self, data: List[Dict]) -> bool:
        """
        Validate fetched options data.
        
        Args:
            data: List of instrument data
            
        Returns:
            True if valid
        """
        if not data:
            logger.warning("Empty data received")
            return True  # Empty is valid, just no options
        
        for item in data:
            # Check required fields
            required = ['instrument_name', 'strike', 'is_active']
            for field in required:
                if field not in item:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate strike price
            if item['strike'] <= 0:
                logger.error(f"Invalid strike price: {item['strike']}")
                return False
        
        return True
    
    # === Data Processing ===
    
    async def process_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Process raw Deribit data into standardized format.
        
        Args:
            raw_data: Raw instrument data from API
            
        Returns:
            List of processed options documents
        """
        processed_data = []
        
        for item in raw_data:
            try:
                # Skip non-option instruments
                if item.get('kind') != 'option':
                    continue
                
                # Skip inactive instruments if desired
                if not item.get('is_active', True):
                    continue
                
                # Parse instrument
                instrument = DeribitInstrument.from_api_data(item)
                
                # Calculate time to maturity using our fixed utility
                current_time = datetime.utcnow()
                time_to_maturity = TimeUtils.calculate_time_to_maturity(
                    current_time,
                    instrument.expiry
                )
                
                # Skip expired options
                if time_to_maturity <= 0:
                    continue
                
                # Get ticker data for prices and Greeks
                ticker_data = await self.get_ticker(instrument.instrument_name)
                
                # Create options data document
                option_doc = {
                    'symbol': instrument.instrument_name,
                    'underlying': instrument.underlying,
                    'strike': instrument.strike,
                    'expiry': instrument.expiry,
                    'option_type': instrument.option_type,
                    'time_to_maturity': time_to_maturity,
                    
                    # Prices (in crypto, not USD!)
                    'bid': ticker_data.get('best_bid_price', 0),
                    'ask': ticker_data.get('best_ask_price', 0),
                    'last_price': ticker_data.get('last_price', 0),
                    'mark_price': ticker_data.get('mark_price', 0),
                    
                    # Volumes
                    'volume': ticker_data.get('stats', {}).get('volume', 0),
                    'volume_usd': ticker_data.get('stats', {}).get('volume_usd', 0),
                    'open_interest': ticker_data.get('open_interest', 0),
                    
                    # Volatility
                    'mark_iv': ticker_data.get('mark_iv', 0),
                    'bid_iv': ticker_data.get('bid_iv', 0),
                    'ask_iv': ticker_data.get('ask_iv', 0),
                    
                    # Greeks
                    'delta': ticker_data.get('greeks', {}).get('delta', None),
                    'gamma': ticker_data.get('greeks', {}).get('gamma', None),
                    'theta': ticker_data.get('greeks', {}).get('theta', None),
                    'vega': ticker_data.get('greeks', {}).get('vega', None),
                    'rho': ticker_data.get('greeks', {}).get('rho', None),
                    
                    # Metadata
                    'timestamp': datetime.utcnow(),
                    'source': 'deribit',
                    'testnet': self.testnet,
                    'underlying_price': ticker_data.get('underlying_price', 0)
                }
                
                processed_data.append(option_doc)
                
            except Exception as e:
                logger.warning(f"Failed to process instrument {item.get('instrument_name')}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_data)} options from {len(raw_data)} instruments")
        return processed_data
    
    # === Options Chain Methods ===
    
    async def get_options_chain(
        self,
        currency: str = "BTC",
        expiry: Optional[datetime] = None,
        min_volume: float = 0,
        strikes_around_atm: int = 10
    ) -> pd.DataFrame:
        """
        Get complete options chain for a currency.
        
        Args:
            currency: Currency (BTC or ETH)
            expiry: Specific expiry date (None for all)
            min_volume: Minimum volume filter
            strikes_around_atm: Number of strikes around ATM to include
            
        Returns:
            DataFrame with options chain
        """
        try:
            # Fast path: avoid processing every instrument (too slow/heavy).
            # 1) Fetch raw instruments only
            raw = await self.fetch_data(currency=currency, kind="option", expired=False)
            if not raw:
                logger.warning(f"No options found for {currency}")
                return pd.DataFrame()

            # 2) Parse instruments and group by expiry
            parsed: List[DeribitInstrument] = []
            for item in raw:
                try:
                    if item.get('kind') != 'option' or not item.get('is_active', True):
                        continue
                    parsed.append(DeribitInstrument.from_api_data(item))
                except Exception:
                    continue

            if not parsed:
                return pd.DataFrame()

            # 3) Determine which expiries to include
            expiries = sorted({inst.expiry for inst in parsed})
            if expiry:
                selected_expiries = [expiry]
            else:
                # Limit to nearest few expiries to keep requests bounded
                selected_expiries = expiries[:3]

            # 4) Get spot price to pick strikes around ATM
            spot_price = await self.get_index_price(currency)

            # 5) Select subset of instruments around ATM for each expiry
            selected: List[DeribitInstrument] = []
            for exp in selected_expiries:
                exp_instruments = [i for i in parsed if i.expiry == exp]
                if not exp_instruments:
                    continue
                strikes = sorted({i.strike for i in exp_instruments})
                if not strikes:
                    continue
                atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
                idx = strikes.index(atm_strike)
                a = max(0, idx - max(1, strikes_around_atm))
                b = min(len(strikes), idx + max(1, strikes_around_atm) + 1)
                target_strikes = set(strikes[a:b])
                selected.extend([i for i in exp_instruments if i.strike in target_strikes])

            # Safety cap to avoid too many network calls
            MAX_INSTRUMENTS = 200
            if len(selected) > MAX_INSTRUMENTS:
                selected = selected[:MAX_INSTRUMENTS]

            # 6) Fetch ticker for selected instruments only
            rows: List[Dict] = []
            for inst in selected:
                try:
                    t = await self.get_ticker(inst.instrument_name)
                    rows.append({
                        'symbol': inst.instrument_name,
                        'underlying': inst.underlying,
                        'strike': inst.strike,
                        'expiry': inst.expiry,
                        'option_type': inst.option_type,
                        'time_to_maturity': TimeUtils.calculate_time_to_maturity(datetime.utcnow(), inst.expiry),
                        'bid': t.get('best_bid_price', 0),
                        'ask': t.get('best_ask_price', 0),
                        'last_price': t.get('last_price', 0),
                        'mark_price': t.get('mark_price', 0),
                        'volume': t.get('stats', {}).get('volume', 0),
                        'volume_usd': t.get('stats', {}).get('volume_usd', 0),
                        'open_interest': t.get('open_interest', 0),
                        'mark_iv': t.get('mark_iv', 0),
                        'bid_iv': t.get('bid_iv', 0),
                        'ask_iv': t.get('ask_iv', 0),
                        'delta': t.get('greeks', {}).get('delta'),
                        'gamma': t.get('greeks', {}).get('gamma'),
                        'theta': t.get('greeks', {}).get('theta'),
                        'vega': t.get('greeks', {}).get('vega'),
                        'rho': t.get('greeks', {}).get('rho'),
                        'timestamp': datetime.utcnow(),
                        'source': 'deribit',
                        'testnet': self.testnet,
                        'underlying_price': t.get('underlying_price', 0),
                    })
                except Exception:
                    continue

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows)

            # Filter by minimum volume
            if min_volume > 0:
                df = df[df['volume'] >= min_volume]

            # Sort and add moneyness flags
            df = df.sort_values(['expiry', 'strike', 'option_type'])
            df['moneyness'] = df['strike'] / max(spot_price, 1e-9)
            df['is_itm'] = df.apply(
                lambda row: (spot_price > row['strike']) if row['option_type'] == 'call'
                else (spot_price < row['strike']),
                axis=1
            )

            return df

        except Exception as e:
            logger.error(f"Failed to get options chain: {e}")
            raise
    
    # === WebSocket Methods ===
    
    async def connect_websocket(self):
        """Establish WebSocket connection to Deribit."""
        try:
            self.ws_connection = await websockets.connect(self.ws_url)
            logger.info("✅ WebSocket connected to Deribit")
            
            # Start listening for messages
            asyncio.create_task(self._ws_listener())
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise APIConnectionError(f"WebSocket connection failed: {e}")
    
    async def _ws_listener(self):
        """Listen for WebSocket messages."""
        try:
            async for message in self.ws_connection:
                data = json.loads(message)
                
                # Handle different message types
                if 'method' in data:
                    await self._handle_ws_notification(data)
                elif 'id' in data:
                    await self._handle_ws_response(data)
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            await self._reconnect_websocket()
        except Exception as e:
            logger.error(f"WebSocket listener error: {e}")
    
    async def _handle_ws_notification(self, data: Dict):
        """Handle WebSocket notifications."""
        method = data['method']
        params = data.get('params', {})
        
        # Route to appropriate handler
        if method == 'subscription':
            channel = params.get('channel', '')
            
            if channel.startswith('ticker'):
                await self._handle_ticker_update(params['data'])
            elif channel.startswith('book'):
                await self._handle_book_update(params['data'])
            elif channel.startswith('trades'):
                await self._handle_trades_update(params['data'])
    
    async def _handle_ticker_update(self, data: Dict):
        """Handle ticker updates."""
        instrument = data['instrument_name']
        
        # Update cache with fresh data
        cache_key = self._generate_cache_key(instrument=instrument)
        await self.set_cached_data(cache_key, data)
        
        # Call registered callbacks
        if instrument in self.ws_callbacks:
            await self.ws_callbacks[instrument](data)
    
    async def subscribe_ticker(
        self,
        instrument_name: str,
        callback: Optional[Any] = None
    ):
        """
        Subscribe to ticker updates for an instrument.
        
        Args:
            instrument_name: Instrument to subscribe to
            callback: Async function to call on updates
        """
        if not self.ws_connection:
            await self.connect_websocket()
        
        msg = {
            "jsonrpc": "2.0",
            "method": "public/subscribe",
            "params": {
                "channels": [f"ticker.{instrument_name}.raw"]
            },
            "id": int(time.time())
        }
        
        await self.ws_connection.send(json.dumps(msg))
        self.ws_subscriptions.add(instrument_name)
        
        if callback:
            self.ws_callbacks[instrument_name] = callback
        
        logger.info(f"Subscribed to ticker: {instrument_name}")
    
    async def _reconnect_websocket(self):
        """Reconnect WebSocket and restore subscriptions."""
        logger.info("Attempting WebSocket reconnection...")
        await asyncio.sleep(5)  # Wait before reconnecting
        
        await self.connect_websocket()
        
        # Restore subscriptions
        for instrument in self.ws_subscriptions:
            await self.subscribe_ticker(instrument)
    
    # === Cleanup ===
    
    async def close(self):
        """Close all connections."""
        if self.ws_connection:
            await self.ws_connection.close()
        
        await super().close()
        logger.info("Closed Deribit collector")

    # === Synchronous convenience ===
    def get_options_data(self, currency: str = "BTC") -> List[Dict]:
        """Synchronous helper to fetch processed options data.
        Returns list of option documents as dictionaries.
        """
        try:
            return asyncio.run(self.collect(currency=currency, kind="option", use_cache=True))
        except RuntimeError:
            # If an event loop is already running (e.g., within notebooks), create a new loop
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(self.collect(currency=currency, kind="option", use_cache=True))
            finally:
                loop.close()

# === Testing ===

async def test_deribit_collector():
    """Test Deribit collector functionality."""
    collector = DeribitCollector(testnet=True)
    
    print("\n📊 Testing Deribit Options Collector")
    print("=" * 50)
    
    try:
        # Test 1: Get BTC options
        print("\n1️⃣ Fetching BTC options instruments...")
        btc_options = await collector.collect(
            currency="BTC",
            kind="option"
        )
        print(f"   ✅ Found {len(btc_options)} BTC options")
        
        if btc_options:
            # Show sample option
            sample = btc_options[0]
            print(f"   Sample: {sample['symbol']}")
            print(f"   Strike: {sample['strike']}")
            print(f"   Expiry: {sample['expiry']}")
            print(f"   Bid/Ask: {sample['bid']:.4f}/{sample['ask']:.4f} BTC")
            
            # Show Greeks if available
            if sample.get('delta'):
                print(f"   Greeks: Δ={sample['delta']:.3f}, Γ={sample['gamma']:.4f}")
        
        # Test 2: Get index price
        print("\n2️⃣ Getting BTC index price...")
        btc_price = await collector.get_index_price("BTC")
        print(f"   ✅ BTC Index Price: ${btc_price:,.2f}")
        
        # Test 3: Get options chain
        print("\n3️⃣ Getting BTC options chain (ATM ± 5 strikes)...")
        chain = await collector.get_options_chain(
            currency="BTC",
            strikes_around_atm=5
        )
        
        if not chain.empty:
            print(f"   ✅ Options chain: {len(chain)} options")
            print(f"   Expiries: {chain['expiry'].nunique()}")
            print(f"   Strikes: {chain['strike'].nunique()}")
            
            # Show ATM options
            atm_options = chain[chain['moneyness'].between(0.95, 1.05)]
            if not atm_options.empty:
                print(f"   ATM options: {len(atm_options)}")
        
        # Show statistics
        print("\n📊 Collector Statistics:")
        stats = collector.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await collector.close()

if __name__ == "__main__":
    # Note: This will connect to Deribit testnet
    # No authentication required for public endpoints
    asyncio.run(test_deribit_collector())
