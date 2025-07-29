# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Fixed Deribit API Integration - Real-time Websocket Implementation
Location: src/data/collectors/deribit_collector.py

This fixes the 400 API errors by using proper websocket implementation
based on the original qortfolio repository and your websocket example.
"""

import asyncio
import json
import time
import websockets
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import ssl
import requests

from ...core.config import get_config
from ...core.logging import get_logger
from ...core.utils.time_utils import calculate_time_to_maturity

@dataclass
class OptionData:
    """Structure for option data."""
    instrument_name: str
    strike_price: float
    expiration_date: datetime
    option_type: str  # 'call' or 'put'
    mark_price: float
    bid_price: Optional[float]
    ask_price: Optional[float]
    mark_iv: Optional[float]
    volume: float
    open_interest: float
    underlying_price: float
    time_to_expiry: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'instrument_name': self.instrument_name,
            'strike': self.strike_price,
            'expiry': self.expiration_date,
            'type': self.option_type,
            'mark_price': self.mark_price,
            'bid': self.bid_price,
            'ask': self.ask_price,
            'iv': self.mark_iv,
            'volume': self.volume,
            'open_interest': self.open_interest,
            'underlying_price': self.underlying_price,
            'time_to_expiry': self.time_to_expiry
        }

class DeribitCollector:
    """
    Fixed Deribit API collector using websockets.
    
    This implementation fixes the 400 API errors by:
    1. Using websocket connection instead of REST API
    2. Proper parameter handling for Deribit API
    3. Real-time data collection
    4. Comprehensive error handling
    """
    
    def __init__(self):
        """Initialize Deribit collector with websocket support."""
        self.config = get_config()
        self.logger = get_logger("deribit_collector")
        
        # Use production websocket URL (change to test if needed)
        self.websocket_url = self.config.get('deribit_api.websocket_url', 'wss://www.deribit.com/ws/api/v2')
        self.test_websocket_url = self.config.get('deribit_api.test_websocket_url', 'wss://test.deribit.com/ws/api/v2')
        
        # Use test environment for development
        self.use_test_env = self.config.get('application.development_mode', True)
        self.current_url = self.test_websocket_url if self.use_test_env else self.websocket_url
        
        # Rate limiting
        self.rate_limit_delay = self.config.get('deribit_api.rate_limit_delay', 0.1)
        self.timeout = self.config.get('deribit_api.timeout', 30)
        
        # Connection management
        self._ws = None
        self._request_id = 1
        
        self.logger.info(f"DeribitCollector initialized with URL: {self.current_url}")
    
    async def _connect(self) -> bool:
        """Establish websocket connection with proper error handling."""
        try:
            if self._ws is not None:
                return True
            
            self.logger.info(f"Connecting to Deribit websocket: {self.current_url}")
            
            # Create SSL context for secure connection
            ssl_context = ssl.create_default_context()
            
            self._ws = await websockets.connect(
                self.current_url,
                ssl=ssl_context,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.logger.info("Websocket connection established successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Deribit websocket: {e}")
            self._ws = None
            return False
    
    async def _disconnect(self):
        """Close websocket connection."""
        if self._ws is not None:
            try:
                await self._ws.close()
                self.logger.debug("Websocket connection closed")
            except Exception as e:
                self.logger.warning(f"Error closing websocket: {e}")
            finally:
                self._ws = None
    
    async def _send_request(self, method: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Send websocket request with proper error handling.
        
        Args:
            method: API method name
            params: Method parameters
            
        Returns:
            API response or None if failed
        """
        try:
            if not await self._connect():
                return None
            
            # Create request message
            request = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params
            }
            
            self._request_id += 1
            
            # Send request
            await self._ws.send(json.dumps(request))
            self.logger.debug(f"Sent request: {method} with params: {params}")
            
            # Wait for response with timeout
            try:
                response_str = await asyncio.wait_for(
                    self._ws.recv(), 
                    timeout=self.timeout
                )
                response = json.loads(response_str)
                
                # Check for API errors
                if 'error' in response:
                    error = response['error']
                    self.logger.error(f"Deribit API error: {error}")
                    return None
                
                if 'result' in response:
                    return response['result']
                else:
                    self.logger.warning(f"No result in response: {response}")
                    return None
                    
            except asyncio.TimeoutError:
                self.logger.error(f"Request timeout for method {method}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in websocket request {method}: {e}")
            return None
    
    async def _get_instruments_async(self, currency: str, kind: str = "option", expired: bool = False) -> Optional[List[Dict]]:
        """
        Get instruments list asynchronously with proper parameters.
        
        This fixes the 400 error by using correct websocket method and parameters.
        """
        try:
            # Use correct Deribit API method
            method = "public/get_instruments"
            params = {
                "currency": currency.upper(),
                "kind": kind,
                "expired": expired
            }
            
            self.logger.info(f"Getting instruments for {currency} {kind}")
            
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            result = await self._send_request(method, params)
            
            if result is None:
                self.logger.error(f"Failed to get instruments for {currency}")
                return None
            
            if not isinstance(result, list):
                self.logger.error(f"Unexpected result type: {type(result)}")
                return None
            
            self.logger.info(f"Retrieved {len(result)} instruments for {currency}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting instruments for {currency}: {e}")
            return None
    
    async def _get_ticker_async(self, instrument_name: str) -> Optional[Dict]:
        """Get ticker data for instrument."""
        try:
            method = "public/ticker"
            params = {"instrument_name": instrument_name}
            
            await asyncio.sleep(self.rate_limit_delay)
            
            result = await self._send_request(method, params)
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting ticker for {instrument_name}: {e}")
            return None
    
    async def _get_spot_price_async(self, currency: str) -> Optional[float]:
        """Get current spot price for currency."""
        try:
            # Get spot index price
            method = "public/get_index_price"
            params = {"index_name": f"{currency.upper()}_USD"}
            
            result = await self._send_request(method, params)
            
            if result and 'index_price' in result:
                return float(result['index_price'])
            
            self.logger.warning(f"Could not get spot price for {currency}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting spot price for {currency}: {e}")
            return None
    
    def get_options_data(self, currency: str) -> pd.DataFrame:
        """
        Get options data synchronously (main public method).
        
        This is the method called by the dashboard.
        """
        try:
            # Run async function in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(self._get_options_data_async(currency))
                return result if result is not None else pd.DataFrame()
            finally:
                loop.close()
                
        except Exception as e:
            self.logger.error(f"Error in get_options_data for {currency}: {e}")
            return pd.DataFrame()
    
    async def _get_options_data_async(self, currency: str) -> Optional[pd.DataFrame]:
        """Get complete options data asynchronously."""
        try:
            # Get instruments
            instruments = await self._get_instruments_async(currency, "option", False)
            
            if not instruments:
                self.logger.warning(f"No instruments found for {currency}")
                return pd.DataFrame()
            
            # Get spot price
            spot_price = await self._get_spot_price_async(currency)
            if spot_price is None:
                self.logger.warning(f"Could not get spot price for {currency}, using fallback")
                # Fallback spot prices (use current realistic values)
                spot_price = 95000.0 if currency.upper() == 'BTC' else 3200.0
            
            self.logger.info(f"Using spot price {spot_price} for {currency}")
            
            # Process options data
            options_data = []
            current_time = datetime.now(timezone.utc)
            
            # Process instruments in batches to respect rate limits
            batch_size = 10
            for i in range(0, len(instruments), batch_size):
                batch = instruments[i:i + batch_size]
                
                for instrument in batch:
                    try:
                        option_data = self._process_instrument(instrument, spot_price, current_time)
                        if option_data:
                            options_data.append(option_data)
                    except Exception as e:
                        self.logger.warning(f"Error processing instrument {instrument.get('instrument_name', 'unknown')}: {e}")
                        continue
                
                # Rate limiting between batches
                if i + batch_size < len(instruments):
                    await asyncio.sleep(self.rate_limit_delay * batch_size)
            
            if not options_data:
                self.logger.warning(f"No valid options data processed for {currency}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df_data = [opt.to_dict() for opt in options_data]
            df = pd.DataFrame(df_data)
            
            # Clean and validate data
            df = self._clean_options_data(df)
            
            self.logger.info(f"Successfully processed {len(df)} options for {currency}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in _get_options_data_async for {currency}: {e}")
            return pd.DataFrame()
        finally:
            await self._disconnect()
    
    def _process_instrument(self, instrument: Dict, spot_price: float, current_time: datetime) -> Optional[OptionData]:
        """Process individual instrument data."""
        try:
            instrument_name = instrument.get('instrument_name', '')
            
            # Parse instrument name (e.g., "BTC-29JUL25-100000-C")
            parts = instrument_name.split('-')
            if len(parts) < 4:
                return None
            
            # Extract strike and option type
            try:
                strike_price = float(parts[2])
                option_type = 'call' if parts[3] == 'C' else 'put'
            except (ValueError, IndexError):
                return None
            
            # Parse expiration date from instrument name
            try:
                date_str = parts[1]  # e.g., "29JUL25"
                expiry_date = self._parse_expiry_date(date_str)
                if expiry_date is None:
                    return None
            except Exception:
                return None
            
            # Calculate time to expiry using FIXED calculation
            time_to_expiry = calculate_time_to_maturity(current_time, expiry_date)
            
            # Get market data
            mark_price = instrument.get('mark_price', 0.0)
            bid_price = instrument.get('bid_price')
            ask_price = instrument.get('ask_price')
            mark_iv = instrument.get('mark_iv')
            volume = instrument.get('volume', 0.0)
            open_interest = instrument.get('open_interest', 0.0)
            
            # Convert implied volatility from percentage to decimal
            if mark_iv is not None:
                mark_iv = mark_iv / 100.0
            
            return OptionData(
                instrument_name=instrument_name,
                strike_price=strike_price,
                expiration_date=expiry_date,
                option_type=option_type,
                mark_price=mark_price or 0.0,
                bid_price=bid_price,
                ask_price=ask_price,
                mark_iv=mark_iv,
                volume=volume or 0.0,
                open_interest=open_interest or 0.0,
                underlying_price=spot_price,
                time_to_expiry=time_to_expiry
            )
            
        except Exception as e:
            self.logger.warning(f"Error processing instrument {instrument.get('instrument_name', 'unknown')}: {e}")
            return None
    
    def _parse_expiry_date(self, date_str: str) -> Optional[datetime]:
        """Parse Deribit expiry date format (e.g., '29JUL25')."""
        try:
            # Handle different date formats
            month_map = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
            }
            
            if len(date_str) == 7:  # e.g., "29JUL25"
                day = int(date_str[:2])
                month_str = date_str[2:5]
                year = int("20" + date_str[5:7])  # Convert "25" to "2025"
                
                if month_str in month_map:
                    month = month_map[month_str]
                    # Set to end of day UTC for options expiry
                    return datetime(year, month, day, 16, 0, 0, tzinfo=timezone.utc)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error parsing date {date_str}: {e}")
            return None
    
    def _clean_options_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate options data."""
        if df.empty:
            return df
        
        try:
            # Remove invalid data
            df = df.dropna(subset=['strike', 'expiry', 'mark_price'])
            
            # Ensure positive strikes and prices
            df = df[df['strike'] > 0]
            df = df[df['mark_price'] >= 0]
            
            # Remove expired options
            current_time = datetime.now(timezone.utc)
            df = df[df['expiry'] > current_time]
            
            # Sort by expiry and strike
            df = df.sort_values(['expiry', 'strike'])
            
            # Reset index
            df = df.reset_index(drop=True)
            
            self.logger.info(f"Cleaned options data: {len(df)} valid options")
            return df
            
        except Exception as e:
            self.logger.error(f"Error cleaning options data: {e}")
            return df
    
    def get_spot_price(self, currency: str) -> Optional[float]:
        """Get current spot price synchronously."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(self._get_spot_price_async(currency))
                return result
            finally:
                loop.close()
                
        except Exception as e:
            self.logger.error(f"Error getting spot price for {currency}: {e}")
            # Return fallback prices
            fallback_prices = {'BTC': 95000.0, 'ETH': 3200.0}
            return fallback_prices.get(currency.upper())
    
    def test_connection(self) -> bool:
        """Test websocket connection."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                return loop.run_until_complete(self._test_connection_async())
            finally:
                loop.close()
                
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    async def _test_connection_async(self) -> bool:
        """Test websocket connection asynchronously."""
        try:
            if await self._connect():
                # Test with a simple API call
                result = await self._send_request("public/test", {})
                await self._disconnect()
                return result is not None
            return False
        except Exception as e:
            self.logger.error(f"Async connection test failed: {e}")
            return False

# Convenience function for external use
def get_deribit_collector() -> DeribitCollector:
    """Get configured Deribit collector instance."""
    return DeribitCollector()