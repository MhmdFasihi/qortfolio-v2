# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Deribit Options Data Collector
Collects real-time options data from Deribit public API
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
import time
import json
from dateutil import parser

from .base_collector import BaseDataCollector, CollectionResult, DataCollectionError
from .base_collector import validate_dataframe_structure, clean_numeric_data
from core.logging import log_data_collection
from core.utils.time_utils import calculate_time_to_maturity


class DeribitCollector(BaseDataCollector):
    """
    Deribit options data collector using public API.
    
    Provides:
    - Options chain data (calls and puts)
    - Current market prices and implied volatility
    - Options Greeks (if available)
    - Instrument details and specifications
    """
    
    def __init__(self):
        """Initialize Deribit data collector."""
        super().__init__("deribit")
        
        # API configuration from config
        self.base_url = self.config.get('deribit_api.base_url', 'https://www.deribit.com/api/v2')
        self.endpoints = self.config.get('deribit_api.public_endpoints', {})
        
        # Required columns for options data
        self.required_options_columns = [
            'instrument_name', 'mark_price', 'bid_price', 'ask_price', 
            'strike', 'option_type', 'expiration_timestamp'
        ]
        self.numeric_options_columns = [
            'mark_price', 'bid_price', 'ask_price', 'strike', 'volume', 
            'open_interest', 'implied_volatility', 'delta', 'gamma', 'theta', 'vega'
        ]
        
        self.logger.info("DeribitCollector initialized for options data")
    
    def collect_data(self, symbol: str, **kwargs) -> CollectionResult:
        """
        Collect options data for a cryptocurrency.
        
        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH")
            **kwargs: Additional parameters:
                - kind: Type of instrument ("option", "future", "spot")
                - expired: Include expired options (default: False)
                - currency: Override currency (uses symbol by default)
                
        Returns:
            Collection result with options data
        """
        start_time = time.time()
        
        # Get Deribit currency from configuration
        currency = kwargs.get('currency') or self.config.get_deribit_currency(symbol)
        if not currency:
            error = f"Symbol '{symbol}' not supported on Deribit or not enabled"
            self.logger.warning(error, extra={"symbol": symbol})
            return self._create_error_result(error, symbol)
        
        try:
            # Get parameters
            kind = kwargs.get('kind', 'option')
            expired = kwargs.get('expired', False)
            
            self.logger.debug(f"Collecting {kind} data for {currency}", extra={
                "symbol": symbol,
                "currency": currency,
                "kind": kind,
                "expired": expired
            })
            
            # Get instruments
            instruments_data = self._get_instruments(currency, kind, expired)
            if not instruments_data:
                error = f"No instruments found for {currency} {kind}"
                return self._create_error_result(error, symbol)
            
            # Collect market data for each instrument
            options_data = []
            
            for instrument in instruments_data:
                try:
                    market_data = self._get_market_data(instrument['instrument_name'])
                    if market_data:
                        # Combine instrument and market data
                        combined_data = {**instrument, **market_data}
                        options_data.append(combined_data)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to get market data for {instrument.get('instrument_name')}: {e}")
                    continue
            
            response_time = time.time() - start_time
            
            if not options_data:
                error = f"No market data collected for {currency} {kind}"
                return self._create_error_result(error, symbol)
            
            # Convert to DataFrame
            df = pd.DataFrame(options_data)
            
            # Validate and clean data
            if not self.validate_data(df):
                error = f"Invalid options data for {currency}"
                self.logger.error(error, extra={
                    "symbol": symbol,
                    "currency": currency,
                    "data_shape": df.shape
                })
                return self._create_error_result(error, symbol)
            
            # Process options data
            processed_data = self._process_options_data(df, symbol, currency)
            
            # Log successful collection
            log_data_collection(
                data_type="options_data",
                symbol=symbol,
                records_count=len(processed_data),
                success=True
            )
            
            self.logger.info(f"Successfully collected options data for {symbol}", extra={
                "symbol": symbol,
                "currency": currency,
                "records": len(processed_data),
                "kind": kind,
                "response_time": response_time
            })
            
            return self._create_success_result(processed_data, response_time)
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"Failed to collect options data for {symbol}: {str(e)}"
            
            self.logger.error(error_msg, extra={
                "symbol": symbol,
                "currency": currency,
                "error": str(e),
                "response_time": response_time
            })
            
            log_data_collection(
                data_type="options_data",
                symbol=symbol,
                records_count=0,
                success=False,
                error=error_msg
            )
            
            return self._create_error_result(error_msg, symbol)
    
    def _get_instruments(self, currency: str, kind: str = "option", expired: bool = False) -> List[Dict]:
        """
        Get list of available instruments.
        
        Args:
            currency: Currency code (e.g., "BTC", "ETH")
            kind: Instrument kind
            expired: Include expired instruments
            
        Returns:
            List of instrument dictionaries
        """
        endpoint = self.endpoints.get('instruments', '/public/get_instruments')
        url = f"{self.base_url}{endpoint}"
        
        params = {
            'currency': currency,
            'kind': kind,
            'expired': expired
        }
        
        try:
            response = self._make_request(url, params)
            data = response.json()
            
            if 'result' in data:
                return data['result']
            else:
                self.logger.error(f"Unexpected response format for instruments: {data}")
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to get instruments for {currency}: {e}")
            return []
    
    def _get_market_data(self, instrument_name: str) -> Optional[Dict]:
        """
        Get market data for a specific instrument.
        
        Args:
            instrument_name: Deribit instrument name
            
        Returns:
            Market data dictionary or None if failed
        """
        endpoint = self.endpoints.get('ticker', '/public/ticker')
        url = f"{self.base_url}{endpoint}"
        
        params = {'instrument_name': instrument_name}
        
        try:
            response = self._make_request(url, params)
            data = response.json()
            
            if 'result' in data:
                return data['result']
            else:
                return None
                
        except Exception as e:
            self.logger.debug(f"Failed to get market data for {instrument_name}: {e}")
            return None
    
    def get_options_chain(self, symbol: str, expiry_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Get options chain for a specific expiry date.
        
        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH")
            expiry_date: Specific expiry date (YYYY-MM-DD) or None for all
            
        Returns:
            DataFrame with options chain or None if failed
        """
        result = self.collect_data(symbol, kind="option", expired=False)
        
        if not result.success:
            return None
        
        data = result.data
        
        if expiry_date:
            # Filter by expiry date
            try:
                target_date = pd.to_datetime(expiry_date).date()
                data = data[data['expiry_date'] == target_date]
            except Exception as e:
                self.logger.error(f"Invalid expiry date format '{expiry_date}': {e}")
                return None
        
        return data
    
    def get_spot_price(self, symbol: str) -> Optional[float]:
        """
        Get current spot price for a cryptocurrency.
        
        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH")
            
        Returns:
            Current spot price or None if failed
        """
        currency = self.config.get_deribit_currency(symbol)
        if not currency:
            return None
        
        try:
            # Get spot index price
            endpoint = '/public/get_index'
            url = f"{self.base_url}{endpoint}"
            params = {'currency': currency}
            
            response = self._make_request(url, params)
            data = response.json()
            
            if 'result' in data:
                return float(data['result'].get('index_price', 0))
            
        except Exception as e:
            self.logger.error(f"Failed to get spot price for {symbol}: {e}")
        
        return None
    
    def get_volatility_index(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get volatility index data.
        
        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH")
            
        Returns:
            Dictionary with volatility data or None if failed
        """
        currency = self.config.get_deribit_currency(symbol)
        if not currency:
            return None
        
        try:
            endpoint = '/public/get_volatility_index_data'
            url = f"{self.base_url}{endpoint}"
            params = {'currency': currency}
            
            response = self._make_request(url, params)
            data = response.json()
            
            if 'result' in data:
                return data['result']
            
        except Exception as e:
            self.logger.error(f"Failed to get volatility index for {symbol}: {e}")
        
        return None
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate collected options data.
        
        Args:
            data: Options data to validate
            
        Returns:
            True if data is valid
        """
        # Check basic structure
        if not validate_dataframe_structure(data, self.required_options_columns):
            return False
        
        # Additional options-specific validation
        if 'strike' in data.columns:
            # Check for valid strike prices
            valid_strikes = data['strike'] > 0
            if not valid_strikes.any():
                return False
        
        if 'option_type' in data.columns:
            # Check for valid option types
            valid_types = data['option_type'].isin(['call', 'put', 'C', 'P'])
            if not valid_types.any():
                return False
        
        return True
    
    def _process_options_data(self, data: pd.DataFrame, symbol: str, currency: str) -> pd.DataFrame:
        """
        Process and clean options data.
        
        Args:
            data: Raw options data from Deribit
            symbol: Original symbol
            currency: Deribit currency
            
        Returns:
            Processed DataFrame
        """
        # Clean numeric data
        processed = clean_numeric_data(data, self.numeric_options_columns)
        
        # Parse and clean instrument names
        processed = self._parse_instrument_names(processed)
        
        # Add metadata
        processed['Symbol'] = symbol
        processed['Currency'] = currency
        processed['Source'] = 'deribit'
        processed['CollectionTimestamp'] = datetime.now()
        
        # Calculate time to maturity using our FIXED calculation
        if 'expiration_timestamp' in processed.columns:
            processed['TimeToMaturity'] = processed['expiration_timestamp'].apply(
                lambda x: self._calculate_time_to_maturity_from_timestamp(x)
            )
        
        # Calculate moneyness (if spot price available)
        spot_price = self.get_spot_price(symbol)
        if spot_price and 'strike' in processed.columns:
            processed['SpotPrice'] = spot_price
            processed['Moneyness'] = processed['strike'] / spot_price
            processed['InTheMoney'] = self._calculate_itm_status(processed, spot_price)
        
        # Calculate bid-ask spread
        if 'bid_price' in processed.columns and 'ask_price' in processed.columns:
            processed['BidAskSpread'] = processed['ask_price'] - processed['bid_price']
            processed['BidAskSpreadPct'] = (
                processed['BidAskSpread'] / processed['mark_price'] * 100
            ).round(2)
        
        # Sort by strike and expiry
        sort_columns = []
        if 'expiry_date' in processed.columns:
            sort_columns.append('expiry_date')
        if 'option_type' in processed.columns:
            sort_columns.append('option_type')
        if 'strike' in processed.columns:
            sort_columns.append('strike')
        
        if sort_columns:
            processed = processed.sort_values(sort_columns)
        
        return processed
    
    def _parse_instrument_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Parse Deribit instrument names to extract components.
        
        Format: BTC-25JUL25-50000-C (currency-expiry-strike-type)
        
        Args:
            data: DataFrame with instrument_name column
            
        Returns:
            DataFrame with parsed components
        """
        if 'instrument_name' not in data.columns:
            return data
        
        # Parse instrument names
        parsed_data = []
        
        for _, row in data.iterrows():
            try:
                instrument_name = row['instrument_name']
                parts = instrument_name.split('-')
                
                if len(parts) >= 4:
                    currency_part = parts[0]
                    expiry_part = parts[1]
                    strike_part = parts[2]
                    option_type_part = parts[3]
                    
                    # Parse expiry date
                    expiry_date = self._parse_expiry_date(expiry_part)
                    
                    # Add parsed fields
                    row_dict = row.to_dict()
                    row_dict.update({
                        'parsed_currency': currency_part,
                        'expiry_string': expiry_part,
                        'expiry_date': expiry_date,
                        'strike': float(strike_part) if strike_part.replace('.', '').isdigit() else row.get('strike', 0),
                        'option_type': 'call' if option_type_part.upper() == 'C' else 'put'
                    })
                    
                    parsed_data.append(row_dict)
                else:
                    # Keep original row if parsing fails
                    parsed_data.append(row.to_dict())
                    
            except Exception as e:
                self.logger.debug(f"Failed to parse instrument name {row.get('instrument_name')}: {e}")
                parsed_data.append(row.to_dict())
        
        return pd.DataFrame(parsed_data)
    
    def _parse_expiry_date(self, expiry_string: str) -> Optional[datetime]:
        """
        Parse Deribit expiry date string.
        
        Args:
            expiry_string: Expiry string (e.g., "25JUL25")
            
        Returns:
            Parsed datetime or None if failed
        """
        try:
            # Handle different formats
            if len(expiry_string) == 7:  # 25JUL25
                day = expiry_string[:2]
                month = expiry_string[2:5]
                year = "20" + expiry_string[5:]
                
                # Convert month abbreviation
                month_map = {
                    'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
                    'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
                    'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
                }
                
                if month in month_map:
                    date_string = f"{year}-{month_map[month]}-{day}"
                    return datetime.strptime(date_string, "%Y-%m-%d")
            
            return None
            
        except Exception:
            return None
    
    def _calculate_time_to_maturity_from_timestamp(self, timestamp: Union[int, float]) -> float:
        """
        Calculate time to maturity from Unix timestamp using our FIXED calculation.
        
        Args:
            timestamp: Unix timestamp (seconds)
            
        Returns:
            Time to maturity in years
        """
        try:
            if pd.isna(timestamp) or timestamp <= 0:
                return 0.0
            
            current_time = datetime.now()
            expiry_time = datetime.fromtimestamp(timestamp / 1000 if timestamp > 1e10 else timestamp)
            
            # Use our FIXED time calculation
            return calculate_time_to_maturity(current_time, expiry_time)
            
        except Exception:
            return 0.0
    
    def _calculate_itm_status(self, data: pd.DataFrame, spot_price: float) -> pd.Series:
        """
        Calculate in-the-money status for options.
        
        Args:
            data: Options data
            spot_price: Current spot price
            
        Returns:
            Series with ITM status (True/False)
        """
        if 'option_type' not in data.columns or 'strike' not in data.columns:
            return pd.Series([False] * len(data))
        
        conditions = [
            (data['option_type'] == 'call') & (spot_price > data['strike']),
            (data['option_type'] == 'put') & (spot_price < data['strike'])
        ]
        
        return pd.Series(np.select(conditions, [True, True], default=False))
    
    def get_supported_currencies(self) -> List[str]:
        """Get list of supported currencies on Deribit."""
        return self.config.deribit_currencies


# Convenience functions
def get_options_data(symbol: str, expired: bool = False) -> Optional[pd.DataFrame]:
    """
    Convenience function to get options data.
    
    Args:
        symbol: Crypto symbol (e.g., "BTC", "ETH")
        expired: Include expired options
        
    Returns:
        DataFrame with options data or None if failed
    """
    collector = DeribitCollector()
    result = collector.collect_data(symbol, kind="option", expired=expired)
    
    if result.success:
        return result.data
    else:
        return None


def get_current_spot_prices(symbols: List[str]) -> Dict[str, float]:
    """
    Get current spot prices from Deribit.
    
    Args:
        symbols: List of crypto symbols
        
    Returns:
        Dictionary mapping symbols to spot prices
    """
    collector = DeribitCollector()
    prices = {}
    
    for symbol in symbols:
        price = collector.get_spot_price(symbol)
        if price is not None:
            prices[symbol] = price
    
    return prices


if __name__ == "__main__":
    # Test the Deribit collector
    print("🧪 Testing Deribit Options Data Collector")
    print("=" * 45)
    
    collector = DeribitCollector()
    
    # Test options data collection
    print("Testing BTC options data collection...")
    result = collector.collect_data("BTC", kind="option", expired=False)
    
    if result.success:
        print(f"✅ Successfully collected {result.records_count} options")
        print(f"📊 Data shape: {result.data.shape}")
        print(f"⏱️ Response time: {result.response_time:.2f}s")
        print(f"📋 Sample columns: {list(result.data.columns)[:10]}")
        
        # Show sample data
        if len(result.data) > 0:
            print("\n📈 Sample options data:")
            sample_cols = ['instrument_name', 'option_type', 'strike', 'mark_price', 'TimeToMaturity']
            available_cols = [col for col in sample_cols if col in result.data.columns]
            print(result.data[available_cols].head(3).to_string())
    else:
        print(f"❌ Collection failed: {result.error}")
    
    # Test spot price
    print("\nTesting spot price...")
    spot_price = collector.get_spot_price("BTC")
    if spot_price:
        print(f"✅ Current BTC spot price: ${spot_price:,.2f}")
    else:
        print("❌ Failed to get spot price")
    
    # Test statistics
    print("\nCollector statistics:")
    stats = collector.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n🎉 Deribit collector test completed!")