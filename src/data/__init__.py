# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Data Collection Module for Qortfolio V2

This module provides comprehensive data collection capabilities for:
- Cryptocurrency price data (via yfinance)
- Options market data (via Deribit)
- Real-time and historical data
- Unified data management with caching

Main Components:
- DataManager: Unified interface for all data operations
- CryptoCollector: Cryptocurrency price data from yfinance
- DeribitCollector: Options data from Deribit public API
- Caching and error handling infrastructure
"""

from .collectors.data_manager import (
    DataManager,
    get_data_manager,
    get_current_prices,
    get_options_data,
    get_crypto_history,
    MarketData
)

from .collectors.crypto_collector import (
    CryptoCollector,
    get_crypto_data,
    get_current_crypto_prices
)

from .collectors.deribit_collector import (
    DeribitCollector,
    get_options_data as get_deribit_options,
    get_current_spot_prices
)

from .collectors.base_collector import (
    BaseDataCollector,
    CollectionResult,
    DataCollectionError
)

# Version info
__version__ = "0.1.0"
__author__ = "Mhmd Fasihi"

# Module-level convenience functions
def collect_market_data(symbols, include_options=True, include_historical=False, **kwargs):
    """
    Convenience function to collect comprehensive market data.
    
    Args:
        symbols: List of cryptocurrency symbols
        include_options: Include options data (default: True)
        include_historical: Include historical data (default: False)
        **kwargs: Additional parameters
        
    Returns:
        MarketData object with collected data
    """
    dm = get_data_manager()
    return dm.get_market_data(
        symbols=symbols,
        include_options=include_options,
        include_historical=include_historical,
        **kwargs
    )


def get_spot_price(symbol):
    """
    Get current spot price for a cryptocurrency.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., "BTC", "ETH")
        
    Returns:
        Current spot price or None if failed
    """
    dm = get_data_manager()
    return dm.get_spot_price(symbol)


def get_options_chain(symbol, expiry_date=None):
    """
    Get options chain for a cryptocurrency.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., "BTC", "ETH")
        expiry_date: Specific expiry date or None for all
        
    Returns:
        DataFrame with options chain or None if failed
    """
    dm = get_data_manager()
    return dm.get_options_chain(symbol, expiry_date)


# Export main classes and functions
__all__ = [
    # Main classes
    'DataManager',
    'CryptoCollector', 
    'DeribitCollector',
    'BaseDataCollector',
    'CollectionResult',
    'DataCollectionError',
    'MarketData',
    
    # Factory functions
    'get_data_manager',
    
    # Convenience functions
    'collect_market_data',
    'get_spot_price',
    'get_options_chain',
    'get_current_prices',
    'get_crypto_history',
    'get_crypto_data',
    'get_current_crypto_prices',
    'get_current_spot_prices',
]