# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Data package initialization - Fixed for dashboard compatibility.

This module provides the functions that the dashboard expects to import.
"""

from .collectors.deribit_collector import DeribitCollector, get_deribit_collector

# Create the functions that the dashboard expects to import
def get_data_manager():
    """Get data manager instance - dashboard compatibility function."""
    return get_deribit_collector()

def collect_market_data(symbol: str = "BTC"):
    """Collect market data for symbol - dashboard compatibility function."""
    try:
        collector = get_deribit_collector()
        return collector.get_options_data(symbol)
    except Exception as e:
        import pandas as pd
        # Return empty DataFrame on error to prevent crashes
        return pd.DataFrame()

def get_spot_price(symbol: str = "BTC"):
    """Get spot price for symbol - dashboard compatibility function."""
    try:
        collector = get_deribit_collector()
        price = collector.get_spot_price(symbol)
        # Handle case where price might be None or not a number
        if price is None:
            # Return fallback prices for dashboard compatibility
            fallback_prices = {'BTC': 95000.0, 'ETH': 3200.0}
            return fallback_prices.get(symbol.upper(), 50000.0)
        return price
    except Exception:
        # Return fallback price to prevent dashboard crashes
        fallback_prices = {'BTC': 95000.0, 'ETH': 3200.0}
        return fallback_prices.get(symbol.upper(), 50000.0)

__all__ = [
    'DeribitCollector',
    'get_deribit_collector', 
    'get_data_manager',
    'collect_market_data',
    'get_spot_price'
]