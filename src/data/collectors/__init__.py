# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Data Collectors Module for Qortfolio V2

Contains specific implementations for different data sources:
- BaseDataCollector: Abstract base class with common functionality
- CryptoCollector: yfinance integration for cryptocurrency prices
- DeribitCollector: Deribit API integration for options data
- DataManager: Unified coordinator for all data collection
"""

from .base_collector import BaseDataCollector, CollectionResult, DataCollectionError
from .crypto_collector import CryptoCollector
from .deribit_collector import DeribitCollector
from .data_manager import DataManager, MarketData

__all__ = [
    'BaseDataCollector',
    'CryptoCollector', 
    'DeribitCollector',
    'DataManager',
    'CollectionResult',
    'DataCollectionError',
    'MarketData'
]