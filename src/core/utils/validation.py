# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Data validation utilities for Qortfolio V2.
Ensures data integrity and validity across the platform.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import re
import logging
from ..exceptions import ValidationError, InvalidTickerError, InvalidDateRangeError

logger = logging.getLogger(__name__)

class ValidationUtils:
    """Data validation utility functions."""
    
    # Valid cryptocurrency tickers
    VALID_CRYPTOS = {
        "BTC", "ETH", "SOL", "LINK", "AVAX", "UNI", "AAVE", 
        "CRV", "MKR", "COMP", "SNX", "NEAR", "FET", "RENDER"
    }
    
    @classmethod
    def validate_ticker(cls, ticker: str) -> str:
        """
        Validate and normalize cryptocurrency ticker.
        
        Args:
            ticker: Ticker symbol to validate
            
        Returns:
            Normalized ticker symbol
            
        Raises:
            InvalidTickerError: If ticker is invalid
        """
        if not ticker:
            raise InvalidTickerError("Ticker cannot be empty")
        
        ticker = ticker.upper().strip()
        
        if not re.match(r'^[A-Z0-9]{1,10}$', ticker):
            raise InvalidTickerError(f"Invalid ticker format: {ticker}")
        
        return ticker
    
    @staticmethod
    def validate_price(price: float, min_price: float = 0.0) -> float:
        """Validate price value."""
        if not isinstance(price, (int, float)):
            raise ValidationError(f"Price must be numeric, got {type(price)}")
        
        if price < min_price:
            raise ValidationError(f"Price {price} is below minimum {min_price}")
        
        if price > 1e10:  # Sanity check
            raise ValidationError(f"Price {price} exceeds maximum allowed value")
        
        return float(price)
    
    @staticmethod
    def validate_date_range(
        start_date: datetime,
        end_date: datetime,
        max_days: Optional[int] = None
    ) -> tuple:
        """
        Validate date range.
        
        Args:
            start_date: Start date
            end_date: End date
            max_days: Maximum allowed days in range
            
        Returns:
            Tuple of (start_date, end_date)
        """
        if start_date >= end_date:
            raise InvalidDateRangeError(
                f"Start date {start_date} must be before end date {end_date}"
            )
        
        if max_days:
            delta = end_date - start_date
            if delta.days > max_days:
                raise InvalidDateRangeError(
                    f"Date range {delta.days} days exceeds maximum {max_days}"
                )
        
        return start_date, end_date
    
    @staticmethod
    def validate_options_data(options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate options data structure."""
        required_fields = ['strike', 'expiry', 'option_type', 'underlying']
        
        for field in required_fields:
            if field not in options_data:
                raise ValidationError(f"Missing required field: {field}")
        
        # Validate option type
        if options_data['option_type'] not in ['call', 'put', 'CALL', 'PUT']:
            raise ValidationError(
                f"Invalid option type: {options_data['option_type']}"
            )
        
        # Normalize option type
        options_data['option_type'] = options_data['option_type'].lower()
        
        return options_data
    
    @staticmethod
    def validate_percentage(value: float, name: str = "value") -> float:
        """Validate percentage value (0-100)."""
        if not 0 <= value <= 100:
            raise ValidationError(f"{name} must be between 0 and 100, got {value}")
        return value

if __name__ == "__main__":
    # Test validation
    try:
        ticker = ValidationUtils.validate_ticker("btc")
        print(f"✅ Valid ticker: {ticker}")
        
        price = ValidationUtils.validate_price(50000.50)
        print(f"✅ Valid price: {price}")
        
    except ValidationError as e:
        print(f"❌ Validation error: {e}")
