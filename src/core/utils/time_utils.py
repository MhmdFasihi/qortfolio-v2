# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Time utilities module with CRITICAL bug fix for time-to-maturity calculations.
This module fixes the mathematical error from legacy repositories.
"""

from datetime import datetime, timedelta
from typing import Union, Optional
import numpy as np
import pytz
import logging

logger = logging.getLogger(__name__)

class TimeUtils:
    """
    Time calculation utilities with corrected mathematical formulas.
    
    CRITICAL FIX: The legacy code had an error in time-to-maturity calculation:
    WRONG: time_diff.total_seconds() / 31536000 * 365
    CORRECT: time_diff.total_seconds() / (365.25 * 24 * 3600)
    """
    
    @staticmethod
    def calculate_time_to_maturity(
        current_time: Union[datetime, str],
        expiry_time: Union[datetime, str],
        annualized: bool = True
    ) -> float:
        """
        Calculate time to maturity in years (corrected formula).
        
        Args:
            current_time: Current datetime or ISO format string
            expiry_time: Expiry datetime or ISO format string
            annualized: If True, return in years; if False, return in days
            
        Returns:
            Time to maturity in years (default) or days
            
        Note:
            This function fixes the critical bug from legacy repositories.
            The correct formula accounts for leap years using 365.25 days per year.
        """
        # Convert strings to datetime if necessary
        if isinstance(current_time, str):
            current_time = datetime.fromisoformat(current_time)
        if isinstance(expiry_time, str):
            expiry_time = datetime.fromisoformat(expiry_time)
            
        # Ensure both times are timezone-aware
        if current_time.tzinfo is None:
            current_time = pytz.UTC.localize(current_time)
        if expiry_time.tzinfo is None:
            expiry_time = pytz.UTC.localize(expiry_time)
            
        # Calculate time difference
        time_diff = expiry_time - current_time
        
        if time_diff.total_seconds() <= 0:
            logger.warning(f"Expiry time {expiry_time} is before current time {current_time}")
            return 0.0
        
        if annualized:
            # CORRECT FORMULA: Account for leap years with 365.25
            seconds_per_year = 365.25 * 24 * 3600
            return time_diff.total_seconds() / seconds_per_year
        else:
            # Return in days
            return time_diff.total_seconds() / (24 * 3600)
    
    @staticmethod
    def get_business_days_between(
        start_date: datetime,
        end_date: datetime,
        holidays: Optional[list] = None
    ) -> int:
        """
        Calculate number of business days between two dates.
        
        Args:
            start_date: Start date
            end_date: End date
            holidays: List of holiday dates to exclude
            
        Returns:
            Number of business days
        """
        if holidays is None:
            holidays = []
            
        business_days = 0
        current_date = start_date
        
        while current_date <= end_date:
            if current_date.weekday() < 5 and current_date not in holidays:
                business_days += 1
            current_date += timedelta(days=1)
            
        return business_days
    
    @staticmethod
    def annualize_volatility(
        volatility: float,
        period: str = "daily"
    ) -> float:
        """
        Annualize volatility from different time periods.
        
        Args:
            volatility: Raw volatility value
            period: Time period ('daily', 'weekly', 'monthly')
            
        Returns:
            Annualized volatility
        """
        period_map = {
            "daily": np.sqrt(365.25),  # Using 365.25 for consistency
            "weekly": np.sqrt(52),
            "monthly": np.sqrt(12),
            "hourly": np.sqrt(365.25 * 24)
        }
        
        if period not in period_map:
            raise ValueError(f"Unknown period: {period}")
            
        return volatility * period_map[period]

# Test function to validate the fix
def validate_time_calculation():
    """Validate that time calculation is working correctly."""
    current = datetime(2025, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
    expiry = datetime(2025, 1, 31, 0, 0, 0, tzinfo=pytz.UTC)
    
    time_to_maturity = TimeUtils.calculate_time_to_maturity(current, expiry)
    expected = 30 / 365.25  # 30 days in years
    
    assert abs(time_to_maturity - expected) < 1e-10, \
        f"Time calculation error: got {time_to_maturity}, expected {expected}"
    
    logger.info("✅ Time calculation bug fix validated successfully!")
    return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    validate_time_calculation()
    print("Time utilities module initialized with bug fix applied.")
