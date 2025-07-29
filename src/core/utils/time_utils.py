# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Fixed Time Utilities - Corrects Critical Mathematical Bug
Location: src/core/utils/time_utils.py

This module fixes the critical time-to-maturity calculation bug from legacy code:
WRONG: time.total_seconds() / 31536000 * 365
CORRECT: time.total_seconds() / (365.25 * 24 * 3600)
"""

from datetime import datetime, timezone, timedelta
from typing import Union, Optional
import logging
from ..logging import get_logger

logger = get_logger("time_utils")

def calculate_time_to_maturity(current_time: datetime, expiry_time: datetime) -> float:
    """
    Calculate time to maturity in years - CORRECTED VERSION.
    
    This function fixes the critical mathematical bug from the legacy qortfolio code.
    
    Args:
        current_time: Current datetime (timezone-aware preferred)
        expiry_time: Option expiry datetime (timezone-aware preferred)
        
    Returns:
        Time to expiry in years (float)
        
    Note:
        CORRECT formula: total_seconds() / (365.25 * 24 * 3600)
        NOT the legacy bug: total_seconds() / 31536000 * 365
        
    Examples:
        >>> from datetime import datetime, timezone, timedelta
        >>> current = datetime(2025, 1, 29, tzinfo=timezone.utc)
        >>> expiry = datetime(2025, 7, 29, tzinfo=timezone.utc)  # 6 months
        >>> time_to_expiry = calculate_time_to_maturity(current, expiry)
        >>> abs(time_to_expiry - 0.5) < 0.01  # Should be ~0.5 years
        True
    """
    try:
        # Ensure timezone awareness
        if expiry_time.tzinfo is None:
            expiry_time = expiry_time.replace(tzinfo=timezone.utc)
            logger.debug("Added UTC timezone to expiry_time")
            
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
            logger.debug("Added UTC timezone to current_time")
        
        # Calculate time difference
        time_diff = expiry_time - current_time
        
        # Handle expired options
        if time_diff.total_seconds() <= 0:
            logger.debug(f"Option expired: expiry={expiry_time}, current={current_time}")
            return 0.0
        
        # CORRECT CALCULATION - fixes the legacy mathematical bug
        # Uses 365.25 days per year to account for leap years
        seconds_per_year = 365.25 * 24 * 3600
        time_to_expiry_years = time_diff.total_seconds() / seconds_per_year
        
        logger.debug(f"Time to expiry: {time_to_expiry_years:.6f} years "
                    f"({time_diff.days} days, {time_diff.seconds} seconds)")
        
        return time_to_expiry_years
        
    except Exception as e:
        logger.error(f"Error calculating time to maturity: {e}")
        logger.error(f"current_time: {current_time}, expiry_time: {expiry_time}")
        return 0.0

def calculate_time_to_maturity_days(current_time: datetime, expiry_time: datetime) -> float:
    """
    Calculate time to maturity in days.
    
    Args:
        current_time: Current datetime
        expiry_time: Option expiry datetime
        
    Returns:
        Time to expiry in days (float)
    """
    try:
        years = calculate_time_to_maturity(current_time, expiry_time)
        return years * 365.25
    except Exception as e:
        logger.error(f"Error calculating time to maturity in days: {e}")
        return 0.0

def get_current_time() -> datetime:
    """Get current UTC time with timezone awareness."""
    return datetime.now(timezone.utc)

def get_nearest_expiry_date(expiry_dates: list, current_time: Optional[datetime] = None) -> Optional[datetime]:
    """
    Get the nearest future expiry date from a list.
    
    Args:
        expiry_dates: List of expiry datetime objects
        current_time: Current time (defaults to now)
        
    Returns:
        Nearest future expiry date or None
    """
    try:
        if not expiry_dates:
            return None
            
        if current_time is None:
            current_time = get_current_time()
        
        # Filter for future dates only
        future_dates = [d for d in expiry_dates if d > current_time]
        
        if not future_dates:
            return None
            
        # Return the nearest one
        return min(future_dates)
        
    except Exception as e:
        logger.error(f"Error finding nearest expiry date: {e}")
        return None

def get_default_expiry_dates(base_date: Optional[datetime] = None, num_expiries: int = 6) -> list:
    """
    Generate default expiry dates (monthly options pattern).
    
    Args:
        base_date: Base date (defaults to current time)
        num_expiries: Number of expiry dates to generate
        
    Returns:
        List of expiry datetime objects
    """
    try:
        if base_date is None:
            base_date = get_current_time()
            
        expiries = []
        
        for i in range(1, num_expiries + 1):
            # Add monthly expiries (3rd Friday of each month approximation)
            expiry = base_date + timedelta(days=30 * i)
            
            # Adjust to 3rd Friday (approximate)
            # Move to Friday (weekday 4)
            days_to_friday = (4 - expiry.weekday()) % 7
            expiry = expiry + timedelta(days=days_to_friday)
            
            # Set to typical option expiry time (4 PM UTC)
            expiry = expiry.replace(hour=16, minute=0, second=0, microsecond=0)
            
            expiries.append(expiry)
        
        return expiries
        
    except Exception as e:
        logger.error(f"Error generating default expiry dates: {e}")
        return []

def parse_expiry_string(expiry_str: str, base_year: Optional[int] = None) -> Optional[datetime]:
    """
    Parse various expiry string formats.
    
    Args:
        expiry_str: Expiry string (e.g., "29JUL25", "2025-07-29")
        base_year: Base year for 2-digit years
        
    Returns:
        Parsed datetime or None
    """
    try:
        if base_year is None:
            base_year = get_current_time().year
        
        # Handle Deribit format (e.g., "29JUL25")
        if len(expiry_str) == 7 and expiry_str[2:5].isalpha():
            month_map = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
            }
            
            day = int(expiry_str[:2])
            month_str = expiry_str[2:5].upper()
            year_str = expiry_str[5:7]
            
            if month_str in month_map:
                month = month_map[month_str]
                
                # Handle 2-digit year
                year = int(year_str)
                if year < 50:  # Assume 2000s
                    year += 2000
                else:  # Assume 1900s (shouldn't happen for options)
                    year += 1900
                
                return datetime(year, month, day, 16, 0, 0, tzinfo=timezone.utc)
        
        # Handle ISO format (e.g., "2025-07-29")
        if '-' in expiry_str:
            return datetime.fromisoformat(expiry_str.replace('Z', '+00:00'))
        
        return None
        
    except Exception as e:
        logger.error(f"Error parsing expiry string '{expiry_str}': {e}")
        return None

def validate_time_calculation():
    """
    Validate time calculation with known test cases.
    
    This function tests the corrected time calculation against known values
    to ensure the mathematical bug has been fixed.
    
    Returns:
        True if all validations pass
    """
    try:
        logger.info("Validating time calculation fix...")
        
        # Test case 1: Exactly 1 year
        current = datetime(2025, 1, 29, 12, 0, 0, tzinfo=timezone.utc)
        expiry_1_year = datetime(2026, 1, 29, 12, 0, 0, tzinfo=timezone.utc)
        
        time_1_year = calculate_time_to_maturity(current, expiry_1_year)
        expected_1_year = 1.0
        
        if abs(time_1_year - expected_1_year) > 0.01:
            logger.error(f"1-year test failed: got {time_1_year}, expected {expected_1_year}")
            return False
        
        # Test case 2: Exactly 6 months
        expiry_6_months = datetime(2025, 7, 29, 12, 0, 0, tzinfo=timezone.utc)
        time_6_months = calculate_time_to_maturity(current, expiry_6_months)
        expected_6_months = 0.5
        
        if abs(time_6_months - expected_6_months) > 0.01:
            logger.error(f"6-months test failed: got {time_6_months}, expected {expected_6_months}")
            return False
        
        # Test case 3: Exactly 30 days
        expiry_30_days = current + timedelta(days=30)
        time_30_days = calculate_time_to_maturity(current, expiry_30_days)
        expected_30_days = 30.0 / 365.25
        
        if abs(time_30_days - expected_30_days) > 0.001:
            logger.error(f"30-days test failed: got {time_30_days}, expected {expected_30_days}")
            return False
        
        # Test case 4: Expired option
        expired_expiry = current - timedelta(days=1)
        time_expired = calculate_time_to_maturity(current, expired_expiry)
        
        if time_expired != 0.0:
            logger.error(f"Expired option test failed: got {time_expired}, expected 0.0")
            return False
        
        logger.info("✅ All time calculation validations passed!")
        logger.info("✅ Mathematical bug from legacy code has been FIXED")
        return True
        
    except Exception as e:
        logger.error(f"Time calculation validation failed: {e}")
        return False

# Backward compatibility (deprecated - use calculate_time_to_maturity)
def time_to_expiry(current_time: datetime, expiry_time: datetime) -> float:
    """Deprecated: Use calculate_time_to_maturity instead."""
    logger.warning("time_to_expiry is deprecated, use calculate_time_to_maturity")
    return calculate_time_to_maturity(current_time, expiry_time)

if __name__ == "__main__":
    # Run validation when script is executed directly
    validate_time_calculation()