# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
src/core/utils/time_utils.py
Crypto-Focused Time Utilities - 24/7 Market Time Calculations

CRITICAL: This module fixes the time-to-maturity calculation bug from legacy code.
OLD BUG: time.total_seconds() / 31536000 * 365  # Mathematically wrong!
CORRECT: time.total_seconds() / (365.25 * 24 * 3600)  # Proper conversion

Note: Crypto markets operate 24/7/365, so all time calculations are calendar-based.
"""

from datetime import datetime, timedelta, timezone
from typing import Union, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Constants for 24/7 crypto market time calculations
SECONDS_PER_YEAR = 365.25 * 24 * 3600  # Accounts for leap years, 24/7 market
DAYS_PER_YEAR = 365.25
HOURS_PER_YEAR = DAYS_PER_YEAR * 24
MINUTES_PER_YEAR = HOURS_PER_YEAR * 60
SECONDS_PER_DAY = 24 * 3600
HOURS_PER_DAY = 24

# Minimum time thresholds for options (crypto markets never close)
MIN_TIME_HOURS = 1  # Minimum 1 hour to expiry
MIN_TIME_YEARS = MIN_TIME_HOURS / HOURS_PER_YEAR


class TimeCalculationError(Exception):
    """Exception raised when time calculation fails."""
    pass


def calculate_time_to_maturity(
    current_time: Union[datetime, str], 
    expiry_time: Union[datetime, str],
    min_time: Optional[float] = None
) -> float:
    """
    Calculate time to maturity in years for 24/7 crypto markets.
    
    CRITICAL FIX: This replaces the legacy bug where time calculation was wrong.
    OLD BUG: time.total_seconds() / 31536000 * 365  # Mathematically incorrect!
    CORRECT: time.total_seconds() / (365.25 * 24 * 3600)  # Proper conversion
    
    Args:
        current_time: Current datetime or ISO string
        expiry_time: Expiration datetime or ISO string
        min_time: Minimum time to return in years (default: 1 hour)
        
    Returns:
        float: Time to maturity in years (accurate to seconds)
        
    Raises:
        TimeCalculationError: If calculation fails
        
    Example:
        >>> current = datetime(2025, 1, 1, 12, 0, 0)
        >>> expiry = datetime(2025, 7, 1, 12, 0, 0)  # 6 months later
        >>> ttm = calculate_time_to_maturity(current, expiry)
        >>> round(ttm, 4)  # Should be ~0.5 years
        0.4956
    """
    try:
        # Convert to datetime if strings
        if isinstance(current_time, str):
            current_time = pd.to_datetime(current_time)
        if isinstance(expiry_time, str):
            expiry_time = pd.to_datetime(expiry_time)
        
        # Ensure timezone consistency
        if current_time.tzinfo is not None and expiry_time.tzinfo is None:
            expiry_time = expiry_time.replace(tzinfo=current_time.tzinfo)
        elif current_time.tzinfo is None and expiry_time.tzinfo is not None:
            current_time = current_time.replace(tzinfo=expiry_time.tzinfo)
            
        # Calculate time difference
        time_diff = expiry_time - current_time
        
        # CRITICAL FIX: Use correct conversion to years (24/7 market)
        time_to_maturity = time_diff.total_seconds() / SECONDS_PER_YEAR
        
        # Apply minimum time constraint (default: 1 hour for crypto markets)
        if min_time is None:
            min_time = MIN_TIME_YEARS
            
        return max(time_to_maturity, min_time)
        
    except Exception as e:
        logger.error(f"Time calculation failed: {e}")
        raise TimeCalculationError(f"Failed to calculate time to maturity: {e}")


def calculate_time_to_maturity_vectorized(
    time_differences: pd.Series,
    min_time: Optional[float] = None
) -> pd.Series:
    """
    Vectorized time-to-maturity calculation for DataFrames (24/7 crypto markets).
    
    CRITICAL FIX: This replaces the legacy buggy lambda function:
    OLD BUG: lambda x: max(round(x.total_seconds() / 31536000, 3), 1e-4) * 365
    CORRECT: Use proper seconds per year calculation for 24/7 markets
    
    Args:
        time_differences: Series of timedelta objects
        min_time: Minimum time in years (default: 1 hour)
        
    Returns:
        Series of time-to-maturity values in years
        
    Example:
        >>> # Fix the legacy bug in option processing
        >>> option_data["time_to_maturity"] = calculate_time_to_maturity_vectorized(
        ...     option_data["maturity_date"] - option_data["date_time"]
        ... )
    """
    try:
        if min_time is None:
            min_time = MIN_TIME_YEARS
            
        # CRITICAL FIX: Use correct time conversion for 24/7 markets
        time_to_maturity = time_differences.dt.total_seconds() / SECONDS_PER_YEAR
        
        # Apply minimum time constraint
        return np.maximum(time_to_maturity, min_time)
        
    except Exception as e:
        logger.error(f"Vectorized time calculation failed: {e}")
        raise TimeCalculationError(f"Vectorized time calculation failed: {e}")


def fix_legacy_time_calculation(df: pd.DataFrame, 
                               current_time_col: str = 'date_time',
                               expiry_time_col: str = 'maturity_date',
                               output_col: str = 'time_to_maturity') -> pd.DataFrame:
    """
    Fix legacy time calculation bugs in existing DataFrames for crypto markets.
    
    This function replaces the problematic legacy calculation:
    OLD BUG: lambda x: max(round(x.total_seconds() / 31536000, 3), 1e-4) * 365
    
    Args:
        df: DataFrame with time columns
        current_time_col: Column name for current time
        expiry_time_col: Column name for expiry time  
        output_col: Column name for output time-to-maturity
        
    Returns:
        DataFrame with fixed time calculations
    """
    try:
        logger.info(f"Fixing legacy time calculations for {len(df)} crypto options records")
        
        # Calculate time differences
        time_diff = df[expiry_time_col] - df[current_time_col]
        
        # Apply fixed calculation for 24/7 crypto markets
        df[output_col] = calculate_time_to_maturity_vectorized(time_diff)
        
        logger.info(f"✅ Fixed {len(df)} crypto options time calculations")
        return df
        
    except Exception as e:
        logger.error(f"Failed to fix legacy time calculations: {e}")
        raise TimeCalculationError(f"Legacy fix failed: {e}")


def validate_time_calculation(ttm_calculated: float, 
                            current_time: datetime,
                            expiry_time: datetime,
                            tolerance: float = 1e-6) -> bool:
    """
    Validate time-to-maturity calculation against known correct result.
    
    Args:
        ttm_calculated: Calculated time-to-maturity
        current_time: Current datetime
        expiry_time: Expiry datetime
        tolerance: Acceptable error tolerance
        
    Returns:
        True if calculation is accurate within tolerance
    """
    try:
        # Calculate expected result manually
        time_diff = expiry_time - current_time
        expected_ttm = time_diff.total_seconds() / SECONDS_PER_YEAR
        
        # Apply minimum time constraint
        expected_ttm = max(expected_ttm, MIN_TIME_YEARS)
        
        error = abs(ttm_calculated - expected_ttm)
        is_valid = error < tolerance
        
        if not is_valid:
            logger.warning(f"Time calculation validation failed: error={error}, tolerance={tolerance}")
            
        return is_valid
        
    except Exception as e:
        logger.error(f"Time validation failed: {e}")
        return False


def calculate_days_to_expiry(
    current_time: Union[datetime, str],
    expiry_time: Union[datetime, str]
) -> float:
    """
    Calculate days to expiration for crypto options (24/7 market).
    
    Args:
        current_time: Current datetime or ISO string
        expiry_time: Expiration datetime or ISO string
        
    Returns:
        float: Days to expiration (can be fractional for crypto's 24/7 nature)
    """
    try:
        if isinstance(current_time, str):
            current_time = pd.to_datetime(current_time)
        if isinstance(expiry_time, str):
            expiry_time = pd.to_datetime(expiry_time)
        
        time_diff = expiry_time - current_time
        days_to_expiry = time_diff.total_seconds() / SECONDS_PER_DAY
        
        # Minimum 1 hour in days
        min_days = MIN_TIME_HOURS / HOURS_PER_DAY
        return max(days_to_expiry, min_days)
        
    except Exception as e:
        logger.error(f"Days calculation failed: {e}")
        return MIN_TIME_HOURS / HOURS_PER_DAY


def calculate_hours_to_expiry(
    current_time: Union[datetime, str],
    expiry_time: Union[datetime, str]
) -> float:
    """
    Calculate hours to expiration for crypto options (useful for short-term options).
    
    Args:
        current_time: Current datetime or ISO string
        expiry_time: Expiration datetime or ISO string
        
    Returns:
        float: Hours to expiration
    """
    try:
        if isinstance(current_time, str):
            current_time = pd.to_datetime(current_time)
        if isinstance(expiry_time, str):
            expiry_time = pd.to_datetime(expiry_time)
        
        time_diff = expiry_time - current_time
        hours_to_expiry = time_diff.total_seconds() / 3600
        
        return max(hours_to_expiry, MIN_TIME_HOURS)
        
    except Exception as e:
        logger.error(f"Hours calculation failed: {e}")
        return MIN_TIME_HOURS


def get_current_crypto_time() -> datetime:
    """
    Get current time for crypto markets (UTC-based since crypto is global 24/7).
    
    Returns:
        datetime: Current UTC time
    """
    return datetime.now(timezone.utc)


def test_time_calculation_fix():
    """
    Test function to validate that our fix produces correct results for crypto markets.
    """
    print("🧪 Testing Crypto Time Calculation Fix...")
    
    test_cases = [
        # (current, expiry, expected_days, description)
        (datetime(2025, 1, 1, 12, 0, 0), datetime(2025, 7, 1, 12, 0, 0), 181, "6 months"),
        (datetime(2025, 1, 1, 12, 0, 0), datetime(2026, 1, 1, 12, 0, 0), 365, "1 year"),
        (datetime(2025, 1, 1, 12, 0, 0), datetime(2025, 1, 31, 12, 0, 0), 30, "30 days"),
        (datetime(2025, 1, 1, 12, 0, 0), datetime(2025, 1, 8, 12, 0, 0), 7, "1 week"),
        (datetime(2025, 1, 1, 12, 0, 0), datetime(2025, 1, 2, 12, 0, 0), 1, "1 day"),
        (datetime(2025, 1, 1, 12, 0, 0), datetime(2025, 1, 1, 18, 0, 0), 0.25, "6 hours"),
    ]
    
    all_passed = True
    
    for current, expiry, expected_days, description in test_cases:
        ttm = calculate_time_to_maturity(current, expiry)
        expected_years = expected_days / 365.25
        error = abs(ttm - expected_years)
        
        if error < 1e-6:
            print(f"  ✅ {description}: {ttm:.6f} years (expected {expected_years:.6f})")
        else:
            print(f"  ❌ {description}: {ttm:.6f} years (expected {expected_years:.6f}) - Error: {error:.2e}")
            all_passed = False
    
    # Test the legacy bug comparison
    print("\n🔍 Legacy Bug Comparison (30-day example):")
    time_diff = timedelta(days=30)
    
    # OLD BUGGY CALCULATION (what was wrong)
    old_buggy = time_diff.total_seconds() / 31536000 * 365
    
    # NEW FIXED CALCULATION
    new_fixed = time_diff.total_seconds() / SECONDS_PER_YEAR
    
    expected = 30 / 365.25
    old_error = abs(old_buggy - expected)
    new_error = abs(new_fixed - expected)
    
    print(f"  Old buggy result: {old_buggy:.6f} (error: {old_error:.2e})")
    print(f"  New fixed result: {new_fixed:.6f} (error: {new_error:.2e})")
    print(f"  Expected result:  {expected:.6f}")
    
    # Handle perfect accuracy (division by zero)
    if new_error > 1e-15:  # Avoid division by zero for perfect accuracy
        improvement = old_error / new_error
        print(f"  Improvement: {improvement:.0f}x more accurate")
    else:
        print("  Improvement: Perfect accuracy achieved!")
    
    # Test crypto-specific features
    print("\n⏰ Crypto Market Features:")
    current = datetime(2025, 1, 1, 12, 0, 0)
    
    # Test short-term crypto options (hours)
    expiry_6h = datetime(2025, 1, 1, 18, 0, 0)
    hours_to_exp = calculate_hours_to_expiry(current, expiry_6h)
    print(f"  ✅ 6-hour option: {hours_to_exp:.1f} hours to expiry")
    
    # Test fractional days
    expiry_1_5d = datetime(2025, 1, 2, 24, 0, 0)  # 1.5 days
    days_to_exp = calculate_days_to_expiry(current, expiry_1_5d)
    print(f"  ✅ 1.5-day option: {days_to_exp:.2f} days to expiry")
    
    # Test current crypto time
    crypto_time = get_current_crypto_time()
    print(f"  ✅ Current crypto time (UTC): {crypto_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    if all_passed:
        print("\n✅ All crypto time calculation tests PASSED! Bug fix validated for 24/7 markets.")
        return True
    else:
        print("\n❌ Some time calculation tests FAILED!")
        return False


# Utility functions for crypto options
def is_option_expired(expiry_time: Union[datetime, str], 
                     current_time: Optional[Union[datetime, str]] = None) -> bool:
    """
    Check if a crypto option has expired (24/7 market).
    
    Args:
        expiry_time: Option expiry time
        current_time: Current time (default: now UTC)
        
    Returns:
        bool: True if option has expired
    """
    if current_time is None:
        current_time = get_current_crypto_time()
    
    if isinstance(expiry_time, str):
        expiry_time = pd.to_datetime(expiry_time)
    if isinstance(current_time, str):
        current_time = pd.to_datetime(current_time)
        
    return current_time >= expiry_time


def get_time_decay_rate(time_to_maturity: float) -> float:
    """
    Calculate time decay rate for crypto options (theta approximation).
    
    Args:
        time_to_maturity: Time to maturity in years
        
    Returns:
        float: Approximate daily time decay rate
    """
    if time_to_maturity <= 0:
        return 0.0
    
    # Simple theta approximation: higher decay as expiry approaches
    # For crypto's 24/7 nature, decay is continuous
    return 1.0 / (time_to_maturity * 365.25)


if __name__ == "__main__":
    # Run tests when executed directly
    test_time_calculation_fix()