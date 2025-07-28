"""
# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

Time Utilities for Qortfolio V2
CRITICAL: Fixes the time-to-maturity calculation bug from legacy code

Mathematical Bug Fixed:
- WRONG: time.total_seconds() / 31536000 * 365
- CORRECT: time.total_seconds() / (365.25 * 24 * 3600)
"""

from datetime import datetime, timedelta
from typing import Union, Optional
import pandas as pd
import numpy as np

# Constants
SECONDS_PER_YEAR = 365.25 * 24 * 3600  # 31,557,600 seconds (accounting for leap years)
DAYS_PER_YEAR = 365.25
HOURS_PER_YEAR = 365.25 * 24
MINUTES_PER_YEAR = 365.25 * 24 * 60


def calculate_time_to_maturity(
    current_time: Union[datetime, pd.Timestamp], 
    expiry_time: Union[datetime, pd.Timestamp],
    min_time: float = 1/365.25  # Minimum 1 day
) -> float:
    """
    Calculate time to maturity in years (fraction).
    
    CRITICAL FIX: Uses correct mathematical conversion
    - OLD BUG: time.total_seconds() / 31536000 * 365 (wrong!)
    - NEW FIX: time.total_seconds() / (365.25 * 24 * 3600) (correct!)
    
    Args:
        current_time: Current timestamp
        expiry_time: Option expiry timestamp  
        min_time: Minimum time to return (default: 1 day)
        
    Returns:
        Time to maturity in years (float)
        
    Examples:
        >>> from datetime import datetime
        >>> current = datetime(2024, 1, 1)
        >>> expiry = datetime(2024, 1, 31)  # 30 days later
        >>> tte = calculate_time_to_maturity(current, expiry)
        >>> round(tte, 4)  # Should be ~0.0821 years (30/365.25)
        0.0821
    """
    # Convert to pandas timestamps for consistent handling
    if isinstance(current_time, datetime):
        current_time = pd.Timestamp(current_time)
    if isinstance(expiry_time, datetime):
        expiry_time = pd.Timestamp(expiry_time)
    
    # Calculate time difference
    time_diff = expiry_time - current_time
    
    # CRITICAL FIX: Correct time conversion
    time_to_maturity_years = time_diff.total_seconds() / SECONDS_PER_YEAR
    
    # Ensure minimum time (prevent division by zero in financial calculations)
    return max(time_to_maturity_years, min_time)


def calculate_time_to_maturity_vectorized(
    current_times: Union[pd.Series, list], 
    expiry_times: Union[pd.Series, list],
    min_time: float = 1/365.25
) -> pd.Series:
    """
    Vectorized version for processing multiple options at once.
    
    Args:
        current_times: Series/list of current timestamps
        expiry_times: Series/list of expiry timestamps
        min_time: Minimum time to return
        
    Returns:
        Series of time-to-maturity values in years
    """
    current_series = pd.to_datetime(pd.Series(current_times))
    expiry_series = pd.to_datetime(pd.Series(expiry_times))
    
    # Vectorized calculation with the FIXED formula
    time_diffs = expiry_series - current_series
    time_to_maturity_years = time_diffs.dt.total_seconds() / SECONDS_PER_YEAR
    
    # Apply minimum time constraint
    return time_to_maturity_years.clip(lower=min_time)


def time_to_maturity_from_days(days: float) -> float:
    """
    Convert days to time-to-maturity in years.
    
    Args:
        days: Number of days
        
    Returns:
        Time in years (fraction)
    """
    return days / DAYS_PER_YEAR


def time_to_maturity_to_days(years: float) -> float:
    """
    Convert time-to-maturity in years to days.
    
    Args:
        years: Time in years (fraction)
        
    Returns:
        Number of days
    """
    return years * DAYS_PER_YEAR


def validate_time_calculation(
    current_time: datetime,
    expiry_time: datetime,
    expected_days: Optional[float] = None
) -> dict:
    """
    Validate time calculation against known values.
    Useful for testing the bug fix.
    
    Args:
        current_time: Current timestamp
        expiry_time: Expiry timestamp  
        expected_days: Expected number of days (optional)
        
    Returns:
        Dictionary with calculation results and validation
    """
    tte_years = calculate_time_to_maturity(current_time, expiry_time)
    tte_days = time_to_maturity_to_days(tte_years)
    
    # Manual calculation for validation
    manual_days = (expiry_time - current_time).days
    manual_years = manual_days / DAYS_PER_YEAR
    
    result = {
        'time_to_maturity_years': tte_years,
        'time_to_maturity_days': tte_days,
        'manual_calculation_days': manual_days,
        'manual_calculation_years': manual_years,
        'calculation_matches': abs(tte_years - manual_years) < 1e-6
    }
    
    if expected_days is not None:
        result['expected_days'] = expected_days
        result['matches_expected'] = abs(tte_days - expected_days) < 0.1
    
    return result


# Legacy function compatibility (with fix)
def legacy_time_calculation_fixed(time_delta: timedelta) -> float:
    """
    Fixed version of the legacy time calculation.
    
    This replaces the buggy legacy code:
    OLD: time_delta.total_seconds() / 31536000 * 365
    NEW: time_delta.total_seconds() / (365.25 * 24 * 3600)
    
    Args:
        time_delta: Time difference as timedelta
        
    Returns:
        Time in years (fraction)
    """
    return time_delta.total_seconds() / SECONDS_PER_YEAR


# Demonstration of the bug fix
def demonstrate_bug_fix():
    """
    Demonstrate the difference between old buggy and new fixed calculations.
    """
    # Example: 30 days from now
    current = datetime(2024, 1, 1)
    expiry = datetime(2024, 1, 31)  # 30 days later
    time_diff = expiry - current
    
    # OLD BUGGY CALCULATION
    old_buggy = time_diff.total_seconds() / 31536000 * 365
    
    # NEW FIXED CALCULATION  
    new_fixed = time_diff.total_seconds() / SECONDS_PER_YEAR
    
    # Expected (manual)
    expected = 30 / 365.25
    
    print("Time Calculation Bug Fix Demonstration:")
    print(f"Time period: 30 days")
    print(f"Old buggy result: {old_buggy:.6f} years")
    print(f"New fixed result: {new_fixed:.6f} years") 
    print(f"Expected result:  {expected:.6f} years")
    print(f"Old error: {abs(old_buggy - expected):.6f}")
    print(f"New error: {abs(new_fixed - expected):.6f}")
    print(f"Bug fix accuracy improvement: {(abs(old_buggy - expected) / abs(new_fixed - expected)):.1f}x better")


if __name__ == "__main__":
    # Run demonstration
    demonstrate_bug_fix()
    
    # Test basic functionality
    current = datetime(2024, 1, 1)
    expiry = datetime(2024, 7, 1)  # 6 months later
    tte = calculate_time_to_maturity(current, expiry)
    print(f"\nTest: 6 months = {tte:.4f} years (expected ~0.5)")