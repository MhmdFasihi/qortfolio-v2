"""
# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

Comprehensive tests for time_utils.py
CRITICAL: Validates the time-to-maturity calculation bug fix

These tests ensure the mathematical fix is correct and prevents regression.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Try multiple import approaches
try:
    # Method 1: Direct import (if package installed correctly)
    from core.utils.time_utils import (
        calculate_time_to_maturity,
        calculate_time_to_maturity_vectorized,
        time_to_maturity_from_days,
        time_to_maturity_to_days,
        validate_time_calculation,
        legacy_time_calculation_fixed,
        SECONDS_PER_YEAR
    )
except ImportError:
    try:
        # Method 2: Import from src prefix
        from src.core.utils.time_utils import (
            calculate_time_to_maturity,
            calculate_time_to_maturity_vectorized,
            time_to_maturity_from_days,
            time_to_maturity_to_days,
            validate_time_calculation,
            legacy_time_calculation_fixed,
            SECONDS_PER_YEAR
        )
    except ImportError:
        # Method 3: Local import
        import core.utils.time_utils as time_utils
        calculate_time_to_maturity = time_utils.calculate_time_to_maturity
        calculate_time_to_maturity_vectorized = time_utils.calculate_time_to_maturity_vectorized
        time_to_maturity_from_days = time_utils.time_to_maturity_from_days
        time_to_maturity_to_days = time_utils.time_to_maturity_to_days
        validate_time_calculation = time_utils.validate_time_calculation
        legacy_time_calculation_fixed = time_utils.legacy_time_calculation_fixed
        SECONDS_PER_YEAR = time_utils.SECONDS_PER_YEAR


class TestCriticalBugFix:
    """Test the critical time calculation bug fix."""
    
    def test_bug_fix_accuracy(self):
        """Test that the bug fix produces correct results."""
        # Test case: exactly 30 days
        current = datetime(2024, 1, 1)
        expiry = datetime(2024, 1, 31)
        
        result = calculate_time_to_maturity(current, expiry)
        expected = 30 / 365.25  # Correct calculation
        
        # Should be accurate to 6 decimal places
        assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"
    
    def test_old_vs_new_calculation(self):
        """Demonstrate the difference between old buggy and new fixed calculations."""
        time_diff = timedelta(days=30)
        
        # OLD BUGGY CALCULATION (what was wrong)
        old_buggy = time_diff.total_seconds() / 31536000 * 365
        
        # NEW FIXED CALCULATION
        new_fixed = legacy_time_calculation_fixed(time_diff)
        
        # Expected correct result
        expected = 30 / 365.25
        
        # New calculation should be much more accurate
        old_error = abs(old_buggy - expected)
        new_error = abs(new_fixed - expected)
        
        assert new_error < old_error / 100, "Bug fix should be much more accurate"
        assert new_error < 1e-10, "New calculation should be nearly perfect"
    
    def test_known_time_periods(self):
        """Test against known time periods."""
        current = datetime(2024, 1, 1)
        
        test_cases = [
            (datetime(2024, 1, 2), 1, "1 day"),
            (datetime(2024, 1, 8), 7, "1 week"),
            (datetime(2024, 1, 31), 30, "30 days"),
            (datetime(2024, 4, 1), 91, "1 quarter"),
            (datetime(2024, 7, 1), 182, "6 months"),
            (datetime(2025, 1, 1), 366, "1 year (leap year)")
        ]
        
        for expiry, expected_days, description in test_cases:
            result = calculate_time_to_maturity(current, expiry)
            expected_years = expected_days / 365.25
            
            assert abs(result - expected_years) < 0.001, \
                f"{description}: Expected {expected_years:.6f}, got {result:.6f}"


class TestTimeToMaturityCalculation:
    """Test the main time-to-maturity calculation function."""
    
    def test_basic_calculation(self):
        """Test basic time-to-maturity calculation."""
        current = datetime(2024, 1, 1, 12, 0, 0)
        expiry = datetime(2024, 1, 1, 18, 0, 0)  # 6 hours later
        
        result = calculate_time_to_maturity(current, expiry, min_time=0)
        expected = 6 / (365.25 * 24)  # 6 hours in years
        
        assert abs(result - expected) < 1e-8
    
    def test_minimum_time_constraint(self):
        """Test minimum time constraint."""
        current = datetime(2024, 1, 1, 12, 0, 0)
        expiry = datetime(2024, 1, 1, 12, 30, 0)  # 30 minutes later
        
        min_time = 1/365.25  # 1 day minimum
        result = calculate_time_to_maturity(current, expiry, min_time)
        
        assert result == min_time, "Should enforce minimum time"
    
    def test_negative_time_handling(self):
        """Test handling of expired options."""
        current = datetime(2024, 1, 1, 12, 0, 0)
        expiry = datetime(2024, 1, 1, 6, 0, 0)  # 6 hours ago
        
        min_time = 1/365.25
        result = calculate_time_to_maturity(current, expiry, min_time)
        
        assert result == min_time, "Expired options should return minimum time"
    
    def test_pandas_timestamp_compatibility(self):
        """Test compatibility with pandas timestamps."""
        current = pd.Timestamp('2024-01-01 12:00:00')
        expiry = pd.Timestamp('2024-01-02 12:00:00')  # 1 day later
        
        result = calculate_time_to_maturity(current, expiry)
        expected = 1 / 365.25
        
        assert abs(result - expected) < 1e-8


class TestVectorizedCalculation:
    """Test vectorized time-to-maturity calculation."""
    
    def test_vectorized_basic(self):
        """Test basic vectorized calculation."""
        current_times = [datetime(2024, 1, 1)] * 3
        expiry_times = [
            datetime(2024, 1, 2),   # 1 day
            datetime(2024, 1, 8),   # 7 days  
            datetime(2024, 1, 31)   # 30 days
        ]
        
        results = calculate_time_to_maturity_vectorized(current_times, expiry_times)
        expected = pd.Series([1/365.25, 7/365.25, 30/365.25])
        
        pd.testing.assert_series_equal(results, expected, atol=1e-8)
    
    def test_vectorized_with_pandas_series(self):
        """Test vectorized calculation with pandas Series."""
        current_times = pd.Series([datetime(2024, 1, 1)] * 2)
        expiry_times = pd.Series([datetime(2024, 1, 15), datetime(2024, 2, 1)])
        
        results = calculate_time_to_maturity_vectorized(current_times, expiry_times)
        
        assert len(results) == 2
        assert all(results > 0)


class TestUtilityFunctions:
    """Test utility conversion functions."""
    
    def test_days_to_years_conversion(self):
        """Test days to years conversion."""
        test_cases = [
            (1, 1/365.25),
            (30, 30/365.25),
            (365.25, 1.0),
            (730.5, 2.0)
        ]
        
        for days, expected_years in test_cases:
            result = time_to_maturity_from_days(days)
            assert abs(result - expected_years) < 1e-10
    
    def test_years_to_days_conversion(self):
        """Test years to days conversion."""
        test_cases = [
            (1.0, 365.25),
            (0.5, 365.25/2),
            (2.0, 365.25*2)
        ]
        
        for years, expected_days in test_cases:
            result = time_to_maturity_to_days(years)
            assert abs(result - expected_days) < 1e-10
    
    def test_round_trip_conversion(self):
        """Test that days->years->days conversion is accurate."""
        original_days = [1, 7, 30, 90, 365.25]
        
        for days in original_days:
            years = time_to_maturity_from_days(days)
            back_to_days = time_to_maturity_to_days(years)
            assert abs(back_to_days - days) < 1e-10


class TestValidationFunction:
    """Test the validation function."""
    
    def test_validation_basic(self):
        """Test basic validation functionality."""
        current = datetime(2024, 1, 1)
        expiry = datetime(2024, 1, 31)  # 30 days
        
        result = validate_time_calculation(current, expiry, expected_days=30)
        
        assert result['calculation_matches'] == True
        assert result['matches_expected'] == True
        assert abs(result['time_to_maturity_days'] - 30) < 0.1


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_short_time_periods(self):
        """Test very short time periods."""
        current = datetime(2024, 1, 1, 12, 0, 0)
        expiry = datetime(2024, 1, 1, 12, 0, 1)  # 1 second later
        
        result = calculate_time_to_maturity(current, expiry, min_time=0)
        
        # Should be greater than 0 but very small
        assert result > 0
        assert result < 1e-6
    
    def test_leap_year_handling(self):
        """Test leap year handling."""
        # Test during a leap year
        current = datetime(2024, 1, 1)  # 2024 is a leap year
        expiry = datetime(2024, 12, 31)
        
        result = calculate_time_to_maturity(current, expiry)
        
        # Should be close to 1 year
        assert 0.99 < result < 1.01
    
    def test_year_boundary_crossing(self):
        """Test calculations that cross year boundaries."""
        current = datetime(2024, 11, 1)
        expiry = datetime(2025, 2, 1)  # 3 months later, crossing year
        
        result = calculate_time_to_maturity(current, expiry)
        expected_days = (expiry - current).days
        expected_years = expected_days / 365.25
        
        assert abs(result - expected_years) < 0.001


class TestConstants:
    """Test that our constants are correct."""
    
    def test_seconds_per_year(self):
        """Test that SECONDS_PER_YEAR is calculated correctly."""
        expected = 365.25 * 24 * 3600
        assert SECONDS_PER_YEAR == expected
    
    def test_constant_consistency(self):
        """Test that all time constants are consistent."""
        from core.utils.time_utils import DAYS_PER_YEAR, HOURS_PER_YEAR, MINUTES_PER_YEAR
        
        assert HOURS_PER_YEAR == DAYS_PER_YEAR * 24
        assert MINUTES_PER_YEAR == HOURS_PER_YEAR * 60
        assert SECONDS_PER_YEAR == MINUTES_PER_YEAR * 60


# Financial impact test
class TestFinancialImpact:
    """Test the financial impact of the bug fix."""
    
    def test_options_pricing_impact(self):
        """Test how the bug fix affects options pricing calculations."""
        # Simulate an option with 30 days to expiry
        current = datetime(2024, 1, 1)
        expiry = datetime(2024, 1, 31)
        time_diff = expiry - current
        
        # OLD BUGGY TIME CALCULATION
        old_tte = time_diff.total_seconds() / 31536000 * 365
        
        # NEW FIXED TIME CALCULATION
        new_tte = calculate_time_to_maturity(current, expiry)
        
        # The bug would cause significant pricing errors
        relative_error = abs(old_tte - new_tte) / new_tte
        
        # Error should be significant (demonstrating importance of fix)
        assert relative_error > 0.01, "Bug fix should have measurable impact"
        
        print(f"Time-to-maturity calculation error from bug: {relative_error:.2%}")


if __name__ == "__main__":
    # Run key tests if executed directly
    pytest.main([__file__, "-v"])