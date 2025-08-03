# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
tests/test_time_calculation_fix.py
Comprehensive Tests for Time Calculation Bug Fix

These tests validate that the time calculation bug has been properly fixed
and that the new implementation is mathematically correct.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, timezone
from unittest.mock import patch
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.utils.time_utils import (
    calculate_time_to_maturity,
    calculate_time_to_maturity_vectorized,
    fix_legacy_time_calculation,
    validate_time_calculation,
    get_business_days_to_maturity,
    SECONDS_PER_YEAR,
    test_time_calculation_fix
)


class TestTimeCalculationFix:
    """Test the core time calculation fix."""
    
    def test_basic_time_calculation(self):
        """Test basic time-to-maturity calculation."""
        current = datetime(2025, 1, 1, 12, 0, 0)
        expiry = datetime(2025, 1, 31, 12, 0, 0)  # 30 days later
        
        result = calculate_time_to_maturity(current, expiry)
        expected = 30 / 365.25  # 30 days in years
        
        assert abs(result - expected) < 1e-6, f"Expected {expected:.6f}, got {result:.6f}"
    
    def test_half_year_calculation(self):
        """Test 6-month calculation."""
        current = datetime(2025, 1, 1, 12, 0, 0)
        expiry = datetime(2025, 7, 1, 12, 0, 0)  # 6 months later
        
        result = calculate_time_to_maturity(current, expiry)
        expected_days = (expiry - current).days
        expected = expected_days / 365.25
        
        assert abs(result - expected) < 1e-6
    
    def test_one_year_calculation(self):
        """Test exactly one year calculation."""
        current = datetime(2025, 1, 1, 12, 0, 0)
        expiry = datetime(2026, 1, 1, 12, 0, 0)  # 1 year later
        
        result = calculate_time_to_maturity(current, expiry)
        expected = 1.0  # Should be exactly 1 year
        
        assert abs(result - expected) < 1e-6
    
    def test_leap_year_calculation(self):
        """Test calculations during a leap year."""
        current = datetime(2024, 1, 1, 12, 0, 0)  # 2024 is a leap year
        expiry = datetime(2024, 12, 31, 12, 0, 0)  # End of leap year
        
        result = calculate_time_to_maturity(current, expiry)
        # Should be close to 1 year
        assert 0.99 < result < 1.01
    
    def test_string_datetime_inputs(self):
        """Test that string datetime inputs work correctly."""
        current = "2025-01-01 12:00:00"
        expiry = "2025-01-31 12:00:00"
        
        result = calculate_time_to_maturity(current, expiry)
        expected = 30 / 365.25
        
        assert abs(result - expected) < 1e-6


class TestLegacyBugComparison:
    """Test that demonstrates the bug fix improvement."""
    
    def test_old_vs_new_calculation(self):
        """Demonstrate the difference between old buggy and new fixed calculations."""
        time_diff = timedelta(days=30)
        
        # OLD BUGGY CALCULATION (what was wrong)
        old_buggy = time_diff.total_seconds() / 31536000 * 365
        
        # NEW FIXED CALCULATION
        new_fixed = time_diff.total_seconds() / SECONDS_PER_YEAR
        
        # Expected correct result
        expected = 30 / 365.25
        
        # Calculate errors
        old_error = abs(old_buggy - expected)
        new_error = abs(new_fixed - expected)
        
        # New calculation should be much more accurate
        assert new_error < old_error / 100, "Bug fix should be much more accurate"
        assert new_error < 1e-10, "New calculation should be nearly perfect"
    
    def test_bug_impact_on_options_pricing(self):
        """Test the financial impact of the bug fix on options pricing."""
        # Simulate an option with 30 days to expiry
        time_diff = timedelta(days=30)
        
        # OLD BUGGY TIME CALCULATION
        old_tte = time_diff.total_seconds() / 31536000 * 365
        
        # NEW FIXED TIME CALCULATION
        new_tte = time_diff.total_seconds() / SECONDS_PER_YEAR
        
        # Calculate relative error
        relative_error = abs(old_tte - new_tte) / new_tte
        
        # The bug should cause significant pricing errors
        assert relative_error > 0.01, "Bug fix should have measurable impact"
        
        print(f"Time-to-maturity calculation error from bug: {relative_error:.2%}")


class TestVectorizedCalculations:
    """Test vectorized time calculations for DataFrames."""
    
    def test_vectorized_time_calculation(self):
        """Test vectorized calculation on a Series of timedeltas."""
        time_diffs = pd.Series([
            timedelta(days=1),
            timedelta(days=7),
            timedelta(days=30),
            timedelta(days=90),
            timedelta(days=365)
        ])
        
        result = calculate_time_to_maturity_vectorized(time_diffs)
        expected = pd.Series([1, 7, 30, 90, 365]) / 365.25
        
        # Check that all results are close to expected
        for i in range(len(result)):
            assert abs(result.iloc[i] - expected.iloc[i]) < 1e-6
    
    def test_legacy_dataframe_fix(self):
        """Test fixing legacy DataFrames with time calculations."""
        # Create sample DataFrame with datetime columns
        df = pd.DataFrame({
            'date_time': [
                datetime(2025, 1, 1),
                datetime(2025, 1, 15),
                datetime(2025, 2, 1)
            ],
            'maturity_date': [
                datetime(2025, 2, 1),
                datetime(2025, 3, 15),
                datetime(2025, 5, 1)
            ],
            'option_price': [100, 150, 200]
        })
        
        # Apply the fix
        fixed_df = fix_legacy_time_calculation(df)
        
        # Check that time_to_maturity column was added
        assert 'time_to_maturity' in fixed_df.columns
        
        # Validate the calculations
        for i, row in fixed_df.iterrows():
            expected_ttm = calculate_time_to_maturity(
                row['date_time'], 
                row['maturity_date']
            )
            assert abs(row['time_to_maturity'] - expected_ttm) < 1e-6


class TestValidation:
    """Test validation functions."""
    
    def test_time_calculation_validation(self):
        """Test that validation function works correctly."""
        current = datetime(2025, 1, 1, 12, 0, 0)
        expiry = datetime(2025, 1, 31, 12, 0, 0)
        
        # Calculate correct value
        correct_ttm = calculate_time_to_maturity(current, expiry)
        
        # Should validate as correct
        assert validate_time_calculation(correct_ttm, current, expiry) == True
        
        # Should reject incorrect value
        incorrect_ttm = correct_ttm * 2  # Wrong value
        assert validate_time_calculation(incorrect_ttm, current, expiry) == False
    
    def test_business_days_calculation(self):
        """Test business days calculation."""
        current = date(2025, 1, 1)  # Wednesday
        expiry = date(2025, 1, 8)   # Next Wednesday (5 business days)
        
        result = get_business_days_to_maturity(current, expiry)
        expected = 5 / 252.0  # 5 trading days out of 252 per year
        
        # Should be close (within 1 trading day)
        assert abs(result - expected) < 2/252.0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_time_to_maturity(self):
        """Test when current time equals expiry time."""
        current = datetime(2025, 1, 1, 12, 0, 0)
        expiry = datetime(2025, 1, 1, 12, 0, 0)
        
        result = calculate_time_to_maturity(current, expiry)
        
        # Should return minimum time (1 hour)
        expected_min = 1.0 / (365.25 * 24)  # 1 hour in years
        assert abs(result - expected_min) < 1e-8
    
    def test_past_expiry_time(self):
        """Test when expiry is in the past."""
        current = datetime(2025, 1, 31, 12, 0, 0)
        expiry = datetime(2025, 1, 1, 12, 0, 0)  # Past date
        
        result = calculate_time_to_maturity(current, expiry)
        
        # Should return minimum time, not negative
        expected_min = 1.0 / (365.25 * 24)  # 1 hour in years
        assert result >= expected_min
    
    def test_timezone_handling(self):
        """Test timezone-aware datetime objects."""
        current = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        expiry = datetime(2025, 1, 31, 12, 0, 0, tzinfo=timezone.utc)
        
        result = calculate_time_to_maturity(current, expiry)
        expected = 30 / 365.25
        
        assert abs(result - expected) < 1e-6


class TestConstants:
    """Test that our constants are correct."""
    
    def test_seconds_per_year(self):
        """Test that SECONDS_PER_YEAR is calculated correctly."""
        expected = 365.25 * 24 * 3600
        assert SECONDS_PER_YEAR == expected
    
    def test_leap_year_accounting(self):
        """Test that leap years are properly accounted for."""
        # 365.25 accounts for leap year every 4 years
        # This should be more accurate than using 365
        
        four_years_seconds = 4 * 365.25 * 24 * 3600
        actual_four_years = (365 + 365 + 365 + 366) * 24 * 3600  # Including one leap year
        
        assert abs(four_years_seconds - actual_four_years) < 86400  # Within 1 day


class TestOptionsDataIntegration:
    """Test integration with options data processing."""
    
    def test_options_dataframe_processing(self):
        """Test processing options data with fixed time calculations."""
        # Simulate options data DataFrame
        options_data = pd.DataFrame({
            'instrument_name': ['BTC-31JAN25-100000-C', 'BTC-28FEB25-105000-C'],
            'timestamp': [1704067200000, 1704153600000],  # Mock timestamps
            'maturity_date': [
                datetime(2025, 1, 31),
                datetime(2025, 2, 28)
            ],
            'date_time': [
                datetime(2025, 1, 1),
                datetime(2025, 1, 2)
            ],
            'price': [5000, 4500],
            'strike_price': [100000, 105000],
            'iv': [0.8, 0.75]
        })
        
        # Apply fixed time calculation
        fixed_data = fix_legacy_time_calculation(options_data)
        
        # Validate that time calculations are reasonable
        assert all(fixed_data['time_to_maturity'] > 0)
        assert all(fixed_data['time_to_maturity'] < 1)  # Should be less than 1 year
        
        # Check specific calculation
        expected_ttm_1 = (datetime(2025, 1, 31) - datetime(2025, 1, 1)).total_seconds() / SECONDS_PER_YEAR
        assert abs(fixed_data['time_to_maturity'].iloc[0] - expected_ttm_1) < 1e-6


def test_main_function():
    """Test the main test function runs without errors."""
    # This should run all internal tests and pass
    result = test_time_calculation_fix()
    assert result == True, "Main test function should pass"


# Integration test that mimics the BTC_Option.py usage
def test_btc_option_integration():
    """Test integration with BTC_Option.py style processing."""
    # Mock the data that would come from BTC_Option.py
    mock_option_data = pd.DataFrame({
        'timestamp': [1704067200000, 1704153600000, 1704240000000],
        'instrument_name': ['BTC-31JAN25-100000-C', 'BTC-28FEB25-105000-C', 'BTC-28MAR25-110000-C'],
        'price': [0.05, 0.04, 0.03],  # In BTC terms
        'index_price': [100000, 105000, 108000],  # USD price of BTC
        'strike_price': [100000, 105000, 110000],
        'iv': [80, 75, 70],  # In percentage
        'option_type': ['c', 'c', 'c']
    })
    
    # Process like BTC_Option.py would (but with fixed time calculation)
    mock_option_data['date_time'] = mock_option_data['timestamp'].apply(
        lambda x: datetime.fromtimestamp(x / 1000)
    )
    
    # Add mock maturity dates
    mock_option_data['maturity_date'] = [
        datetime(2025, 1, 31),
        datetime(2025, 2, 28),
        datetime(2025, 3, 28)
    ]
    
    # Apply the FIXED time calculation (not the legacy bug)
    fixed_data = fix_legacy_time_calculation(
        mock_option_data,
        current_time_col='date_time',
        expiry_time_col='maturity_date',
        output_col='time_to_maturity'
    )
    
    # Verify the results are reasonable
    assert all(fixed_data['time_to_maturity'] > 0)
    assert all(fixed_data['time_to_maturity'] < 1)  # All less than 1 year
    
    # Convert IV to decimal (like BTC_Option.py does)
    fixed_data['iv'] = fixed_data['iv'] / 100
    assert all(fixed_data['iv'] <= 1.0)  # Should be decimal format
    
    # Calculate option price in USD (like BTC_Option.py does)
    fixed_data['price_usd'] = fixed_data['price'] * fixed_data['index_price']
    assert all(fixed_data['price_usd'] > 0)
    
    print("✅ BTC_Option.py integration test passed")


if __name__ == "__main__":
    # Run all tests if executed directly
    print("🧪 Running Time Calculation Fix Tests...")
    
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
    
    # Also run the integration test
    test_btc_option_integration()
    
    print("✅ All time calculation fix tests completed!")
    