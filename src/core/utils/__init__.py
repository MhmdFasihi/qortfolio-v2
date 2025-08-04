# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
src/core/utils/__init__.py
Core utilities package initialization.

This module provides easy access to all core utility functions including:
- Time calculations with bug fixes
- Financial mathematics for options
- Data validation utilities
"""

# Time utilities (CRITICAL BUG FIXES)
from .time_utils import (
    calculate_time_to_maturity,
    calculate_time_to_maturity_vectorized,
    fix_legacy_time_calculation,
    validate_time_calculation,
    get_business_days_to_maturity,
    time_to_maturity_from_days,
    time_to_maturity_to_days,
    get_current_utc_time,
    SECONDS_PER_YEAR,
    DAYS_PER_YEAR,
    MIN_TIME_YEARS,
    TimeCalculationError
)

# Mathematical utilities  
from .math_utils import (
    # Validation functions
    validate_positive,
    validate_non_negative,
    validate_probability,
    validate_volatility as validate_vol_math,
    validate_time_to_maturity as validate_ttm_math,
    
    # Statistical functions
    safe_log,
    safe_sqrt,
    safe_exp,
    normal_cdf,
    normal_pdf,
    
    # Black-Scholes functions
    calculate_d1_d2,
    black_scholes_call,
    black_scholes_put,
    black_scholes_price,
    
    # Greeks functions
    calculate_delta,
    calculate_gamma,
    calculate_theta,
    calculate_vega,
    calculate_rho,
    calculate_all_greeks,
    
    # Implied volatility
    calculate_implied_volatility,
    
    # Volatility functions
    calculate_historical_volatility,
    calculate_realized_volatility,
    
    # Utility functions
    moneyness,
    is_otm,
    is_itm,
    intrinsic_value,
    time_value,
    
    # Constants
    DEFAULT_RISK_FREE_RATE,
    MIN_TIME_TO_MATURITY
)

# Data validation utilities
from .validation import (
    # Basic validation functions
    is_valid_number,
    is_valid_integer,
    is_valid_string,
    is_valid_date,
    
    # Financial data validation
    validate_price,
    validate_volatility,
    validate_time_to_maturity,
    validate_option_type,
    validate_currency_symbol,
    
    # DataFrame validation
    validate_dataframe_not_empty,
    validate_required_columns,
    validate_options_dataframe,
    validate_price_dataframe,
    
    # API response validation
    validate_api_response,
    validate_deribit_options_response,
    
    # Utility functions
    clean_numeric_data,
    validate_data_integrity,
    get_validation_summary,
    
    # Constants
    VALID_OPTION_TYPES,
    VALID_CURRENCIES,
    MIN_PRICE,
    MAX_PRICE,
    MIN_VOLATILITY,
    MAX_VOLATILITY
)

# ==================== CONVENIENCE FUNCTIONS ====================

def validate_option_parameters(S: float, K: float, T: float, r: float, sigma: float, 
                              option_type: str) -> dict:
    """
    Validate all parameters for option calculations.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility
        option_type: Option type ('call' or 'put')
    
    Returns:
        Dictionary of validated parameters
    """
    return {
        'S': validate_price(S, 'spot_price'),
        'K': validate_price(K, 'strike_price'),
        'T': validate_time_to_maturity(T),
        'r': validate_non_negative(r, 'risk_free_rate'),
        'sigma': validate_volatility(sigma),
        'option_type': validate_option_type(option_type)
    }


def calculate_option_with_validation(S: float, K: float, T: float, r: float, 
                                   sigma: float, option_type: str) -> dict:
    """
    Calculate option price and Greeks with full validation.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility
        option_type: Option type ('call' or 'put')
    
    Returns:
        Dictionary with price and Greeks
    """
    # Validate all parameters
    params = validate_option_parameters(S, K, T, r, sigma, option_type)
    
    # Calculate price and Greeks
    price = black_scholes_price(**params)
    greeks = calculate_all_greeks(**params)
    
    return {
        'price': price,
        'intrinsic_value': intrinsic_value(params['S'], params['K'], params['option_type']),
        'time_value': time_value(price, params['S'], params['K'], params['option_type']),
        'moneyness': moneyness(params['S'], params['K']),
        **greeks
    }


def validate_and_clean_options_data(df, drop_invalid: bool = True) -> tuple:
    """
    Validate and clean options DataFrame.
    
    Args:
        df: Options DataFrame
        drop_invalid: Whether to drop invalid rows
    
    Returns:
        Tuple of (cleaned_df, validation_summary)
    """
    try:
        # Validate structure
        cleaned_df = validate_options_dataframe(df)
        
        # Clean numeric data
        numeric_columns = ['strike', 'mark_price', 'underlying_price', 'time_to_expiry']
        if 'iv' in cleaned_df.columns:
            numeric_columns.append('iv')
            
        cleaned_df = clean_numeric_data(cleaned_df, numeric_columns, drop_invalid)
        
        # Generate summary
        validation_summary = {
            'original_rows': len(df),
            'cleaned_rows': len(cleaned_df),
            'rows_removed': len(df) - len(cleaned_df),
            'success': True,
            'message': f'Successfully validated {len(cleaned_df)} options'
        }
        
        return cleaned_df, validation_summary
        
    except Exception as e:
        validation_summary = {
            'original_rows': len(df) if hasattr(df, '__len__') else 0,
            'cleaned_rows': 0,
            'rows_removed': 0,
            'success': False,
            'message': f'Validation failed: {str(e)}'
        }
        
        return None, validation_summary


# ==================== EXPORTS ====================

__all__ = [
    # Time utilities
    'calculate_time_to_maturity',
    'calculate_time_to_maturity_vectorized',
    'fix_legacy_time_calculation',
    'validate_time_calculation',
    'get_business_days_to_maturity',
    'time_to_maturity_from_days',
    'time_to_maturity_to_days',
    'get_current_utc_time',
    'SECONDS_PER_YEAR',
    'DAYS_PER_YEAR',
    'MIN_TIME_YEARS',
    'TimeCalculationError',
    
    # Mathematical utilities
    'validate_positive',
    'validate_non_negative',
    'validate_probability',
    'validate_vol_math',
    'validate_ttm_math',
    'safe_log',
    'safe_sqrt',
    'safe_exp',
    'normal_cdf',
    'normal_pdf',
    'calculate_d1_d2',
    'black_scholes_call',
    'black_scholes_put',
    'black_scholes_price',
    'calculate_delta',
    'calculate_gamma',
    'calculate_theta',
    'calculate_vega',
    'calculate_rho',
    'calculate_all_greeks',
    'calculate_implied_volatility',
    'calculate_historical_volatility',
    'calculate_realized_volatility',
    'moneyness',
    'is_otm',
    'is_itm',
    'intrinsic_value',
    'time_value',
    'DEFAULT_RISK_FREE_RATE',
    'MIN_TIME_TO_MATURITY',
    
    # Data validation utilities
    'is_valid_number',
    'is_valid_integer',
    'is_valid_string',
    'is_valid_date',
    'validate_price',
    'validate_volatility',
    'validate_time_to_maturity',
    'validate_option_type',
    'validate_currency_symbol',
    'validate_dataframe_not_empty',
    'validate_required_columns',
    'validate_options_dataframe',
    'validate_price_dataframe',
    'validate_api_response',
    'validate_deribit_options_response',
    'clean_numeric_data',
    'validate_data_integrity',
    'get_validation_summary',
    'VALID_OPTION_TYPES',
    'VALID_CURRENCIES',
    'MIN_PRICE',
    'MAX_PRICE',
    'MIN_VOLATILITY',
    'MAX_VOLATILITY',
    
    # Convenience functions
    'validate_option_parameters',
    'calculate_option_with_validation',
    'validate_and_clean_options_data'
]