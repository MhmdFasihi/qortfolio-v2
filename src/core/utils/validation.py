# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Data Validation Utilities for Qortfolio V2
Location: src/core/utils/validation.py

Comprehensive data validation functions for financial data, API responses,
and system inputs to ensure data quality and integrity.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime, date, timedelta
import re
import logging
from decimal import Decimal, InvalidOperation

from ..exceptions import (
    DataValidationError,
    InvalidParameterError,
    EmptyDataError,
    DataIntegrityError
)

logger = logging.getLogger(__name__)

# ==================== CONSTANTS ====================

# Valid option types
VALID_OPTION_TYPES = {'call', 'put', 'C', 'P'}

# Valid currency symbols
VALID_CURRENCIES = {'BTC', 'ETH', 'SOL', 'BNB', 'ADA', 'XRP', 'LTC', 'BCH', 'DOT', 'LINK'}

# Date format patterns
DATE_FORMATS = [
    '%Y-%m-%d',
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%dT%H:%M:%S',
    '%Y-%m-%dT%H:%M:%SZ',
    '%d-%m-%Y',
    '%m/%d/%Y'
]

# Numeric validation limits
MIN_PRICE = 1e-10
MAX_PRICE = 1e10
MIN_VOLATILITY = 1e-6
MAX_VOLATILITY = 10.0
MIN_TIME_TO_MATURITY = 1.0 / (365.25 * 24)  # 1 hour


# ==================== BASIC VALIDATION FUNCTIONS ====================

def is_valid_number(value: Any, allow_negative: bool = True, 
                   allow_zero: bool = True) -> bool:
    """
    Check if value is a valid number.
    
    Args:
        value: Value to check
        allow_negative: Whether negative numbers are allowed
        allow_zero: Whether zero is allowed
    
    Returns:
        True if valid number
    """
    try:
        num = float(value)
        
        if np.isnan(num) or np.isinf(num):
            return False
        
        if not allow_negative and num < 0:
            return False
            
        if not allow_zero and num == 0:
            return False
            
        return True
        
    except (ValueError, TypeError):
        return False


def is_valid_integer(value: Any, min_value: Optional[int] = None, 
                    max_value: Optional[int] = None) -> bool:
    """
    Check if value is a valid integer within bounds.
    
    Args:
        value: Value to check
        min_value: Minimum allowed value
        max_value: Maximum allowed value
    
    Returns:
        True if valid integer
    """
    try:
        int_val = int(value)
        
        if min_value is not None and int_val < min_value:
            return False
            
        if max_value is not None and int_val > max_value:
            return False
            
        return True
        
    except (ValueError, TypeError):
        return False


def is_valid_string(value: Any, min_length: int = 1, 
                   max_length: Optional[int] = None, 
                   pattern: Optional[str] = None) -> bool:
    """
    Check if value is a valid string.
    
    Args:
        value: Value to check
        min_length: Minimum string length
        max_length: Maximum string length
        pattern: Regex pattern to match
    
    Returns:
        True if valid string
    """
    try:
        if not isinstance(value, str):
            return False
            
        if len(value) < min_length:
            return False
            
        if max_length is not None and len(value) > max_length:
            return False
            
        if pattern is not None and not re.match(pattern, value):
            return False
            
        return True
        
    except Exception:
        return False


def is_valid_date(value: Any, date_formats: List[str] = None) -> bool:
    """
    Check if value is a valid date.
    
    Args:
        value: Value to check
        date_formats: List of date format strings to try
    
    Returns:
        True if valid date
    """
    if date_formats is None:
        date_formats = DATE_FORMATS
        
    # Handle datetime objects
    if isinstance(value, (datetime, date)):
        return True
        
    # Handle pandas Timestamp
    if hasattr(value, 'to_pydatetime'):
        return True
        
    # Handle string dates
    if isinstance(value, str):
        for fmt in date_formats:
            try:
                datetime.strptime(value, fmt)
                return True
            except ValueError:
                continue
                
    return False


# ==================== FINANCIAL DATA VALIDATION ====================

def validate_price(price: Any, field_name: str = "price") -> float:
    """
    Validate price value.
    
    Args:
        price: Price value to validate
        field_name: Name of the field for error messages
    
    Returns:
        Validated price as float
        
    Raises:
        DataValidationError: If price is invalid
    """
    try:
        if price is None:
            raise DataValidationError(f"{field_name} cannot be None", field=field_name)
            
        price_float = float(price)
        
        if np.isnan(price_float) or np.isinf(price_float):
            raise DataValidationError(f"{field_name} must be a finite number", 
                                    field=field_name, value=price)
        
        if price_float < MIN_PRICE:
            raise DataValidationError(f"{field_name} must be >= {MIN_PRICE}", 
                                    field=field_name, value=price_float)
        
        if price_float > MAX_PRICE:
            raise DataValidationError(f"{field_name} must be <= {MAX_PRICE}", 
                                    field=field_name, value=price_float)
        
        return price_float
        
    except DataValidationError:
        raise
    except Exception as e:
        raise DataValidationError(f"Invalid {field_name}: {e}", field=field_name, value=price)


def validate_volatility(volatility: Any, field_name: str = "volatility") -> float:
    """
    Validate volatility value.
    
    Args:
        volatility: Volatility value to validate
        field_name: Name of the field for error messages
    
    Returns:
        Validated volatility as float
        
    Raises:
        DataValidationError: If volatility is invalid
    """
    try:
        if volatility is None:
            raise DataValidationError(f"{field_name} cannot be None", field=field_name)
            
        vol_float = float(volatility)
        
        if np.isnan(vol_float) or np.isinf(vol_float):
            raise DataValidationError(f"{field_name} must be a finite number", 
                                    field=field_name, value=volatility)
        
        if vol_float < MIN_VOLATILITY:
            raise DataValidationError(f"{field_name} must be >= {MIN_VOLATILITY}", 
                                    field=field_name, value=vol_float)
        
        if vol_float > MAX_VOLATILITY:
            raise DataValidationError(f"{field_name} must be <= {MAX_VOLATILITY}", 
                                    field=field_name, value=vol_float)
        
        return vol_float
        
    except DataValidationError:
        raise
    except Exception as e:
        raise DataValidationError(f"Invalid {field_name}: {e}", field=field_name, value=volatility)


def validate_time_to_maturity(ttm: Any, field_name: str = "time_to_maturity") -> float:
    """
    Validate time to maturity value.
    
    Args:
        ttm: Time to maturity value to validate
        field_name: Name of the field for error messages
    
    Returns:
        Validated time to maturity as float
        
    Raises:
        DataValidationError: If time to maturity is invalid
    """
    try:
        if ttm is None:
            raise DataValidationError(f"{field_name} cannot be None", field=field_name)
            
        ttm_float = float(ttm)
        
        if np.isnan(ttm_float) or np.isinf(ttm_float):
            raise DataValidationError(f"{field_name} must be a finite number", 
                                    field=field_name, value=ttm)
        
        if ttm_float < MIN_TIME_TO_MATURITY:
            raise DataValidationError(
                f"{field_name} must be >= {MIN_TIME_TO_MATURITY} (1 hour minimum)", 
                field=field_name, value=ttm_float
            )
        
        # Reasonable upper bound (10 years)
        if ttm_float > 10.0:
            raise DataValidationError(f"{field_name} must be <= 10 years", 
                                    field=field_name, value=ttm_float)
        
        return ttm_float
        
    except DataValidationError:
        raise
    except Exception as e:
        raise DataValidationError(f"Invalid {field_name}: {e}", field=field_name, value=ttm)


def validate_option_type(option_type: Any, field_name: str = "option_type") -> str:
    """
    Validate option type value.
    
    Args:
        option_type: Option type to validate
        field_name: Name of the field for error messages
    
    Returns:
        Validated option type as standardized string
        
    Raises:
        DataValidationError: If option type is invalid
    """
    try:
        if option_type is None:
            raise DataValidationError(f"{field_name} cannot be None", field=field_name)
            
        option_str = str(option_type).strip()
        
        if option_str.lower() not in {'call', 'put', 'c', 'p'}:
            raise DataValidationError(
                f"{field_name} must be 'call', 'put', 'C', or 'P'", 
                field=field_name, value=option_type
            )
        
        # Standardize to lowercase
        return 'call' if option_str.lower() in {'call', 'c'} else 'put'
        
    except DataValidationError:
        raise
    except Exception as e:
        raise DataValidationError(f"Invalid {field_name}: {e}", field=field_name, value=option_type)


def validate_currency_symbol(symbol: Any, field_name: str = "currency") -> str:
    """
    Validate currency symbol.
    
    Args:
        symbol: Currency symbol to validate
        field_name: Name of the field for error messages
    
    Returns:
        Validated currency symbol as uppercase string
        
    Raises:
        DataValidationError: If currency symbol is invalid
    """
    try:
        if symbol is None:
            raise DataValidationError(f"{field_name} cannot be None", field=field_name)
            
        symbol_str = str(symbol).strip().upper()
        
        if not symbol_str:
            raise DataValidationError(f"{field_name} cannot be empty", field=field_name)
        
        # Check against known currencies (can be extended)
        if symbol_str not in VALID_CURRENCIES:
            logger.warning(f"Unknown currency symbol: {symbol_str}")
            # Don't raise error for unknown currencies, just log warning
        
        return symbol_str
        
    except DataValidationError:
        raise
    except Exception as e:
        raise DataValidationError(f"Invalid {field_name}: {e}", field=field_name, value=symbol)


# ==================== DATAFRAME VALIDATION ====================

def validate_dataframe_not_empty(df: pd.DataFrame, name: str = "DataFrame") -> pd.DataFrame:
    """
    Validate that DataFrame is not empty.
    
    Args:
        df: DataFrame to validate
        name: Name for error messages
    
    Returns:
        The DataFrame if valid
        
    Raises:
        EmptyDataError: If DataFrame is empty
        DataValidationError: If input is not a DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"{name} must be a pandas DataFrame", 
                                field="dataframe_type", value=type(df))
    
    if df.empty:
        raise EmptyDataError(f"{name} is empty", source="dataframe_validation")
    
    return df


def validate_required_columns(df: pd.DataFrame, required_columns: List[str], 
                             name: str = "DataFrame") -> pd.DataFrame:
    """
    Validate that DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Name for error messages
    
    Returns:
        The DataFrame if valid
        
    Raises:
        DataValidationError: If required columns are missing
    """
    validate_dataframe_not_empty(df, name)
    
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        raise DataValidationError(
            f"{name} missing required columns: {list(missing_columns)}", 
            field="missing_columns", 
            value=list(missing_columns)
        )
    
    return df


def validate_options_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate options DataFrame structure and data.
    
    Args:
        df: Options DataFrame to validate
    
    Returns:
        Validated DataFrame
        
    Raises:
        DataValidationError: If DataFrame is invalid
    """
    # Check required columns
    required_columns = [
        'instrument_name', 'strike', 'expiry', 'option_type', 
        'mark_price', 'underlying_price', 'time_to_expiry'
    ]
    
    validate_required_columns(df, required_columns, "Options DataFrame")
    
    # Validate each row
    validated_data = []
    errors = []
    
    for idx, row in df.iterrows():
        try:
            validated_row = {
                'instrument_name': str(row['instrument_name']).strip(),
                'strike': validate_price(row['strike'], 'strike'),
                'expiry': row['expiry'],  # Keep as-is, might be datetime
                'option_type': validate_option_type(row['option_type']),
                'mark_price': validate_price(row['mark_price'], 'mark_price'),
                'underlying_price': validate_price(row['underlying_price'], 'underlying_price'),
                'time_to_expiry': validate_time_to_maturity(row['time_to_expiry'])
            }
            
            # Add optional columns if present
            for col in ['bid', 'ask', 'volume', 'open_interest', 'iv']:
                if col in row and pd.notna(row[col]):
                    if col == 'iv':
                        validated_row[col] = validate_volatility(row[col], 'implied_volatility')
                    else:
                        validated_row[col] = validate_price(row[col], col)
            
            validated_data.append(validated_row)
            
        except (DataValidationError, InvalidParameterError) as e:
            errors.append(f"Row {idx}: {e}")
            
    if errors:
        error_summary = f"Validation failed for {len(errors)} rows:\n" + "\n".join(errors[:10])
        if len(errors) > 10:
            error_summary += f"\n... and {len(errors) - 10} more errors"
        
        raise DataValidationError(error_summary, field="dataframe_validation")
    
    return pd.DataFrame(validated_data)


def validate_price_dataframe(df: pd.DataFrame, price_column: str = 'Close') -> pd.DataFrame:
    """
    Validate price DataFrame structure and data.
    
    Args:
        df: Price DataFrame to validate
        price_column: Name of the price column
    
    Returns:
        Validated DataFrame
        
    Raises:
        DataValidationError: If DataFrame is invalid
    """
    validate_dataframe_not_empty(df, "Price DataFrame")
    
    # Check for price column
    if price_column not in df.columns:
        raise DataValidationError(
            f"Price DataFrame missing '{price_column}' column", 
            field="missing_column", 
            value=price_column
        )
    
    # Validate prices
    invalid_prices = []
    for idx, price in df[price_column].items():
        if not is_valid_number(price, allow_negative=False, allow_zero=False):
            invalid_prices.append(f"Row {idx}: {price}")
    
    if invalid_prices:
        error_msg = f"Invalid prices found:\n" + "\n".join(invalid_prices[:10])
        if len(invalid_prices) > 10:
            error_msg += f"\n... and {len(invalid_prices) - 10} more"
        
        raise DataValidationError(error_msg, field="price_validation")
    
    return df


# ==================== API RESPONSE VALIDATION ====================

def validate_api_response(response: Dict[str, Any], required_fields: List[str], 
                         api_name: str = "API") -> Dict[str, Any]:
    """
    Validate API response structure.
    
    Args:
        response: API response dictionary
        required_fields: List of required field names
        api_name: Name of the API for error messages
    
    Returns:
        Validated response
        
    Raises:
        DataValidationError: If response is invalid
    """
    if not isinstance(response, dict):
        raise DataValidationError(
            f"{api_name} response must be a dictionary", 
            field="response_type", 
            value=type(response)
        )
    
    missing_fields = set(required_fields) - set(response.keys())
    
    if missing_fields:
        raise DataValidationError(
            f"{api_name} response missing required fields: {list(missing_fields)}", 
            field="missing_fields", 
            value=list(missing_fields)
        )
    
    return response


def validate_deribit_options_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Validate Deribit options API response.
    
    Args:
        response: Deribit API response
    
    Returns:
        List of validated option data
        
    Raises:
        DataValidationError: If response is invalid
    """
    # Validate basic response structure
    validate_api_response(response, ['result'], 'Deribit')
    
    result = response['result']
    
    if not isinstance(result, list):
        raise DataValidationError(
            "Deribit options response 'result' must be a list", 
            field="result_type", 
            value=type(result)
        )
    
    if not result:
        raise EmptyDataError("options data", "Deribit API")
    
    # Validate each option
    validated_options = []
    
    for idx, option in enumerate(result):
        try:
            required_fields = ['instrument_name', 'mark_price', 'underlying_price']
            validate_api_response(option, required_fields, f'Deribit option {idx}')
            
            validated_options.append(option)
            
        except DataValidationError as e:
            logger.warning(f"Skipping invalid option {idx}: {e}")
            continue
    
    if not validated_options:
        raise DataValidationError("No valid options found in Deribit response")
    
    return validated_options


# ==================== UTILITY FUNCTIONS ====================

def clean_numeric_data(df: pd.DataFrame, numeric_columns: List[str], 
                      drop_invalid: bool = True) -> pd.DataFrame:
    """
    Clean numeric data in DataFrame.
    
    Args:
        df: DataFrame to clean
        numeric_columns: List of columns that should be numeric
        drop_invalid: Whether to drop rows with invalid data
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    for col in numeric_columns:
        if col in df_clean.columns:
            # Convert to numeric, replacing errors with NaN
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Replace infinite values with NaN
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
    
    if drop_invalid:
        # Drop rows where all numeric columns are NaN
        df_clean = df_clean.dropna(subset=numeric_columns, how='all')
    
    return df_clean


def validate_data_integrity(df: pd.DataFrame, checks: Dict[str, Callable]) -> Dict[str, bool]:
    """
    Run custom data integrity checks on DataFrame.
    
    Args:
        df: DataFrame to check
        checks: Dictionary of check_name -> check_function
    
    Returns:
        Dictionary of check results
    """
    results = {}
    
    for check_name, check_func in checks.items():
        try:
            results[check_name] = check_func(df)
        except Exception as e:
            logger.error(f"Data integrity check '{check_name}' failed: {e}")
            results[check_name] = False
    
    return results


def get_validation_summary(df: pd.DataFrame, validations: Dict[str, bool]) -> Dict[str, Any]:
    """
    Get summary of validation results.
    
    Args:
        df: DataFrame that was validated
        validations: Dictionary of validation results
    
    Returns:
        Validation summary
    """
    total_checks = len(validations)
    passed_checks = sum(validations.values())
    
    return {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'total_checks': total_checks,
        'passed_checks': passed_checks,
        'failed_checks': total_checks - passed_checks,
        'success_rate': passed_checks / max(total_checks, 1),
        'validation_details': validations
    }


# ==================== EXPORTS ====================

__all__ = [
    # Basic validation functions
    'is_valid_number',
    'is_valid_integer',
    'is_valid_string',
    'is_valid_date',
    
    # Financial data validation
    'validate_price',
    'validate_volatility',
    'validate_time_to_maturity',
    'validate_option_type',
    'validate_currency_symbol',
    
    # DataFrame validation
    'validate_dataframe_not_empty',
    'validate_required_columns',
    'validate_options_dataframe',
    'validate_price_dataframe',
    
    # API response validation
    'validate_api_response',
    'validate_deribit_options_response',
    
    # Utility functions
    'clean_numeric_data',
    'validate_data_integrity',
    'get_validation_summary',
    
    # Constants
    'VALID_OPTION_TYPES',
    'VALID_CURRENCIES',
    'MIN_PRICE',
    'MAX_PRICE',
    'MIN_VOLATILITY',
    'MAX_VOLATILITY'
]