# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Financial Mathematics Utilities for Qortfolio V2
Location: src/core/utils/math_utils.py

Professional-grade financial mathematics functions for options analytics,
volatility calculations, and risk management.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar, brentq
from typing import Union, Optional, Tuple, Dict, Any
import warnings
from datetime import datetime, timedelta
import logging

from ..exceptions import (
    MathematicalError, 
    InvalidParameterError,
    CalculationError
)

logger = logging.getLogger(__name__)

# ==================== CONSTANTS ====================

# Mathematical constants
SQRT_2PI = np.sqrt(2 * np.pi)
DAYS_PER_YEAR = 365.25
HOURS_PER_YEAR = DAYS_PER_YEAR * 24
MINUTES_PER_YEAR = HOURS_PER_YEAR * 60

# Financial constants
DEFAULT_RISK_FREE_RATE = 0.05
MIN_VOLATILITY = 1e-6
MAX_VOLATILITY = 5.0
MIN_TIME_TO_MATURITY = 1.0 / (DAYS_PER_YEAR * 24)  # 1 hour minimum


# ==================== VALIDATION FUNCTIONS ====================

def validate_positive(value: float, name: str) -> float:
    """Validate that a value is positive."""
    if value <= 0:
        raise InvalidParameterError(name, value, "must be positive")
    return float(value)


def validate_non_negative(value: float, name: str) -> float:
    """Validate that a value is non-negative."""
    if value < 0:
        raise InvalidParameterError(name, value, "must be non-negative")
    return float(value)


def validate_probability(value: float, name: str) -> float:
    """Validate that a value is a valid probability (0-1)."""
    if not 0 <= value <= 1:
        raise InvalidParameterError(name, value, "must be between 0 and 1")
    return float(value)


def validate_volatility(volatility: float) -> float:
    """Validate volatility parameter."""
    if volatility < MIN_VOLATILITY:
        raise InvalidParameterError("volatility", volatility, f"must be >= {MIN_VOLATILITY}")
    if volatility > MAX_VOLATILITY:
        raise InvalidParameterError("volatility", volatility, f"must be <= {MAX_VOLATILITY}")
    return float(volatility)


def validate_time_to_maturity(time_to_maturity: float) -> float:
    """Validate time to maturity parameter."""
    if time_to_maturity < MIN_TIME_TO_MATURITY:
        raise InvalidParameterError(
            "time_to_maturity", 
            time_to_maturity, 
            f"must be >= {MIN_TIME_TO_MATURITY} (1 hour)"
        )
    return float(time_to_maturity)


# ==================== STATISTICAL FUNCTIONS ====================

def safe_log(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Safely compute logarithm, handling edge cases."""
    try:
        x = np.asarray(x)
        # Replace zeros and negative values with small positive number
        x_safe = np.where(x <= 0, 1e-10, x)
        return np.log(x_safe)
    except Exception as e:
        raise MathematicalError(f"Failed to compute logarithm: {e}", operation="log")


def safe_sqrt(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Safely compute square root, handling edge cases."""
    try:
        x = np.asarray(x)
        # Replace negative values with zero
        x_safe = np.maximum(x, 0)
        return np.sqrt(x_safe)
    except Exception as e:
        raise MathematicalError(f"Failed to compute square root: {e}", operation="sqrt")


def safe_exp(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Safely compute exponential, handling overflow."""
    try:
        x = np.asarray(x)
        # Clip to prevent overflow
        x_clipped = np.clip(x, -700, 700)
        return np.exp(x_clipped)
    except Exception as e:
        raise MathematicalError(f"Failed to compute exponential: {e}", operation="exp")


def normal_cdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Standard normal cumulative distribution function."""
    try:
        return stats.norm.cdf(x)
    except Exception as e:
        raise MathematicalError(f"Failed to compute normal CDF: {e}", operation="normal_cdf")


def normal_pdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Standard normal probability density function."""
    try:
        return stats.norm.pdf(x)
    except Exception as e:
        raise MathematicalError(f"Failed to compute normal PDF: {e}", operation="normal_pdf")


# ==================== BLACK-SCHOLES FUNCTIONS ====================

def calculate_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
    """
    Calculate d1 and d2 parameters for Black-Scholes formula.
    
    Args:
        S: Current spot price
        K: Strike price
        T: Time to maturity in years
        r: Risk-free rate
        sigma: Volatility (annualized)
    
    Returns:
        Tuple of (d1, d2)
        
    Raises:
        InvalidParameterError: If parameters are invalid
        MathematicalError: If calculation fails
    """
    try:
        # Validate inputs
        validate_positive(S, "spot_price")
        validate_positive(K, "strike_price")
        validate_time_to_maturity(T)
        validate_volatility(sigma)
        
        # Calculate d1 and d2
        sqrt_T = safe_sqrt(T)
        sigma_sqrt_T = sigma * sqrt_T
        
        if sigma_sqrt_T == 0:
            raise MathematicalError("Volatility * sqrt(T) is zero", operation="d1_d2_calculation")
        
        d1 = (safe_log(S / K) + (r + 0.5 * sigma**2) * T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T
        
        return float(d1), float(d2)
        
    except (InvalidParameterError, MathematicalError):
        raise
    except Exception as e:
        raise MathematicalError(
            f"Failed to calculate d1, d2: {e}", 
            operation="d1_d2_calculation",
            parameters={'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma}
        )


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate Black-Scholes call option price.
    
    Args:
        S: Current spot price
        K: Strike price
        T: Time to maturity in years
        r: Risk-free rate
        sigma: Volatility (annualized)
    
    Returns:
        Call option price
    """
    try:
        d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
        
        call_price = (
            S * normal_cdf(d1) - 
            K * safe_exp(-r * T) * normal_cdf(d2)
        )
        
        return max(0.0, float(call_price))
        
    except Exception as e:
        raise MathematicalError(
            f"Black-Scholes call calculation failed: {e}",
            operation="black_scholes_call",
            parameters={'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma}
        )


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate Black-Scholes put option price.
    
    Args:
        S: Current spot price
        K: Strike price
        T: Time to maturity in years
        r: Risk-free rate
        sigma: Volatility (annualized)
    
    Returns:
        Put option price
    """
    try:
        d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
        
        put_price = (
            K * safe_exp(-r * T) * normal_cdf(-d2) - 
            S * normal_cdf(-d1)
        )
        
        return max(0.0, float(put_price))
        
    except Exception as e:
        raise MathematicalError(
            f"Black-Scholes put calculation failed: {e}",
            operation="black_scholes_put",
            parameters={'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma}
        )


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, 
                       option_type: str) -> float:
    """
    Calculate Black-Scholes option price for call or put.
    
    Args:
        S: Current spot price
        K: Strike price
        T: Time to maturity in years
        r: Risk-free rate
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'
    
    Returns:
        Option price
    """
    option_type = option_type.lower()
    
    if option_type == 'call':
        return black_scholes_call(S, K, T, r, sigma)
    elif option_type == 'put':
        return black_scholes_put(S, K, T, r, sigma)
    else:
        raise InvalidParameterError("option_type", option_type, "must be 'call' or 'put'")


# ==================== GREEKS CALCULATIONS ====================

def calculate_delta(S: float, K: float, T: float, r: float, sigma: float, 
                   option_type: str) -> float:
    """
    Calculate option delta.
    
    Args:
        S: Current spot price
        K: Strike price
        T: Time to maturity in years
        r: Risk-free rate
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'
    
    Returns:
        Delta value
    """
    try:
        d1, _ = calculate_d1_d2(S, K, T, r, sigma)
        
        if option_type.lower() == 'call':
            return float(normal_cdf(d1))
        elif option_type.lower() == 'put':
            return float(normal_cdf(d1) - 1.0)
        else:
            raise InvalidParameterError("option_type", option_type, "must be 'call' or 'put'")
            
    except Exception as e:
        raise MathematicalError(f"Delta calculation failed: {e}", operation="delta")


def calculate_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate option gamma (same for calls and puts).
    
    Args:
        S: Current spot price
        K: Strike price
        T: Time to maturity in years
        r: Risk-free rate
        sigma: Volatility (annualized)
    
    Returns:
        Gamma value
    """
    try:
        d1, _ = calculate_d1_d2(S, K, T, r, sigma)
        
        gamma = normal_pdf(d1) / (S * sigma * safe_sqrt(T))
        return float(gamma)
        
    except Exception as e:
        raise MathematicalError(f"Gamma calculation failed: {e}", operation="gamma")


def calculate_theta(S: float, K: float, T: float, r: float, sigma: float, 
                   option_type: str) -> float:
    """
    Calculate option theta (time decay).
    
    Args:
        S: Current spot price
        K: Strike price
        T: Time to maturity in years
        r: Risk-free rate
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'
    
    Returns:
        Theta value (per day)
    """
    try:
        d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
        
        # Common terms
        sqrt_T = safe_sqrt(T)
        exp_rT = safe_exp(-r * T)
        
        # First term (same for both call and put)
        theta_term1 = -(S * normal_pdf(d1) * sigma) / (2 * sqrt_T)
        
        if option_type.lower() == 'call':
            # Call theta
            theta_term2 = -r * K * exp_rT * normal_cdf(d2)
            theta = theta_term1 + theta_term2
        elif option_type.lower() == 'put':
            # Put theta
            theta_term2 = r * K * exp_rT * normal_cdf(-d2)
            theta = theta_term1 + theta_term2
        else:
            raise InvalidParameterError("option_type", option_type, "must be 'call' or 'put'")
        
        # Convert to per-day theta
        return float(theta / DAYS_PER_YEAR)
        
    except Exception as e:
        raise MathematicalError(f"Theta calculation failed: {e}", operation="theta")


def calculate_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate option vega (same for calls and puts).
    
    Args:
        S: Current spot price
        K: Strike price
        T: Time to maturity in years
        r: Risk-free rate
        sigma: Volatility (annualized)
    
    Returns:
        Vega value (per 1% volatility change)
    """
    try:
        d1, _ = calculate_d1_d2(S, K, T, r, sigma)
        
        vega = S * normal_pdf(d1) * safe_sqrt(T)
        
        # Convert to per 1% volatility change
        return float(vega / 100.0)
        
    except Exception as e:
        raise MathematicalError(f"Vega calculation failed: {e}", operation="vega")


def calculate_rho(S: float, K: float, T: float, r: float, sigma: float, 
                 option_type: str) -> float:
    """
    Calculate option rho (interest rate sensitivity).
    
    Args:
        S: Current spot price
        K: Strike price
        T: Time to maturity in years
        r: Risk-free rate
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'
    
    Returns:
        Rho value (per 1% interest rate change)
    """
    try:
        _, d2 = calculate_d1_d2(S, K, T, r, sigma)
        
        exp_rT = safe_exp(-r * T)
        
        if option_type.lower() == 'call':
            rho = K * T * exp_rT * normal_cdf(d2)
        elif option_type.lower() == 'put':
            rho = -K * T * exp_rT * normal_cdf(-d2)
        else:
            raise InvalidParameterError("option_type", option_type, "must be 'call' or 'put'")
        
        # Convert to per 1% interest rate change
        return float(rho / 100.0)
        
    except Exception as e:
        raise MathematicalError(f"Rho calculation failed: {e}", operation="rho")


def calculate_all_greeks(S: float, K: float, T: float, r: float, sigma: float, 
                        option_type: str) -> Dict[str, float]:
    """
    Calculate all Greeks for an option.
    
    Args:
        S: Current spot price
        K: Strike price
        T: Time to maturity in years
        r: Risk-free rate
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'
    
    Returns:
        Dictionary with all Greeks
    """
    try:
        return {
            'delta': calculate_delta(S, K, T, r, sigma, option_type),
            'gamma': calculate_gamma(S, K, T, r, sigma),
            'theta': calculate_theta(S, K, T, r, sigma, option_type),
            'vega': calculate_vega(S, K, T, r, sigma),
            'rho': calculate_rho(S, K, T, r, sigma, option_type)
        }
    except Exception as e:
        raise MathematicalError(f"Greeks calculation failed: {e}", operation="all_greeks")


# ==================== IMPLIED VOLATILITY ====================

def implied_volatility_objective(sigma: float, market_price: float, S: float, K: float, 
                                T: float, r: float, option_type: str) -> float:
    """Objective function for implied volatility calculation."""
    try:
        theoretical_price = black_scholes_price(S, K, T, r, sigma, option_type)
        return (theoretical_price - market_price) ** 2
    except Exception:
        return 1e10  # Return large error if calculation fails


def calculate_implied_volatility(market_price: float, S: float, K: float, T: float, 
                               r: float, option_type: str, 
                               initial_guess: float = 0.2) -> Optional[float]:
    """
    Calculate implied volatility using optimization.
    
    Args:
        market_price: Observed option price
        S: Current spot price
        K: Strike price
        T: Time to maturity in years
        r: Risk-free rate
        option_type: 'call' or 'put'
        initial_guess: Initial volatility guess
    
    Returns:
        Implied volatility or None if calculation fails
    """
    try:
        # Validate inputs
        validate_positive(market_price, "market_price")
        validate_positive(S, "spot_price")
        validate_positive(K, "strike_price")
        validate_time_to_maturity(T)
        
        # Check if option is worthless
        intrinsic_value = max(S - K, 0) if option_type.lower() == 'call' else max(K - S, 0)
        if market_price < intrinsic_value:
            logger.warning(f"Market price {market_price} below intrinsic value {intrinsic_value}")
            return None
        
        # Use optimization to find implied volatility
        result = minimize_scalar(
            implied_volatility_objective,
            args=(market_price, S, K, T, r, option_type),
            bounds=(MIN_VOLATILITY, MAX_VOLATILITY),
            method='bounded'
        )
        
        if result.success and result.fun < 1e-6:
            return float(result.x)
        else:
            logger.warning(f"Implied volatility calculation failed: {result}")
            return None
            
    except Exception as e:
        logger.error(f"Implied volatility calculation error: {e}")
        return None


# ==================== VOLATILITY FUNCTIONS ====================

def calculate_historical_volatility(prices: Union[pd.Series, np.ndarray], 
                                   window: int = 30) -> float:
    """
    Calculate historical volatility from price series.
    
    Args:
        prices: Price series
        window: Number of periods for calculation
    
    Returns:
        Annualized historical volatility
    """
    try:
        prices = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
        
        if len(prices) < window + 1:
            raise InvalidParameterError("prices", len(prices), f"must have at least {window + 1} values")
        
        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        if len(log_returns) < window:
            raise InvalidParameterError("log_returns", len(log_returns), f"must have at least {window} values")
        
        # Calculate rolling volatility
        rolling_std = log_returns.rolling(window=window).std()
        latest_vol = rolling_std.iloc[-1]
        
        if pd.isna(latest_vol):
            raise MathematicalError("Historical volatility calculation resulted in NaN")
        
        # Annualize volatility (assuming daily data)
        annualized_vol = latest_vol * np.sqrt(DAYS_PER_YEAR)
        
        return float(annualized_vol)
        
    except Exception as e:
        raise MathematicalError(f"Historical volatility calculation failed: {e}")


def calculate_realized_volatility(prices: Union[pd.Series, np.ndarray], 
                                 periods: int = 30) -> float:
    """
    Calculate realized volatility over specified periods.
    
    Args:
        prices: Price series
        periods: Number of periods to look back
    
    Returns:
        Annualized realized volatility
    """
    try:
        prices = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
        recent_prices = prices.tail(periods + 1)
        
        return calculate_historical_volatility(recent_prices, periods)
        
    except Exception as e:
        raise MathematicalError(f"Realized volatility calculation failed: {e}")


# ==================== UTILITY FUNCTIONS ====================

def moneyness(S: float, K: float) -> float:
    """Calculate moneyness (S/K)."""
    try:
        validate_positive(S, "spot_price")
        validate_positive(K, "strike_price")
        return float(S / K)
    except Exception as e:
        raise MathematicalError(f"Moneyness calculation failed: {e}")


def is_otm(S: float, K: float, option_type: str) -> bool:
    """Check if option is out-of-the-money."""
    try:
        if option_type.lower() == 'call':
            return S < K
        elif option_type.lower() == 'put':
            return S > K
        else:
            raise InvalidParameterError("option_type", option_type, "must be 'call' or 'put'")
    except Exception as e:
        raise MathematicalError(f"OTM check failed: {e}")


def is_itm(S: float, K: float, option_type: str) -> bool:
    """Check if option is in-the-money."""
    return not is_otm(S, K, option_type)


def intrinsic_value(S: float, K: float, option_type: str) -> float:
    """Calculate intrinsic value of option."""
    try:
        validate_positive(S, "spot_price")
        validate_positive(K, "strike_price")
        
        if option_type.lower() == 'call':
            return max(0.0, S - K)
        elif option_type.lower() == 'put':
            return max(0.0, K - S)
        else:
            raise InvalidParameterError("option_type", option_type, "must be 'call' or 'put'")
    except Exception as e:
        raise MathematicalError(f"Intrinsic value calculation failed: {e}")


def time_value(option_price: float, S: float, K: float, option_type: str) -> float:
    """Calculate time value of option."""
    try:
        intrinsic = intrinsic_value(S, K, option_type)
        time_val = option_price - intrinsic
        return max(0.0, float(time_val))
    except Exception as e:
        raise MathematicalError(f"Time value calculation failed: {e}")


# ==================== EXPORTS ====================

__all__ = [
    # Validation functions
    'validate_positive',
    'validate_non_negative', 
    'validate_probability',
    'validate_volatility',
    'validate_time_to_maturity',
    
    # Statistical functions
    'safe_log',
    'safe_sqrt',
    'safe_exp',
    'normal_cdf',
    'normal_pdf',
    
    # Black-Scholes functions
    'calculate_d1_d2',
    'black_scholes_call',
    'black_scholes_put',
    'black_scholes_price',
    
    # Greeks functions
    'calculate_delta',
    'calculate_gamma',
    'calculate_theta',
    'calculate_vega',
    'calculate_rho',
    'calculate_all_greeks',
    
    # Implied volatility
    'calculate_implied_volatility',
    
    # Volatility functions
    'calculate_historical_volatility',
    'calculate_realized_volatility',
    
    # Utility functions
    'moneyness',
    'is_otm',
    'is_itm',
    'intrinsic_value',
    'time_value',
    
    # Constants
    'DAYS_PER_YEAR',
    'DEFAULT_RISK_FREE_RATE',
    'MIN_TIME_TO_MATURITY'
]