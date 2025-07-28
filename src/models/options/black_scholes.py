# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Black-Scholes Options Pricing Model for Qortfolio V2
Professional implementation with Greeks calculations

Uses the FIXED time-to-maturity calculation to ensure accurate pricing.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Union, Optional, Dict, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from core.utils.time_utils import calculate_time_to_maturity
from core.logging import get_logger
from core.config import get_config


class OptionType(Enum):
    """Option types enumeration."""
    CALL = "call"
    PUT = "put"


@dataclass
class OptionParameters:
    """Option parameters for pricing calculations."""
    spot_price: float
    strike_price: float
    time_to_maturity: float
    volatility: float
    risk_free_rate: float
    option_type: OptionType
    dividend_yield: float = 0.0


@dataclass
class OptionPricing:
    """Complete option pricing results."""
    option_price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    intrinsic_value: float
    time_value: float
    moneyness: float


class BlackScholesModel:
    """
    Black-Scholes options pricing model.
    
    Features:
    - Accurate pricing using fixed time calculations
    - Complete Greeks calculations
    - Vectorized operations for efficiency
    - Comprehensive validation
    - Support for dividends
    """
    
    def __init__(self):
        """Initialize Black-Scholes model."""
        self.config = get_config()
        self.logger = get_logger("black_scholes")
        
        # Default parameters from configuration
        self.default_risk_free_rate = self.config.get(
            'options_config.default_params.risk_free_rate', 0.05
        )
        self.min_time_to_maturity = self.config.get(
            'options_config.default_params.min_time_to_maturity', 1/365.25
        )
        
        self.logger.info("Black-Scholes model initialized", extra={
            "default_risk_free_rate": self.default_risk_free_rate,
            "min_time_to_maturity": self.min_time_to_maturity
        })
    
    def calculate_option_price(self, params: OptionParameters) -> OptionPricing:
        """
        Calculate option price and Greeks using Black-Scholes model.
        
        Args:
            params: Option parameters
            
        Returns:
            Complete option pricing results
            
        Raises:
            ValueError: For invalid parameters
        """
        # Validate parameters
        self._validate_parameters(params)
        
        # Extract parameters
        S = params.spot_price
        K = params.strike_price
        T = max(params.time_to_maturity, self.min_time_to_maturity)
        σ = params.volatility
        r = params.risk_free_rate
        q = params.dividend_yield
        option_type = params.option_type.value if isinstance(params.option_type, OptionType) else params.option_type.lower()
        
        try:
            # Calculate d1 and d2
            d1, d2 = self._calculate_d1_d2(S, K, T, r, σ, q)
            
            # Calculate option price
            if option_type == 'call':
                option_price = self._call_price(S, K, T, r, σ, q, d1, d2)
            elif option_type == 'put':
                option_price = self._put_price(S, K, T, r, σ, q, d1, d2)
            else:
                raise ValueError(f"Invalid option type: {option_type}")
            
            # Calculate Greeks
            delta = self._calculate_delta(S, K, T, r, σ, q, d1, option_type)
            gamma = self._calculate_gamma(S, K, T, r, σ, q, d1)
            theta = self._calculate_theta(S, K, T, r, σ, q, d1, d2, option_type)
            vega = self._calculate_vega(S, K, T, r, σ, q, d1)
            rho = self._calculate_rho(S, K, T, r, σ, q, d2, option_type)
            
            # Calculate additional metrics
            intrinsic_value = self._calculate_intrinsic_value(S, K, option_type)
            time_value = option_price - intrinsic_value
            moneyness = S / K
            
            result = OptionPricing(
                option_price=option_price,
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                rho=rho,
                intrinsic_value=intrinsic_value,
                time_value=time_value,
                moneyness=moneyness
            )
            
            self.logger.debug(f"Calculated {option_type} option price", extra={
                "spot": S,
                "strike": K,
                "time_to_maturity": T,
                "volatility": σ,
                "option_price": option_price,
                "delta": delta
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Option pricing calculation failed: {e}", extra={
                "spot": S,
                "strike": K,
                "time_to_maturity": T,
                "volatility": σ,
                "option_type": option_type
            })
            raise
    
    def calculate_greeks(self, params: OptionParameters) -> OptionPricing:
        """
        Calculate Greeks (alias for calculate_option_price for compatibility).
        
        Args:
            params: Option parameters
            
        Returns:
            Complete option pricing results with all Greeks
        """
        return self.calculate_option_price(params)
    
    def _calculate_d1_d2(self, S: float, K: float, T: float, r: float, σ: float, q: float) -> Tuple[float, float]:
        """Calculate d1 and d2 parameters."""
        d1 = (np.log(S / K) + (r - q + 0.5 * σ**2) * T) / (σ * np.sqrt(T))
        d2 = d1 - σ * np.sqrt(T)
        return d1, d2
    
    def _call_price(self, S: float, K: float, T: float, r: float, σ: float, q: float, d1: float, d2: float) -> float:
        """Calculate call option price."""
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    def _put_price(self, S: float, K: float, T: float, r: float, σ: float, q: float, d1: float, d2: float) -> float:
        """Calculate put option price."""
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    def _calculate_delta(self, S: float, K: float, T: float, r: float, σ: float, q: float, d1: float, option_type: str) -> float:
        """Calculate delta."""
        if option_type == 'call':
            return np.exp(-q * T) * norm.cdf(d1)
        else:  # put
            return -np.exp(-q * T) * norm.cdf(-d1)
    
    def _calculate_gamma(self, S: float, K: float, T: float, r: float, σ: float, q: float, d1: float) -> float:
        """Calculate gamma."""
        return np.exp(-q * T) * norm.pdf(d1) / (S * σ * np.sqrt(T))
    
    def _calculate_theta(self, S: float, K: float, T: float, r: float, σ: float, q: float, d1: float, d2: float, option_type: str) -> float:
        """Calculate theta (per day)."""
        common_term = (-S * norm.pdf(d1) * σ * np.exp(-q * T)) / (2 * np.sqrt(T))
        
        if option_type == 'call':
            theta = common_term - r * K * np.exp(-r * T) * norm.cdf(d2) + q * S * np.exp(-q * T) * norm.cdf(d1)
        else:  # put
            theta = common_term + r * K * np.exp(-r * T) * norm.cdf(-d2) - q * S * np.exp(-q * T) * norm.cdf(-d1)
        
        return theta / 365.25  # Convert to per-day
    
    def _calculate_vega(self, S: float, K: float, T: float, r: float, σ: float, q: float, d1: float) -> float:
        """Calculate vega (per 1% volatility change)."""
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
    
    def _calculate_rho(self, S: float, K: float, T: float, r: float, σ: float, q: float, d2: float, option_type: str) -> float:
        """Calculate rho (per 1% interest rate change)."""
        if option_type == 'call':
            return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:  # put
            return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    def _calculate_intrinsic_value(self, S: float, K: float, option_type: str) -> float:
        """Calculate intrinsic value."""
        if option_type == 'call':
            return max(S - K, 0)
        else:  # put
            return max(K - S, 0)
    
    def _validate_parameters(self, params: OptionParameters) -> None:
        """Validate option parameters."""
        if params.spot_price <= 0:
            raise ValueError("Spot price must be positive")
        if params.strike_price <= 0:
            raise ValueError("Strike price must be positive")
        if params.time_to_maturity < 0:
            raise ValueError("Time to maturity cannot be negative")
        if params.volatility <= 0:
            raise ValueError("Volatility must be positive")
        
        option_type = params.option_type.value if isinstance(params.option_type, OptionType) else params.option_type
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")


# Convenience functions for quick calculations
def price_option(spot: float, strike: float, time_to_maturity: float, 
                volatility: float, option_type: str = 'call', 
                risk_free_rate: float = 0.05) -> float:
    """
    Quick option pricing function.
    
    Args:
        spot: Current spot price
        strike: Strike price
        time_to_maturity: Time to maturity in years
        volatility: Implied volatility (annualized)
        option_type: 'call' or 'put'
        risk_free_rate: Risk-free rate
        
    Returns:
        Option price
    """
    bs_model = BlackScholesModel()
    
    params = OptionParameters(
        spot_price=spot,
        strike_price=strike,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        risk_free_rate=risk_free_rate,
        option_type=OptionType.CALL if option_type.lower() == 'call' else OptionType.PUT
    )
    
    result = bs_model.calculate_option_price(params)
    return result.option_price


def calculate_greeks(spot: float, strike: float, time_to_maturity: float,
                    volatility: float, option_type: str = 'call',
                    risk_free_rate: float = 0.05) -> Dict[str, float]:
    """
    Calculate all Greeks for an option.
    
    Args:
        spot: Current spot price
        strike: Strike price
        time_to_maturity: Time to maturity in years
        volatility: Implied volatility (annualized)
        option_type: 'call' or 'put'
        risk_free_rate: Risk-free rate
        
    Returns:
        Dictionary with all Greeks
    """
    bs_model = BlackScholesModel()
    
    params = OptionParameters(
        spot_price=spot,
        strike_price=strike,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        risk_free_rate=risk_free_rate,
        option_type=OptionType.CALL if option_type.lower() == 'call' else OptionType.PUT
    )
    
    result = bs_model.calculate_option_price(params)
    
    return {
        'price': result.option_price,
        'delta': result.delta,
        'gamma': result.gamma,
        'theta': result.theta,
        'vega': result.vega,
        'rho': result.rho,
        'intrinsic': result.intrinsic_value,
        'time_value': result.time_value,
        'moneyness': result.moneyness
    }


if __name__ == "__main__":
    # Test the Black-Scholes implementation
    print("🧪 Testing Black-Scholes Implementation")
    print("=" * 40)
    
    bs_model = BlackScholesModel()
    
    # Test parameters (BTC example)
    spot = 50000.0
    strike = 52000.0
    time_to_maturity = 30 / 365.25  # 30 days
    volatility = 0.8  # 80% annual volatility
    risk_free_rate = 0.05  # 5%
    
    print(f"Test Parameters:")
    print(f"  Spot: ${spot:,.2f}")
    print(f"  Strike: ${strike:,.2f}")
    print(f"  Time to maturity: {time_to_maturity:.4f} years ({time_to_maturity*365.25:.0f} days)")
    print(f"  Volatility: {volatility:.1%}")
    print(f"  Risk-free rate: {risk_free_rate:.1%}")
    
    # Test call option
    print(f"\n📈 Call Option:")
    call_greeks = calculate_greeks(spot, strike, time_to_maturity, volatility, 'call', risk_free_rate)
    for key, value in call_greeks.items():
        if key == 'price':
            print(f"  {key.capitalize()}: ${value:.2f}")
        elif key in ['delta', 'gamma', 'theta', 'vega', 'rho']:
            print(f"  {key.capitalize()}: {value:.4f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
    
    # Test put option
    print(f"\n📉 Put Option:")
    put_greeks = calculate_greeks(spot, strike, time_to_maturity, volatility, 'put', risk_free_rate)
    for key, value in put_greeks.items():
        if key == 'price':
            print(f"  {key.capitalize()}: ${value:.2f}")
        elif key in ['delta', 'gamma', 'theta', 'vega', 'rho']:
            print(f"  {key.capitalize()}: {value:.4f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
    
    print("\n✅ Black-Scholes implementation test completed!")