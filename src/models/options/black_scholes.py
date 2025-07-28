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

from core.utils.time_utils import calculate_time_to_maturity
from core.logging import get_logger
from core.config import get_config


@dataclass
class OptionParameters:
    """Option parameters for pricing calculations."""
    spot_price: float
    strike_price: float
    time_to_maturity: float
    volatility: float
    risk_free_rate: float
    option_type: str  # 'call' or 'put'
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
        option_type = params.option_type.lower()
        
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
    
    def calculate_option_price_simple(self, spot: float, strike: float, 
                                    time_to_maturity: float, volatility: float,
                                    option_type: str = 'call', 
                                    risk_free_rate: Optional[float] = None) -> float:
        """
        Simplified option pricing for quick calculations.
        
        Args:
            spot: Current spot price
            strike: Strike price
            time_to_maturity: Time to maturity in years
            volatility: Implied volatility (annualized)
            option_type: 'call' or 'put'
            risk_free_rate: Risk-free rate (uses default if None)
            
        Returns:
            Option price
        """
        params = OptionParameters(
            spot_price=spot,
            strike_price=strike,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            risk_free_rate=risk_free_rate or self.default_risk_free_rate,
            option_type=option_type
        )
        
        result = self.calculate_option_price(params)
        return result.option_price
    
    def calculate_implied_volatility(self, market_price: float, spot: float, 
                                   strike: float, time_to_maturity: float,
                                   option_type: str = 'call',
                                   risk_free_rate: Optional[float] = None,
                                   max_iterations: int = 100,
                                   tolerance: float = 1e-6) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            market_price: Observed market price
            spot: Current spot price
            strike: Strike price
            time_to_maturity: Time to maturity in years
            option_type: 'call' or 'put'
            risk_free_rate: Risk-free rate
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance
            
        Returns:
            Implied volatility or None if convergence failed
        """
        risk_free_rate = risk_free_rate or self.default_risk_free_rate
        
        # Initial guess (using rule of thumb)
        vol = 0.2  # Start with 20% volatility
        
        for i in range(max_iterations):
            try:
                # Calculate theoretical price and vega
                params = OptionParameters(
                    spot_price=spot,
                    strike_price=strike,
                    time_to_maturity=time_to_maturity,
                    volatility=vol,
                    risk_free_rate=risk_free_rate,
                    option_type=option_type
                )
                
                result = self.calculate_option_price(params)
                theoretical_price = result.option_price
                vega = result.vega
                
                # Price difference
                price_diff = theoretical_price - market_price
                
                # Check convergence
                if abs(price_diff) < tolerance:
                    self.logger.debug(f"IV converged in {i+1} iterations", extra={
                        "implied_volatility": vol,
                        "market_price": market_price,
                        "theoretical_price": theoretical_price
                    })
                    return vol
                
                # Newton-Raphson update
                if abs(vega) < 1e-10:  # Avoid division by zero
                    break
                
                vol = vol - price_diff / (vega * 100)  # vega is per 1% vol change
                
                # Keep volatility in reasonable bounds
                vol = max(0.001, min(vol, 10.0))  # 0.1% to 1000%
                
            except Exception as e:
                self.logger.warning(f"IV calculation iteration {i} failed: {e}")
                break
        
        self.logger.warning("Implied volatility calculation did not converge", extra={
            "market_price": market_price,
            "spot": spot,
            "strike": strike,
            "time_to_maturity": time_to_maturity
        })
        return None
    
    def calculate_portfolio_greeks(self, positions: List[Dict]) -> Dict[str, float]:
        """
        Calculate portfolio-level Greeks.
        
        Args:
            positions: List of position dictionaries with:
                - quantity: Number of contracts (positive for long, negative for short)
                - spot_price, strike_price, time_to_maturity, volatility, risk_free_rate, option_type
                
        Returns:
            Dictionary with portfolio Greeks
        """
        portfolio_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0,
            'total_value': 0.0
        }
        
        for position in positions:
            try:
                quantity = position['quantity']
                
                # Create option parameters
                params = OptionParameters(
                    spot_price=position['spot_price'],
                    strike_price=position['strike_price'],
                    time_to_maturity=position['time_to_maturity'],
                    volatility=position['volatility'],
                    risk_free_rate=position.get('risk_free_rate', self.default_risk_free_rate),
                    option_type=position['option_type'],
                    dividend_yield=position.get('dividend_yield', 0.0)
                )
                
                # Calculate option pricing
                result = self.calculate_option_price(params)
                
                # Add to portfolio totals
                portfolio_greeks['delta'] += quantity * result.delta
                portfolio_greeks['gamma'] += quantity * result.gamma
                portfolio_greeks['theta'] += quantity * result.theta
                portfolio_greeks['vega'] += quantity * result.vega
                portfolio_greeks['rho'] += quantity * result.rho
                portfolio_greeks['total_value'] += quantity * result.option_price
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate Greeks for position: {e}")
                continue
        
        return portfolio_greeks
    
    # Private methods for calculations
    
    def _calculate_d1_d2(self, S: float, K: float, T: float, r: float, σ: float, q: float = 0.0) -> Tuple[float, float]:
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
        if params.option_type.lower() not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")


# Vectorized functions for batch calculations
def calculate_black_scholes_vectorized(spot_prices: np.ndarray, strike_prices: np.ndarray,
                                     times_to_maturity: np.ndarray, volatilities: np.ndarray,
                                     risk_free_rates: np.ndarray, option_types: np.ndarray) -> pd.DataFrame:
    """
    Vectorized Black-Scholes calculation for multiple options.
    
    Args:
        spot_prices: Array of spot prices
        strike_prices: Array of strike prices
        times_to_maturity: Array of times to maturity
        volatilities: Array of volatilities
        risk_free_rates: Array of risk-free rates
        option_types: Array of option types ('call' or 'put')
        
    Returns:
        DataFrame with pricing results
    """
    # This would be implemented for high-performance batch calculations
    # For now, we'll use the single-option method in a loop
    
    bs_model = BlackScholesModel()
    results = []
    
    for i in range(len(spot_prices)):
        try:
            params = OptionParameters(
                spot_price=spot_prices[i],
                strike_price=strike_prices[i],
                time_to_maturity=times_to_maturity[i],
                volatility=volatilities[i],
                risk_free_rate=risk_free_rates[i],
                option_type=option_types[i]
            )
            
            result = bs_model.calculate_option_price(params)
            results.append({
                'spot_price': spot_prices[i],
                'strike_price': strike_prices[i],
                'time_to_maturity': times_to_maturity[i],
                'volatility': volatilities[i],
                'option_type': option_types[i],
                'option_price': result.option_price,
                'delta': result.delta,
                'gamma': result.gamma,
                'theta': result.theta,
                'vega': result.vega,
                'rho': result.rho,
                'intrinsic_value': result.intrinsic_value,
                'time_value': result.time_value,
                'moneyness': result.moneyness
            })
            
        except Exception as e:
            # Handle individual failures
            results.append({
                'spot_price': spot_prices[i],
                'strike_price': strike_prices[i],
                'error': str(e)
            })
    
    return pd.DataFrame(results)


# Convenience functions
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
    return bs_model.calculate_option_price_simple(
        spot, strike, time_to_maturity, volatility, option_type, risk_free_rate
    )


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
        option_type=option_type
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
    
    # Test implied volatility calculation
    print(f"\n🔍 Implied Volatility Test:")
    market_price = call_greeks['price']
    implied_vol = bs_model.calculate_implied_volatility(
        market_price, spot, strike, time_to_maturity, 'call', risk_free_rate
    )
    if implied_vol:
        print(f"  Market price: ${market_price:.2f}")
        print(f"  Implied volatility: {implied_vol:.1%}")
        print(f"  Original volatility: {volatility:.1%}")
        print(f"  Difference: {abs(implied_vol - volatility):.4f}")
    
    print("\n✅ Black-Scholes implementation test completed!")