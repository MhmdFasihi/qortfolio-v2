# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Advanced Greeks Calculator for Qortfolio V2
Specialized module for calculating and analyzing option Greeks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

from .black_scholes import BlackScholesModel, OptionParameters
from core.logging import get_logger
from core.config import get_config


@dataclass
class GreeksProfile:
    """Complete Greeks profile for an option or portfolio."""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    
    # Second-order Greeks
    volga: Optional[float] = None  # d²V/dσ² (vega convexity)
    vanna: Optional[float] = None  # d²V/dS/dσ (delta-vega sensitivity)
    
    # Risk metrics
    delta_dollar: Optional[float] = None  # Delta in dollar terms
    gamma_dollar: Optional[float] = None  # Gamma in dollar terms
    theta_dollar: Optional[float] = None  # Theta in dollar terms per day
    
    # Portfolio-specific
    net_delta: Optional[float] = None
    net_gamma: Optional[float] = None
    portfolio_value: Optional[float] = None


@dataclass
class RiskMetrics:
    """Risk metrics derived from Greeks."""
    max_loss_1_day: float  # Maximum loss in 1 day (theta decay)
    max_loss_1_percent_move: float  # Loss from 1% spot move
    gamma_risk: float  # Risk from gamma exposure
    vega_risk: float  # Risk from volatility changes
    portfolio_var: Optional[float] = None  # Value at Risk


class GreeksCalculator:
    """
    Advanced Greeks calculator with portfolio analysis.
    
    Features:
    - All first and second-order Greeks
    - Portfolio-level aggregation
    - Risk metrics calculation
    - Greeks P&L attribution
    - Sensitivity analysis
    """
    
    def __init__(self):
        """Initialize Greeks calculator."""
        self.config = get_config()
        self.logger = get_logger("greeks_calculator")
        self.bs_model = BlackScholesModel()
        
        self.logger.info("Advanced Greeks calculator initialized")
    
    def calculate_portfolio_greeks(self, positions: List[Dict], 
                                 current_spot: Optional[float] = None) -> GreeksProfile:
        """
        Calculate comprehensive portfolio Greeks.
        
        Args:
            positions: List of position dictionaries containing:
                - quantity: Number of contracts (+ for long, - for short)
                - option_params: Dictionary with spot_price, strike_price, etc.
                - Or individual parameters: spot_price, strike_price, time_to_maturity, volatility, option_type
            current_spot: Current spot price for all positions (if provided)
            
        Returns:
            Complete Greeks profile for the portfolio
        """
        if not positions:
            return GreeksProfile(0.0, 0.0, 0.0, 0.0, 0.0)
        
        portfolio_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0,
            'total_value': 0.0,
            'delta_dollar': 0.0,
            'gamma_dollar': 0.0,
            'theta_dollar': 0.0
        }
        
        successful_positions = 0
        
        for i, position in enumerate(positions):
            try:
                quantity = position['quantity']
                
                # Handle different parameter formats
                if 'option_params' in position:
                    params_dict = position['option_params']
                else:
                    params_dict = position
                
                # Override spot price if provided
                if current_spot is not None:
                    params_dict = params_dict.copy()
                    params_dict['spot_price'] = current_spot
                
                # Create option parameters
                params = OptionParameters(
                    spot_price=params_dict['spot_price'],
                    strike_price=params_dict['strike_price'],
                    time_to_maturity=params_dict['time_to_maturity'],
                    volatility=params_dict['volatility'],
                    risk_free_rate=params_dict.get('risk_free_rate', 0.05),
                    option_type=params_dict['option_type'],
                    dividend_yield=params_dict.get('dividend_yield', 0.0)
                )
                
                # Calculate Greeks for this position
                result = self.bs_model.calculate_option_price(params)
                
                # Add to portfolio totals
                portfolio_greeks['delta'] += quantity * result.delta
                portfolio_greeks['gamma'] += quantity * result.gamma
                portfolio_greeks['theta'] += quantity * result.theta
                portfolio_greeks['vega'] += quantity * result.vega
                portfolio_greeks['rho'] += quantity * result.rho
                portfolio_greeks['total_value'] += quantity * result.option_price
                
                # Calculate dollar Greeks
                spot_price = params.spot_price
                portfolio_greeks['delta_dollar'] += quantity * result.delta * spot_price
                portfolio_greeks['gamma_dollar'] += quantity * result.gamma * spot_price * spot_price / 100
                portfolio_greeks['theta_dollar'] += quantity * result.theta
                
                successful_positions += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate Greeks for position {i}: {e}")
                continue
        
        if successful_positions == 0:
            self.logger.error("No positions could be processed for Greeks calculation")
            return GreeksProfile(0.0, 0.0, 0.0, 0.0, 0.0)
        
        self.logger.info(f"Portfolio Greeks calculated for {successful_positions} positions", extra={
            "portfolio_delta": portfolio_greeks['delta'],
            "portfolio_gamma": portfolio_greeks['gamma'],
            "portfolio_value": portfolio_greeks['total_value']
        })
        
        return GreeksProfile(
            delta=portfolio_greeks['delta'],
            gamma=portfolio_greeks['gamma'],
            theta=portfolio_greeks['theta'],
            vega=portfolio_greeks['vega'],
            rho=portfolio_greeks['rho'],
            delta_dollar=portfolio_greeks['delta_dollar'],
            gamma_dollar=portfolio_greeks['gamma_dollar'],
            theta_dollar=portfolio_greeks['theta_dollar'],
            net_delta=portfolio_greeks['delta'],
            net_gamma=portfolio_greeks['gamma'],
            portfolio_value=portfolio_greeks['total_value']
        )
    
    def calculate_risk_metrics(self, greeks: GreeksProfile, 
                              current_spot: float,
                              volatility_shock: float = 0.1) -> RiskMetrics:
        """
        Calculate risk metrics from Greeks.
        
        Args:
            greeks: Portfolio Greeks profile
            current_spot: Current underlying spot price
            volatility_shock: Volatility shock size (default 10%)
            
        Returns:
            Risk metrics
        """
        # Theta risk (time decay over 1 day)
        max_loss_1_day = abs(greeks.theta_dollar or greeks.theta) if greeks.theta_dollar else abs(greeks.theta)
        
        # Delta risk (1% spot move)
        spot_move_1_percent = current_spot * 0.01
        delta_pnl = (greeks.delta_dollar or greeks.delta * current_spot) * 0.01
        gamma_pnl = 0.5 * (greeks.gamma_dollar or greeks.gamma * current_spot * current_spot / 100) * (0.01 ** 2)
        max_loss_1_percent_move = abs(delta_pnl + gamma_pnl)
        
        # Gamma risk (convexity risk from large moves)
        # Assuming 5% spot move as stress scenario
        large_move = 0.05
        gamma_risk = abs(0.5 * (greeks.gamma_dollar or greeks.gamma * current_spot * current_spot / 100) * (large_move ** 2))
        
        # Vega risk (volatility change)
        vega_risk = abs(greeks.vega * volatility_shock * 100)  # 10% vol shock
        
        return RiskMetrics(
            max_loss_1_day=max_loss_1_day,
            max_loss_1_percent_move=max_loss_1_percent_move,
            gamma_risk=gamma_risk,
            vega_risk=vega_risk
        )
    
    def calculate_pnl_attribution(self, greeks: GreeksProfile,
                                 spot_change: float, time_decay: float,
                                 volatility_change: float, rate_change: float = 0.0) -> Dict[str, float]:
        """
        Attribute P&L changes to different Greeks.
        
        Args:
            greeks: Portfolio Greeks
            spot_change: Change in spot price (absolute)
            time_decay: Time decay (in days)
            volatility_change: Change in volatility (absolute, e.g., 0.1 for 10%)
            rate_change: Change in interest rate (absolute, e.g., 0.01 for 1%)
            
        Returns:
            Dictionary with P&L attribution
        """
        # First-order effects
        delta_pnl = greeks.delta * spot_change
        theta_pnl = greeks.theta * time_decay
        vega_pnl = greeks.vega * volatility_change * 100  # vega per 1% vol change
        rho_pnl = greeks.rho * rate_change * 100  # rho per 1% rate change
        
        # Second-order effects
        gamma_pnl = 0.5 * greeks.gamma * (spot_change ** 2)
        
        # Total explained P&L
        total_explained = delta_pnl + gamma_pnl + theta_pnl + vega_pnl + rho_pnl
        
        attribution = {
            'delta_pnl': delta_pnl,
            'gamma_pnl': gamma_pnl,
            'theta_pnl': theta_pnl,
            'vega_pnl': vega_pnl,
            'rho_pnl': rho_pnl,
            'total_explained': total_explained,
            'spot_change': spot_change,
            'time_decay': time_decay,
            'volatility_change': volatility_change,
            'rate_change': rate_change
        }
        
        self.logger.debug("P&L attribution calculated", extra=attribution)
        
        return attribution
    
    def calculate_hedging_ratios(self, portfolio_greeks: GreeksProfile,
                               current_spot: float) -> Dict[str, float]:
        """
        Calculate hedging ratios to neutralize portfolio Greeks.
        
        Args:
            portfolio_greeks: Current portfolio Greeks
            current_spot: Current spot price
            
        Returns:
            Dictionary with hedging ratios
        """
        hedging_ratios = {}
        
        # Delta hedge: number of underlying units to hedge
        if portfolio_greeks.delta != 0:
            delta_hedge_ratio = -portfolio_greeks.delta
            hedging_ratios['delta_hedge_units'] = delta_hedge_ratio
            hedging_ratios['delta_hedge_notional'] = delta_hedge_ratio * current_spot
        
        # Gamma hedge: typically requires options
        # This is a simplified approximation
        if portfolio_greeks.gamma != 0:
            hedging_ratios['gamma_exposure'] = portfolio_greeks.gamma
            hedging_ratios['gamma_risk_5pct_move'] = 0.5 * portfolio_greeks.gamma * (current_spot * 0.05) ** 2
        
        # Vega hedge: requires options with different expiries/strikes
        if portfolio_greeks.vega != 0:
            hedging_ratios['vega_exposure'] = portfolio_greeks.vega
            hedging_ratios['vega_risk_10pct_vol'] = portfolio_greeks.vega * 10  # 10% vol move
        
        return hedging_ratios
    
    def simulate_pnl_scenarios(self, greeks: GreeksProfile, current_spot: float,
                              spot_scenarios: List[float], vol_scenarios: List[float],
                              time_decay_days: float = 1.0) -> pd.DataFrame:
        """
        Simulate P&L under different market scenarios.
        
        Args:
            greeks: Portfolio Greeks
            current_spot: Current spot price
            spot_scenarios: List of spot price changes (as percentages, e.g., [-0.1, 0, 0.1])
            vol_scenarios: List of volatility changes (absolute, e.g., [-0.1, 0, 0.1])
            time_decay_days: Number of days for theta decay
            
        Returns:
            DataFrame with P&L scenarios
        """
        scenarios = []
        
        for spot_pct_change in spot_scenarios:
            for vol_change in vol_scenarios:
                spot_change = current_spot * spot_pct_change
                
                # Calculate P&L attribution
                pnl_attr = self.calculate_pnl_attribution(
                    greeks=greeks,
                    spot_change=spot_change,
                    time_decay=time_decay_days,
                    volatility_change=vol_change
                )
                
                scenarios.append({
                    'spot_change_pct': spot_pct_change * 100,
                    'spot_change_abs': spot_change,
                    'vol_change': vol_change * 100,  # Convert to percentage
                    'new_spot_price': current_spot + spot_change,
                    'delta_pnl': pnl_attr['delta_pnl'],
                    'gamma_pnl': pnl_attr['gamma_pnl'],
                    'theta_pnl': pnl_attr['theta_pnl'],
                    'vega_pnl': pnl_attr['vega_pnl'],
                    'total_pnl': pnl_attr['total_explained']
                })
        
        return pd.DataFrame(scenarios)
    
    def analyze_greeks_evolution(self, positions: List[Dict], 
                               days_forward: List[int] = [1, 7, 14, 30]) -> pd.DataFrame:
        """
        Analyze how Greeks evolve over time.
        
        Args:
            positions: Portfolio positions
            days_forward: List of days to project forward
            
        Returns:
            DataFrame with Greeks evolution
        """
        evolution_data = []
        
        for days in days_forward:
            # Update time to maturity for all positions
            updated_positions = []
            for position in positions:
                updated_position = position.copy()
                
                # Get original parameters
                if 'option_params' in position:
                    params = position['option_params'].copy()
                else:
                    params = position.copy()
                
                # Reduce time to maturity
                original_tte = params['time_to_maturity']
                new_tte = max(original_tte - days / 365.25, 1/365.25)  # Minimum 1 day
                params['time_to_maturity'] = new_tte
                
                if 'option_params' in position:
                    updated_position['option_params'] = params
                else:
                    updated_position.update(params)
                
                updated_positions.append(updated_position)
            
            # Calculate Greeks for this time point
            try:
                greeks = self.calculate_portfolio_greeks(updated_positions)
                
                evolution_data.append({
                    'days_forward': days,
                    'delta': greeks.delta,
                    'gamma': greeks.gamma,
                    'theta': greeks.theta,
                    'vega': greeks.vega,
                    'rho': greeks.rho,
                    'portfolio_value': greeks.portfolio_value
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate Greeks for {days} days forward: {e}")
                continue
        
        return pd.DataFrame(evolution_data)
    
    def generate_greeks_report(self, positions: List[Dict], 
                             current_spot: float) -> Dict[str, any]:
        """
        Generate comprehensive Greeks analysis report.
        
        Args:
            positions: Portfolio positions
            current_spot: Current underlying spot price
            
        Returns:
            Dictionary with complete Greeks analysis
        """
        # Calculate current Greeks
        greeks = self.calculate_portfolio_greeks(positions, current_spot)
        
        # Calculate risk metrics
        risk_metrics = self.calculate_risk_metrics(greeks, current_spot)
        
        # Calculate hedging ratios
        hedging_ratios = self.calculate_hedging_ratios(greeks, current_spot)
        
        # Scenario analysis
        spot_scenarios = [-0.1, -0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05, 0.1]
        vol_scenarios = [-0.1, -0.05, 0, 0.05, 0.1]
        scenario_analysis = self.simulate_pnl_scenarios(
            greeks, current_spot, spot_scenarios, vol_scenarios
        )
        
        # Greeks evolution
        greeks_evolution = self.analyze_greeks_evolution(positions)
        
        report = {
            'timestamp': datetime.now(),
            'current_spot': current_spot,
            'portfolio_greeks': greeks,
            'risk_metrics': risk_metrics,
            'hedging_ratios': hedging_ratios,
            'scenario_analysis': scenario_analysis,
            'greeks_evolution': greeks_evolution,
            'position_count': len(positions),
            'summary': {
                'net_delta_exposure': greeks.delta_dollar or greeks.delta * current_spot,
                'max_daily_theta_loss': risk_metrics.max_loss_1_day,
                'gamma_risk_5pct_move': risk_metrics.gamma_risk,
                'vega_risk_10pct_vol': risk_metrics.vega_risk,
                'portfolio_value': greeks.portfolio_value
            }
        }
        
        self.logger.info("Comprehensive Greeks report generated", extra={
            "position_count": len(positions),
            "portfolio_value": greeks.portfolio_value,
            "net_delta": greeks.delta,
            "net_gamma": greeks.gamma
        })
        
        return report


# Convenience functions
def calculate_option_greeks(spot: float, strike: float, time_to_maturity: float,
                          volatility: float, option_type: str = 'call',
                          risk_free_rate: float = 0.05) -> GreeksProfile:
    """
    Calculate Greeks for a single option.
    
    Args:
        spot: Current spot price
        strike: Strike price
        time_to_maturity: Time to maturity in years
        volatility: Implied volatility
        option_type: 'call' or 'put'
        risk_free_rate: Risk-free rate
        
    Returns:
        Greeks profile for the option
    """
    calculator = GreeksCalculator()
    
    position = {
        'quantity': 1.0,
        'spot_price': spot,
        'strike_price': strike,
        'time_to_maturity': time_to_maturity,
        'volatility': volatility,
        'risk_free_rate': risk_free_rate,
        'option_type': option_type
    }
    
    return calculator.calculate_portfolio_greeks([position], spot)


def analyze_portfolio_risk(positions: List[Dict], current_spot: float) -> Dict[str, float]:
    """
    Quick portfolio risk analysis.
    
    Args:
        positions: Portfolio positions
        current_spot: Current spot price
        
    Returns:
        Dictionary with key risk metrics
    """
    calculator = GreeksCalculator()
    greeks = calculator.calculate_portfolio_greeks(positions, current_spot)
    risk_metrics = calculator.calculate_risk_metrics(greeks, current_spot)
    
    return {
        'portfolio_delta': greeks.delta,
        'portfolio_gamma': greeks.gamma,
        'daily_theta_decay': abs(greeks.theta),
        'max_loss_1_day': risk_metrics.max_loss_1_day,
        'max_loss_1pct_move': risk_metrics.max_loss_1_percent_move,
        'gamma_risk': risk_metrics.gamma_risk,
        'vega_risk': risk_metrics.vega_risk,
        'portfolio_value': greeks.portfolio_value
    }


if __name__ == "__main__":
    # Test the Greeks calculator
    print("🧪 Testing Advanced Greeks Calculator")
    print("=" * 40)
    
    calculator = GreeksCalculator()
    
    # Create sample portfolio
    current_spot = 50000.0
    
    sample_positions = [
        {
            'quantity': 10,  # Long 10 calls
            'spot_price': current_spot,
            'strike_price': 52000,
            'time_to_maturity': 30 / 365.25,
            'volatility': 0.8,
            'risk_free_rate': 0.05,
            'option_type': 'call'
        },
        {
            'quantity': -5,  # Short 5 puts
            'spot_price': current_spot,
            'strike_price': 48000,
            'time_to_maturity': 30 / 365.25,
            'volatility': 0.8,
            'risk_free_rate': 0.05,
            'option_type': 'put'
        }
    ]
    
    # Calculate portfolio Greeks
    print("📊 Portfolio Greeks Analysis:")
    greeks = calculator.calculate_portfolio_greeks(sample_positions, current_spot)
    
    print(f"  Portfolio Delta: {greeks.delta:.4f}")
    print(f"  Portfolio Gamma: {greeks.gamma:.6f}")
    print(f"  Portfolio Theta: {greeks.theta:.4f}")
    print(f"  Portfolio Vega: {greeks.vega:.4f}")
    print(f"  Portfolio Value: ${greeks.portfolio_value:.2f}")
    
    # Risk analysis
    print(f"\n⚠️ Risk Analysis:")
    risk_metrics = calculator.calculate_risk_metrics(greeks, current_spot)
    print(f"  Max daily theta loss: ${risk_metrics.max_loss_1_day:.2f}")
    print(f"  Risk from 1% move: ${risk_metrics.max_loss_1_percent_move:.2f}")
    print(f"  Gamma risk (5% move): ${risk_metrics.gamma_risk:.2f}")
    print(f"  Vega risk (10% vol): ${risk_metrics.vega_risk:.2f}")
    
    # P&L attribution example
    print(f"\n💰 P&L Attribution (Example: +$1000 spot, -1 day, +5% vol):")
    pnl_attr = calculator.calculate_pnl_attribution(
        greeks=greeks,
        spot_change=1000,
        time_decay=1,
        volatility_change=0.05
    )
    
    for key, value in pnl_attr.items():
        if 'pnl' in key:
            print(f"  {key.replace('_', ' ').title()}: ${value:.2f}")
    
    print("\n✅ Advanced Greeks calculator test completed!")