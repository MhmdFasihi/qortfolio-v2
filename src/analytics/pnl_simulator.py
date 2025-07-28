# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
PnL Simulator with Taylor Expansion - Fixed Parameter Names
Location: src/analytics/pnl_simulator.py

Provides accurate PnL estimation using Taylor expansion for options portfolios.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging

from core.utils.time_utils import calculate_time_to_maturity
from models.options.black_scholes import BlackScholesModel, OptionParameters, OptionType

logger = logging.getLogger(__name__)

@dataclass
class PnLScenario:
    """Single PnL scenario result."""
    spot_change: float
    vol_change: float
    time_decay_days: float
    pnl_taylor: float
    pnl_revaluation: float
    pnl_components: Dict[str, float]
    
@dataclass
class PortfolioPosition:
    """Single portfolio position."""
    symbol: str
    option_type: OptionType
    strike: float
    expiry: datetime
    quantity: float
    current_price: float
    spot_price: float
    volatility: float
    risk_free_rate: float = 0.05

class TaylorExpansionPnL:
    """
    Taylor Expansion PnL Simulator.
    
    Uses Taylor series expansion for fast and accurate PnL estimation:
    ΔC ≈ δΔS + ½γ(ΔS)² + θΔt + νΔσ
    """
    
    def __init__(self):
        self.bs_model = BlackScholesModel()
        self.logger = logging.getLogger(__name__)
        
    def calculate_position_pnl(
        self,
        position: PortfolioPosition,
        spot_shock: float = 0.0,
        vol_shock: float = 0.0,
        time_decay_days: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculate PnL for a single position using Taylor expansion.
        
        Args:
            position: Portfolio position
            spot_shock: Relative spot price change (e.g., 0.1 for +10%)
            vol_shock: Absolute volatility change (e.g., 0.05 for +5pp)
            time_decay_days: Days of time decay
            
        Returns:
            Dictionary with PnL components
        """
        try:
            current_time = datetime.now()
            time_to_maturity = calculate_time_to_maturity(current_time, position.expiry)
            
            # Calculate current Greeks
            params = OptionParameters(
                spot_price=position.spot_price,
                strike_price=position.strike,
                time_to_maturity=time_to_maturity,
                volatility=position.volatility,
                risk_free_rate=position.risk_free_rate,
                option_type=position.option_type
            )
            
            greeks_result = self.bs_model.calculate_greeks(params)
            
            # Price changes
            delta_s = position.spot_price * spot_shock
            delta_vol = vol_shock
            delta_t = time_decay_days / 365.25
            
            # Taylor expansion components
            delta_pnl = greeks_result.delta * delta_s
            gamma_pnl = 0.5 * greeks_result.gamma * (delta_s ** 2)
            theta_pnl = greeks_result.theta * delta_t
            vega_pnl = greeks_result.vega * delta_vol
            
            # Total Taylor PnL per unit
            taylor_pnl_per_unit = delta_pnl + gamma_pnl + theta_pnl + vega_pnl
            
            # Scale by position size
            total_taylor_pnl = taylor_pnl_per_unit * position.quantity
            
            # For validation: calculate exact revaluation
            new_spot = position.spot_price * (1 + spot_shock)
            new_vol = position.volatility + vol_shock
            new_time = time_to_maturity - delta_t
            
            if new_time > 0:
                new_params = OptionParameters(
                    spot_price=new_spot,
                    strike_price=position.strike,
                    time_to_maturity=new_time,
                    volatility=new_vol,
                    risk_free_rate=position.risk_free_rate,
                    option_type=position.option_type
                )
                
                new_price = self.bs_model.calculate_option_price(new_params).option_price
                exact_pnl = (new_price - position.current_price) * position.quantity
            else:
                # Option expired
                if position.option_type == OptionType.CALL:
                    intrinsic = max(new_spot - position.strike, 0)
                else:
                    intrinsic = max(position.strike - new_spot, 0)
                exact_pnl = (intrinsic - position.current_price) * position.quantity
            
            return {
                'delta_pnl': delta_pnl * position.quantity,
                'gamma_pnl': gamma_pnl * position.quantity,
                'theta_pnl': theta_pnl * position.quantity,
                'vega_pnl': vega_pnl * position.quantity,
                'taylor_total': total_taylor_pnl,
                'exact_revaluation': exact_pnl,
                'approximation_error': abs(total_taylor_pnl - exact_pnl),
                'greeks': {
                    'delta': greeks_result.delta,
                    'gamma': greeks_result.gamma,
                    'theta': greeks_result.theta,
                    'vega': greeks_result.vega
                }
            }
            
        except Exception as e:
            self.logger.error(f"PnL calculation failed for position: {e}")
            return self._get_empty_pnl_result()
    
    def calculate_portfolio_pnl(
        self,
        positions: List[PortfolioPosition],
        spot_shock: float = 0.0,
        vol_shock: float = 0.0,
        time_decay_days: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Calculate PnL for entire portfolio.
        
        Args:
            positions: List of portfolio positions
            spot_shock: Relative spot price change
            vol_shock: Absolute volatility change
            time_decay_days: Days of time decay
            
        Returns:
            Portfolio PnL analysis
        """
        try:
            position_results = []
            portfolio_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
            
            for position in positions:
                pos_result = self.calculate_position_pnl(
                    position, spot_shock, vol_shock, time_decay_days
                )
                position_results.append(pos_result)
                
                # Aggregate Greeks
                for greek in portfolio_greeks:
                    portfolio_greeks[greek] += pos_result['greeks'][greek] * position.quantity
            
            # Portfolio totals
            total_taylor_pnl = sum(r['taylor_total'] for r in position_results)
            total_exact_pnl = sum(r['exact_revaluation'] for r in position_results)
            total_delta_pnl = sum(r['delta_pnl'] for r in position_results)
            total_gamma_pnl = sum(r['gamma_pnl'] for r in position_results)
            total_theta_pnl = sum(r['theta_pnl'] for r in position_results)
            total_vega_pnl = sum(r['vega_pnl'] for r in position_results)
            
            return {
                'portfolio_pnl': {
                    'taylor_total': total_taylor_pnl,
                    'exact_revaluation': total_exact_pnl,
                    'approximation_error': abs(total_taylor_pnl - total_exact_pnl),
                    'error_percentage': abs(total_taylor_pnl - total_exact_pnl) / abs(total_exact_pnl) * 100 if total_exact_pnl != 0 else 0
                },
                'pnl_components': {
                    'delta_pnl': total_delta_pnl,
                    'gamma_pnl': total_gamma_pnl,
                    'theta_pnl': total_theta_pnl,
                    'vega_pnl': total_vega_pnl
                },
                'portfolio_greeks': portfolio_greeks,
                'position_details': position_results,
                'scenario': {
                    'spot_shock': spot_shock,
                    'vol_shock': vol_shock,
                    'time_decay_days': time_decay_days
                }
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio PnL calculation failed: {e}")
            return self._get_empty_portfolio_result()
    
    def scenario_analysis(
        self,
        positions: List[PortfolioPosition],
        spot_shocks: List[float] = None,
        vol_shocks: List[float] = None,
        time_horizons: List[float] = None
    ) -> pd.DataFrame:
        """
        Run comprehensive scenario analysis.
        
        Args:
            positions: Portfolio positions
            spot_shocks: List of spot price shock scenarios
            vol_shocks: List of volatility shock scenarios  
            time_horizons: List of time horizons in days
            
        Returns:
            DataFrame with scenario results
        """
        if spot_shocks is None:
            spot_shocks = [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2]
        if vol_shocks is None:
            vol_shocks = [-0.1, 0, 0.1]
        if time_horizons is None:
            time_horizons = [0, 1, 7, 30]
        
        scenarios = []
        
        for spot_shock in spot_shocks:
            for vol_shock in vol_shocks:
                for time_days in time_horizons:
                    result = self.calculate_portfolio_pnl(
                        positions, spot_shock, vol_shock, time_days
                    )
                    
                    scenarios.append({
                        'spot_change_%': spot_shock * 100,
                        'vol_change_pp': vol_shock * 100,
                        'time_decay_days': time_days,
                        'taylor_pnl': result['portfolio_pnl']['taylor_total'],
                        'exact_pnl': result['portfolio_pnl']['exact_revaluation'],
                        'delta_pnl': result['pnl_components']['delta_pnl'],
                        'gamma_pnl': result['pnl_components']['gamma_pnl'],
                        'theta_pnl': result['pnl_components']['theta_pnl'],
                        'vega_pnl': result['pnl_components']['vega_pnl'],
                        'error_%': result['portfolio_pnl']['error_percentage']
                    })
        
        return pd.DataFrame(scenarios)
    
    def _get_empty_pnl_result(self) -> Dict[str, float]:
        """Return empty PnL result structure."""
        return {
            'delta_pnl': 0.0,
            'gamma_pnl': 0.0,
            'theta_pnl': 0.0,
            'vega_pnl': 0.0,
            'taylor_total': 0.0,
            'exact_revaluation': 0.0,
            'approximation_error': 0.0,
            'greeks': {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        }
    
    def _get_empty_portfolio_result(self) -> Dict[str, Any]:
        """Return empty portfolio result structure."""
        return {
            'portfolio_pnl': {
                'taylor_total': 0.0,
                'exact_revaluation': 0.0,
                'approximation_error': 0.0,
                'error_percentage': 0.0
            },
            'pnl_components': {
                'delta_pnl': 0.0,
                'gamma_pnl': 0.0,
                'theta_pnl': 0.0,
                'vega_pnl': 0.0
            },
            'portfolio_greeks': {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0},
            'position_details': [],
            'scenario': {'spot_shock': 0, 'vol_shock': 0, 'time_decay_days': 0}
        }


# Convenience functions
def create_sample_portfolio() -> List[PortfolioPosition]:
    """Create sample portfolio for testing."""
    current_time = datetime.now()
    
    positions = [
        PortfolioPosition(
            symbol="BTC",
            option_type=OptionType.CALL,
            strike=52000,
            expiry=current_time + timedelta(days=30),
            quantity=10,
            current_price=2500,
            spot_price=50000,
            volatility=0.8
        ),
        PortfolioPosition(
            symbol="BTC", 
            option_type=OptionType.PUT,
            strike=48000,
            expiry=current_time + timedelta(days=30),
            quantity=-5,
            current_price=2000,
            spot_price=50000,
            volatility=0.8
        )
    ]
    
    return positions


if __name__ == "__main__":
    # Test the PnL simulator
    print("🧪 Testing Taylor Expansion PnL Simulator")
    print("=" * 50)
    
    simulator = TaylorExpansionPnL()
    positions = create_sample_portfolio()
    
    # Single scenario test
    print("📊 Single Scenario Test:")
    result = simulator.calculate_portfolio_pnl(
        positions, spot_shock=0.1, vol_shock=0.05, time_decay_days=7
    )
    
    print(f"Taylor PnL: ${result['portfolio_pnl']['taylor_total']:,.2f}")
    print(f"Exact PnL: ${result['portfolio_pnl']['exact_revaluation']:,.2f}")
    print(f"Error: {result['portfolio_pnl']['error_percentage']:.2f}%")
    
    # Scenario analysis test
    print("\n📋 Scenario Analysis Test:")
    scenarios_df = simulator.scenario_analysis(positions)
    print(f"Generated {len(scenarios_df)} scenarios")
    print("Sample scenarios:")
    print(scenarios_df.head())
    
    print("\n✅ PnL Simulator test completed!")