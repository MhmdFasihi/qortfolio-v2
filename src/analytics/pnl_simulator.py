# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Taylor Expansion PnL Simulator for Qortfolio V2
Advanced PnL simulation using Taylor series expansion

Core Formula: ΔC ≈ δΔS + ½γ(ΔS)² + θΔt + νΔσ + ρΔr + higher order terms

This was a key feature request - implementing proper Taylor expansion for options PnL analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import norm

from models.options.black_scholes import BlackScholesModel, OptionParameters
from models.options.greeks_calculator import GreeksCalculator, GreeksProfile
from core.logging import get_logger
from core.config import get_config


@dataclass
class MarketScenario:
    """Market scenario for PnL simulation."""
    spot_change: float  # Absolute change in spot price
    time_change: float  # Time change in days
    volatility_change: float  # Absolute change in volatility
    rate_change: float = 0.0  # Change in risk-free rate
    scenario_name: str = ""


@dataclass
class PnLResult:
    """PnL simulation result."""
    scenario: MarketScenario
    
    # Taylor expansion components
    delta_pnl: float
    gamma_pnl: float
    theta_pnl: float
    vega_pnl: float
    rho_pnl: float
    
    # Higher order terms
    second_order_pnl: float
    cross_gamma_pnl: float = 0.0  # Cross-gamma effects
    
    # Total P&L
    taylor_total_pnl: float
    
    # Validation
    actual_pnl: Optional[float] = None  # True Black-Scholes PnL
    taylor_error: Optional[float] = None  # Difference between Taylor and actual
    
    # Metrics
    largest_component: str = ""
    risk_contribution: Dict[str, float] = None


class TaylorPnLSimulator:
    """
    Taylor expansion-based PnL simulator for options portfolios.
    
    Implements the mathematical formula:
    ΔC ≈ δΔS + ½γ(ΔS)² + θΔt + νΔσ + ρΔr + second-order terms
    
    Features:
    - Complete Taylor series implementation
    - Portfolio-level analysis
    - Scenario stress testing
    - Error analysis vs. actual Black-Scholes
    - Risk attribution by Greek
    """
    
    def __init__(self):
        """Initialize Taylor PnL simulator."""
        self.config = get_config()
        self.logger = get_logger("pnl_simulator")
        self.bs_model = BlackScholesModel()
        self.greeks_calculator = GreeksCalculator()
        
        self.logger.info("Taylor expansion PnL simulator initialized")
    
    def simulate_pnl(self, positions: List[Dict], scenarios: List[MarketScenario],
                    include_second_order: bool = True, 
                    validate_with_bs: bool = True) -> List[PnLResult]:
        """
        Simulate portfolio PnL using Taylor expansion.
        
        Args:
            positions: Portfolio positions
            scenarios: List of market scenarios to simulate
            include_second_order: Include second-order terms
            validate_with_bs: Validate against actual Black-Scholes pricing
            
        Returns:
            List of PnL results for each scenario
        """
        # Calculate current portfolio Greeks
        current_greeks = self.greeks_calculator.calculate_portfolio_greeks(positions)
        
        results = []
        
        for scenario in scenarios:
            try:
                # Calculate Taylor expansion PnL
                pnl_result = self._calculate_taylor_pnl(
                    greeks=current_greeks,
                    scenario=scenario,
                    include_second_order=include_second_order
                )
                
                # Validate with actual Black-Scholes if requested
                if validate_with_bs:
                    actual_pnl = self._calculate_actual_pnl(positions, scenario)
                    pnl_result.actual_pnl = actual_pnl
                    pnl_result.taylor_error = pnl_result.taylor_total_pnl - actual_pnl
                
                # Identify largest P&L component
                pnl_result.largest_component = self._identify_largest_component(pnl_result)
                
                # Calculate risk contribution by Greek
                pnl_result.risk_contribution = self._calculate_risk_contribution(pnl_result)
                
                results.append(pnl_result)
                
            except Exception as e:
                self.logger.warning(f"Failed to simulate scenario {scenario.scenario_name}: {e}")
                continue
        
        self.logger.info(f"PnL simulation completed for {len(results)} scenarios", extra={
            "portfolio_delta": current_greeks.delta,
            "portfolio_gamma": current_greeks.gamma,
            "scenarios_count": len(scenarios)
        })
        
        return results
    
    def create_stress_scenarios(self, base_spot: float, 
                              max_spot_move_pct: float = 0.2,
                              max_vol_move: float = 0.3,
                              time_horizons: List[int] = [1, 7, 30]) -> List[MarketScenario]:
        """
        Create comprehensive stress test scenarios.
        
        Args:
            base_spot: Current spot price
            max_spot_move_pct: Maximum spot move as percentage (e.g., 0.2 for ±20%)
            max_vol_move: Maximum volatility move (absolute, e.g., 0.3 for ±30%)
            time_horizons: Time horizons in days
            
        Returns:
            List of stress test scenarios
        """
        scenarios = []
        
        # Spot move scenarios
        spot_moves = [-max_spot_move_pct, -max_spot_move_pct/2, 0, max_spot_move_pct/2, max_spot_move_pct]
        vol_moves = [-max_vol_move, -max_vol_move/2, 0, max_vol_move/2, max_vol_move]
        
        scenario_id = 1
        
        for time_days in time_horizons:
            for spot_pct in spot_moves:
                for vol_change in vol_moves:
                    spot_change = base_spot * spot_pct
                    
                    scenarios.append(MarketScenario(
                        spot_change=spot_change,
                        time_change=time_days,
                        volatility_change=vol_change,
                        rate_change=0.0,
                        scenario_name=f"Stress_{scenario_id:03d}_T{time_days}_S{spot_pct:.1%}_V{vol_change:.1f}"
                    ))
                    scenario_id += 1
        
        # Add extreme scenarios
        extreme_scenarios = [
            MarketScenario(base_spot * -0.3, 1, 0.5, 0.0, "Extreme_Crash"),
            MarketScenario(base_spot * 0.3, 1, -0.3, 0.0, "Extreme_Rally"),
            MarketScenario(0, 30, 0.8, 0.0, "Vol_Explosion"),
            MarketScenario(0, 30, -0.5, 0.0, "Vol_Collapse"),
            MarketScenario(base_spot * -0.1, 1, 0.3, 0.02, "Rate_Shock_Down"),
            MarketScenario(base_spot * 0.1, 1, -0.1, -0.02, "Rate_Shock_Up")
        ]
        
        scenarios.extend(extreme_scenarios)
        
        self.logger.info(f"Created {len(scenarios)} stress test scenarios")
        
        return scenarios
    
    def analyze_taylor_accuracy(self, pnl_results: List[PnLResult]) -> Dict[str, any]:
        """
        Analyze accuracy of Taylor expansion vs. actual Black-Scholes.
        
        Args:
            pnl_results: List of PnL results with validation data
            
        Returns:
            Dictionary with accuracy analysis
        """
        if not pnl_results or not any(r.actual_pnl is not None for r in pnl_results):
            return {"error": "No validation data available"}
        
        # Filter results with validation data
        validated_results = [r for r in pnl_results if r.actual_pnl is not None]
        
        # Calculate error statistics
        errors = [r.taylor_error for r in validated_results]
        taylor_pnls = [r.taylor_total_pnl for r in validated_results]
        actual_pnls = [r.actual_pnl for r in validated_results]
        
        # Relative errors (avoid division by zero)
        relative_errors = []
        for i, actual in enumerate(actual_pnls):
            if abs(actual) > 1e-6:  # Avoid division by very small numbers
                relative_errors.append(abs(errors[i]) / abs(actual))
            else:
                relative_errors.append(0.0)
        
        analysis = {
            "total_scenarios": len(validated_results),
            "error_statistics": {
                "mean_absolute_error": np.mean(np.abs(errors)),
                "max_absolute_error": np.max(np.abs(errors)),
                "rmse": np.sqrt(np.mean(np.array(errors) ** 2)),
                "mean_relative_error": np.mean(relative_errors),
                "max_relative_error": np.max(relative_errors)
            },
            "accuracy_by_scenario_type": {},
            "worst_case_scenarios": []
        }
        
        # Find worst-case scenarios
        error_indices = np.argsort(np.abs(errors))[-5:]  # Top 5 worst
        for idx in reversed(error_indices):
            result = validated_results[idx]
            analysis["worst_case_scenarios"].append({
                "scenario": result.scenario.scenario_name,
                "taylor_pnl": result.taylor_total_pnl,
                "actual_pnl": result.actual_pnl,
                "error": result.taylor_error,
                "relative_error": relative_errors[idx] if idx < len(relative_errors) else 0.0
            })
        
        # Accuracy by scenario characteristics
        large_moves = [r for r in validated_results if abs(r.scenario.spot_change) > 1000]
        small_moves = [r for r in validated_results if abs(r.scenario.spot_change) <= 1000]
        
        if large_moves:
            large_move_errors = [r.taylor_error for r in large_moves]
            analysis["accuracy_by_scenario_type"]["large_moves"] = {
                "count": len(large_moves),
                "mean_error": np.mean(np.abs(large_move_errors)),
                "max_error": np.max(np.abs(large_move_errors))
            }
        
        if small_moves:
            small_move_errors = [r.taylor_error for r in small_moves]
            analysis["accuracy_by_scenario_type"]["small_moves"] = {
                "count": len(small_moves),
                "mean_error": np.mean(np.abs(small_move_errors)),
                "max_error": np.max(np.abs(small_move_errors))
            }
        
        self.logger.info("Taylor expansion accuracy analysis completed", extra={
            "scenarios_analyzed": len(validated_results),
            "mean_absolute_error": analysis["error_statistics"]["mean_absolute_error"],
            "mean_relative_error": analysis["error_statistics"]["mean_relative_error"]
        })
        
        return analysis
    
    def generate_pnl_report(self, positions: List[Dict], 
                           base_spot: float) -> Dict[str, any]:
        """
        Generate comprehensive PnL analysis report.
        
        Args:
            positions: Portfolio positions
            base_spot: Current spot price
            
        Returns:
            Complete PnL analysis report
        """
        # Create scenarios
        scenarios = self.create_stress_scenarios(base_spot)
        
        # Run simulation
        pnl_results = self.simulate_pnl(positions, scenarios, 
                                      include_second_order=True, 
                                      validate_with_bs=True)
        
        # Analyze accuracy
        accuracy_analysis = self.analyze_taylor_accuracy(pnl_results)
        
        # Create summary statistics
        summary_stats = self._create_summary_statistics(pnl_results)
        
        # Risk analysis
        risk_analysis = self._analyze_risk_scenarios(pnl_results)
        
        report = {
            "timestamp": datetime.now(),
            "base_spot_price": base_spot,
            "portfolio_size": len(positions),
            "scenarios_analyzed": len(pnl_results),
            "summary_statistics": summary_stats,
            "accuracy_analysis": accuracy_analysis,
            "risk_analysis": risk_analysis,
            "detailed_results": pnl_results[:20],  # Include top 20 for detail
            "methodology": {
                "description": "Taylor expansion PnL simulation using: ΔC ≈ δΔS + ½γ(ΔS)² + θΔt + νΔσ + ρΔr",
                "validation": "Results validated against full Black-Scholes revaluation",
                "scenarios": "Comprehensive stress testing across spot, volatility, and time dimensions"
            }
        }
        
        self.logger.info("Comprehensive PnL report generated", extra={
            "portfolio_size": len(positions),
            "scenarios_count": len(pnl_results),
            "max_pnl": summary_stats.get("max_pnl", 0),
            "min_pnl": summary_stats.get("min_pnl", 0)
        })
        
        return report
    
    # Private helper methods
    
    def _calculate_taylor_pnl(self, greeks: GreeksProfile, scenario: MarketScenario,
                            include_second_order: bool = True) -> PnLResult:
        """Calculate PnL using Taylor expansion."""
        
        # First-order terms
        delta_pnl = greeks.delta * scenario.spot_change
        theta_pnl = greeks.theta * scenario.time_change
        vega_pnl = greeks.vega * scenario.volatility_change * 100  # Vega per 1% vol change
        rho_pnl = greeks.rho * scenario.rate_change * 100  # Rho per 1% rate change
        
        # Second-order terms
        gamma_pnl = 0.5 * greeks.gamma * (scenario.spot_change ** 2) if include_second_order else 0.0
        
        # Additional second-order cross terms (simplified)
        second_order_pnl = gamma_pnl
        
        # Total Taylor expansion PnL
        taylor_total = delta_pnl + gamma_pnl + theta_pnl + vega_pnl + rho_pnl
        
        return PnLResult(
            scenario=scenario,
            delta_pnl=delta_pnl,
            gamma_pnl=gamma_pnl,
            theta_pnl=theta_pnl,
            vega_pnl=vega_pnl,
            rho_pnl=rho_pnl,
            second_order_pnl=second_order_pnl,
            taylor_total_pnl=taylor_total
        )
    
    def _calculate_actual_pnl(self, positions: List[Dict], scenario: MarketScenario) -> float:
        """Calculate actual PnL using full Black-Scholes revaluation."""
        original_value = 0.0
        new_value = 0.0
        
        for position in positions:
            try:
                quantity = position['quantity']
                
                # Original parameters
                if 'option_params' in position:
                    params_dict = position['option_params']
                else:
                    params_dict = position
                
                original_params = OptionParameters(
                    spot_price=params_dict['spot_price'],
                    strike_price=params_dict['strike_price'],
                    time_to_maturity=params_dict['time_to_maturity'],
                    volatility=params_dict['volatility'],
                    risk_free_rate=params_dict.get('risk_free_rate', 0.05),
                    option_type=params_dict['option_type'],
                    dividend_yield=params_dict.get('dividend_yield', 0.0)
                )
                
                # New parameters after scenario changes
                new_params = OptionParameters(
                    spot_price=params_dict['spot_price'] + scenario.spot_change,
                    strike_price=params_dict['strike_price'],
                    time_to_maturity=max(params_dict['time_to_maturity'] - scenario.time_change / 365.25, 1/365.25),
                    volatility=params_dict['volatility'] + scenario.volatility_change,
                    risk_free_rate=params_dict.get('risk_free_rate', 0.05) + scenario.rate_change,
                    option_type=params_dict['option_type'],
                    dividend_yield=params_dict.get('dividend_yield', 0.0)
                )
                
                # Calculate original and new values
                original_result = self.bs_model.calculate_option_price(original_params)
                new_result = self.bs_model.calculate_option_price(new_params)
                
                position_pnl = quantity * (new_result.option_price - original_result.option_price)
                
                original_value += quantity * original_result.option_price
                new_value += quantity * new_result.option_price
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate actual PnL for position: {e}")
                continue
        
        return new_value - original_value
    
    def _identify_largest_component(self, result: PnLResult) -> str:
        """Identify the largest P&L component."""
        components = {
            'delta': abs(result.delta_pnl),
            'gamma': abs(result.gamma_pnl),
            'theta': abs(result.theta_pnl),
            'vega': abs(result.vega_pnl),
            'rho': abs(result.rho_pnl)
        }
        
        return max(components, key=components.get)
    
    def _calculate_risk_contribution(self, result: PnLResult) -> Dict[str, float]:
        """Calculate risk contribution by Greek."""
        total_abs_pnl = (abs(result.delta_pnl) + abs(result.gamma_pnl) + 
                        abs(result.theta_pnl) + abs(result.vega_pnl) + abs(result.rho_pnl))
        
        if total_abs_pnl == 0:
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
        
        return {
            "delta": abs(result.delta_pnl) / total_abs_pnl,
            "gamma": abs(result.gamma_pnl) / total_abs_pnl,
            "theta": abs(result.theta_pnl) / total_abs_pnl,
            "vega": abs(result.vega_pnl) / total_abs_pnl,
            "rho": abs(result.rho_pnl) / total_abs_pnl
        }
    
    def _create_summary_statistics(self, results: List[PnLResult]) -> Dict[str, float]:
        """Create summary statistics from PnL results."""
        if not results:
            return {}
        
        pnls = [r.taylor_total_pnl for r in results]
        
        return {
            "max_pnl": max(pnls),
            "min_pnl": min(pnls),
            "mean_pnl": np.mean(pnls),
            "std_pnl": np.std(pnls),
            "var_95": np.percentile(pnls, 5),  # 95% VaR (5th percentile)
            "var_99": np.percentile(pnls, 1),  # 99% VaR (1st percentile)
            "scenarios_positive": sum(1 for p in pnls if p > 0),
            "scenarios_negative": sum(1 for p in pnls if p < 0)
        }
    
    def _analyze_risk_scenarios(self, results: List[PnLResult]) -> Dict[str, any]:
        """Analyze risk across different scenario types."""
        # Group by scenario characteristics
        extreme_moves = [r for r in results if abs(r.scenario.spot_change) > 5000]
        vol_shocks = [r for r in results if abs(r.scenario.volatility_change) > 0.2]
        time_decay = [r for r in results if r.scenario.time_change > 7]
        
        analysis = {}
        
        if extreme_moves:
            extreme_pnls = [r.taylor_total_pnl for r in extreme_moves]
            analysis["extreme_spot_moves"] = {
                "count": len(extreme_moves),
                "max_loss": min(extreme_pnls),
                "max_gain": max(extreme_pnls),
                "average": np.mean(extreme_pnls)
            }
        
        if vol_shocks:
            vol_pnls = [r.taylor_total_pnl for r in vol_shocks]
            analysis["volatility_shocks"] = {
                "count": len(vol_shocks),
                "max_loss": min(vol_pnls),
                "max_gain": max(vol_pnls),
                "average": np.mean(vol_pnls)
            }
        
        if time_decay:
            time_pnls = [r.taylor_total_pnl for r in time_decay]
            analysis["time_decay_scenarios"] = {
                "count": len(time_decay),
                "max_loss": min(time_pnls),
                "max_gain": max(time_pnls),
                "average": np.mean(time_pnls)
            }
        
        return analysis


# Convenience functions
def simulate_option_pnl(spot: float, strike: float, time_to_maturity: float,
                       volatility: float, option_type: str, quantity: float = 1.0,
                       scenarios: Optional[List[MarketScenario]] = None) -> List[PnLResult]:
    """
    Quick PnL simulation for a single option.
    
    Args:
        spot, strike, time_to_maturity, volatility, option_type: Option parameters
        quantity: Number of contracts
        scenarios: Custom scenarios (will create default if None)
        
    Returns:
        List of PnL results
    """
    simulator = TaylorPnLSimulator()
    
    position = [{
        'quantity': quantity,
        'spot_price': spot,
        'strike_price': strike,
        'time_to_maturity': time_to_maturity,
        'volatility': volatility,
        'option_type': option_type,
        'risk_free_rate': 0.05
    }]
    
    if scenarios is None:
        scenarios = simulator.create_stress_scenarios(spot, 0.1, 0.2, [1, 7])
    
    return simulator.simulate_pnl(position, scenarios)


if __name__ == "__main__":
    # Test the Taylor PnL simulator
    print("🧪 Testing Taylor Expansion PnL Simulator")
    print("=" * 45)
    
    simulator = TaylorPnLSimulator()
    
    # Sample position
    current_spot = 50000.0
    position = [{
        'quantity': 10,
        'spot_price': current_spot,
        'strike_price': 52000,
        'time_to_maturity': 30 / 365.25,
        'volatility': 0.8,
        'option_type': 'call',
        'risk_free_rate': 0.05
    }]
    
    # Create test scenarios
    test_scenarios = [
        MarketScenario(1000, 1, 0.1, 0, "Small_Up_Move"),
        MarketScenario(-2000, 1, -0.05, 0, "Medium_Down_Move"),
        MarketScenario(5000, 7, 0.2, 0, "Large_Up_Vol_Shock"),
        MarketScenario(-3000, 7, -0.1, 0, "Down_Vol_Crush")
    ]
    
    print("📊 Running PnL simulation...")
    results = simulator.simulate_pnl(position, test_scenarios, 
                                   include_second_order=True, 
                                   validate_with_bs=True)
    
    print(f"\n💰 PnL Results ({len(results)} scenarios):")
    for result in results:
        print(f"\n  Scenario: {result.scenario.scenario_name}")
        print(f"    Spot change: ${result.scenario.spot_change:,.0f}")
        print(f"    Vol change: {result.scenario.volatility_change:.1%}")
        print(f"    Time decay: {result.scenario.time_change} days")
        print(f"    Taylor PnL: ${result.taylor_total_pnl:.2f}")
        if result.actual_pnl is not None:
            print(f"    Actual PnL: ${result.actual_pnl:.2f}")
            print(f"    Error: ${result.taylor_error:.2f} ({abs(result.taylor_error)/abs(result.actual_pnl)*100:.1f}%)")
        print(f"    Largest component: {result.largest_component}")
        print(f"    Delta PnL: ${result.delta_pnl:.2f}")
        print(f"    Gamma PnL: ${result.gamma_pnl:.2f}")
        print(f"    Theta PnL: ${result.theta_pnl:.2f}")
        print(f"    Vega PnL: ${result.vega_pnl:.2f}")
    
    # Test accuracy analysis
    print(f"\n🎯 Taylor Expansion Accuracy Analysis:")
    accuracy = simulator.analyze_taylor_accuracy(results)
    if "error_statistics" in accuracy:
        stats = accuracy["error_statistics"]
        print(f"  Mean Absolute Error: ${stats['mean_absolute_error']:.2f}")
        print(f"  Mean Relative Error: {stats['mean_relative_error']:.1%}")
        print(f"  Max Absolute Error: ${stats['max_absolute_error']:.2f}")
    
    print(f"\n✅ Taylor expansion PnL simulator test completed!")
    print(f"📈 Core formula implemented: ΔC ≈ δΔS + ½γ(ΔS)² + θΔt + νΔσ + ρΔr")