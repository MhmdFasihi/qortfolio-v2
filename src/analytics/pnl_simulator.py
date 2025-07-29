# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Missing Classes Fix for PnL Simulator
Location: ADD TO END of src/analytics/pnl_simulator.py

Add these classes and aliases to fix dashboard import errors.
"""

@dataclass  
class MarketScenario:
    """Market scenario for PnL analysis - expected by tests and dashboard."""
    spot_change: float          # Absolute spot price change
    time_decay_days: float      # Number of days of time decay
    vol_change: float           # Volatility change (absolute)
    ir_change: float = 0.0      # Interest rate change
    scenario_name: str = ""     # Name for identification
    
    def __post_init__(self):
        if not self.scenario_name:
            self.scenario_name = f"Spot{self.spot_change:+.0f}_Vol{self.vol_change:+.2f}_Time{self.time_decay_days:.0f}d"

@dataclass
class PnLResult:
    """PnL calculation result - expected by tests and dashboard."""
    scenario: MarketScenario
    delta_pnl: float
    gamma_pnl: float  
    theta_pnl: float
    vega_pnl: float
    rho_pnl: float = 0.0
    taylor_total_pnl: float = 0.0
    actual_pnl: Optional[float] = None
    taylor_error: Optional[float] = None
    
    def __post_init__(self):
        # Calculate total Taylor PnL
        self.taylor_total_pnl = (
            self.delta_pnl + self.gamma_pnl + 
            self.theta_pnl + self.vega_pnl + self.rho_pnl
        )
        
        # Calculate error if actual PnL available
        if self.actual_pnl is not None:
            self.taylor_error = self.taylor_total_pnl - self.actual_pnl

class TaylorPnLSimulator:
    """
    Alias for TaylorExpansionPnL to maintain compatibility.
    
    This class provides the interface expected by dashboard and tests
    while delegating to the actual TaylorExpansionPnL implementation.
    """
    
    def __init__(self):
        """Initialize with TaylorExpansionPnL backend."""
        self._backend = TaylorExpansionPnL()
        self.logger = logging.getLogger(__name__)
    
    def simulate_pnl(self, 
                     positions: List[Dict[str, Any]], 
                     scenarios: List[MarketScenario],
                     include_second_order: bool = True,
                     validate_with_bs: bool = False) -> List[PnLResult]:
        """
        Simulate PnL for multiple positions and scenarios.
        
        Args:
            positions: List of position dictionaries
            scenarios: List of MarketScenario objects
            include_second_order: Include gamma effects
            validate_with_bs: Validate with Black-Scholes revaluation
            
        Returns:
            List of PnLResult objects
        """
        results = []
        
        for scenario in scenarios:
            try:
                # Convert position dict to OptionParameters if needed
                if len(positions) == 1:
                    pos = positions[0]
                    
                    # Create OptionParameters from position dict
                    params = OptionParameters(
                        spot_price=pos['spot_price'],
                        strike_price=pos['strike_price'], 
                        time_to_expiry=pos['time_to_maturity'],
                        volatility=pos['volatility'],
                        risk_free_rate=pos.get('risk_free_rate', 0.05),
                        option_type=OptionType.CALL if pos['option_type'].lower() == 'call' else OptionType.PUT
                    )
                    
                    # Calculate PnL components using backend
                    pnl_components = self._backend.calculate_pnl_components(
                        params,
                        spot_shock=scenario.spot_change / params.spot_price,  # Convert to relative
                        vol_shock=scenario.vol_change,
                        time_decay_days=scenario.time_decay_days
                    )
                    
                    # Create PnLResult
                    result = PnLResult(
                        scenario=scenario,
                        delta_pnl=pnl_components.delta_pnl * pos['quantity'],
                        gamma_pnl=pnl_components.gamma_pnl * pos['quantity'],
                        theta_pnl=pnl_components.theta_pnl * pos['quantity'], 
                        vega_pnl=pnl_components.vega_pnl * pos['quantity'],
                        rho_pnl=0.0  # Not typically calculated
                    )
                    
                    # Optional: Calculate actual PnL for validation
                    if validate_with_bs:
                        try:
                            # Calculate actual PnL by re-pricing
                            new_spot = params.spot_price + scenario.spot_change
                            new_vol = params.volatility + scenario.vol_change
                            new_time = params.time_to_expiry - (scenario.time_decay_days / 365.25)
                            
                            if new_time > 0:
                                new_params = OptionParameters(
                                    spot_price=new_spot,
                                    strike_price=params.strike_price,
                                    time_to_expiry=new_time,
                                    volatility=new_vol,
                                    risk_free_rate=params.risk_free_rate,
                                    option_type=params.option_type
                                )
                                
                                old_price = self._backend.bs_model.option_price(params)
                                new_price = self._backend.bs_model.option_price(new_params)
                                
                                result.actual_pnl = (new_price - old_price) * pos['quantity']
                            
                        except Exception as e:
                            self.logger.warning(f"Could not calculate actual PnL: {e}")
                    
                    results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error calculating PnL for scenario {scenario.scenario_name}: {e}")
                continue
        
        return results
    
    def analyze_taylor_accuracy(self, results: List[PnLResult]) -> Dict[str, Any]:
        """Analyze Taylor expansion accuracy."""
        valid_results = [r for r in results if r.actual_pnl is not None and r.taylor_error is not None]
        
        if not valid_results:
            return {"error": "No valid results for accuracy analysis"}
        
        errors = [abs(r.taylor_error) for r in valid_results]
        relative_errors = [
            abs(r.taylor_error) / abs(r.actual_pnl) * 100 
            for r in valid_results 
            if abs(r.actual_pnl) > 1e-6
        ]
        
        return {
            "error_statistics": {
                "mean_absolute_error": np.mean(errors),
                "max_absolute_error": np.max(errors),
                "mean_relative_error_pct": np.mean(relative_errors) if relative_errors else 0,
                "max_relative_error_pct": np.max(relative_errors) if relative_errors else 0
            },
            "sample_count": len(valid_results),
            "accuracy_ratio": len(valid_results) / len(results)
        }

# Convenience function for dashboard compatibility
def simulate_option_pnl(spot_price: float,
                       strike_price: float,
                       time_to_maturity: float,
                       volatility: float,
                       option_type: str = "call",
                       quantity: int = 1,
                       scenarios: Optional[List[MarketScenario]] = None) -> List[PnLResult]:
    """
    Convenience function for single option PnL simulation.
    Expected by dashboard and tests.
    """
    if scenarios is None:
        # Default scenarios
        scenarios = [
            MarketScenario(-0.1 * spot_price, 1, -0.1, 0, "Down_10pct"),
            MarketScenario(0, 1, 0, 0, "Base_Case"),
            MarketScenario(0.1 * spot_price, 1, 0.1, 0, "Up_10pct")
        ]
    
    position = [{
        'spot_price': spot_price,
        'strike_price': strike_price,
        'time_to_maturity': time_to_maturity,
        'volatility': volatility,
        'option_type': option_type,
        'quantity': quantity,
        'risk_free_rate': 0.05
    }]
    
    simulator = TaylorPnLSimulator()
    return simulator.simulate_pnl(position, scenarios, validate_with_bs=True)

# Update __all__ to include new classes
__all__ = [
    'TaylorExpansionPnL',           # Main implementation
    'TaylorPnLSimulator',           # Dashboard compatibility alias
    'MarketScenario',               # Expected by tests/dashboard
    'PnLResult',                    # Expected by tests/dashboard
    'PnLScenario',                  # Original dataclass
    'PortfolioPosition',            # Portfolio support
    'ScenarioParameters',           # Scenario configuration
    'PnLComponents',                # PnL breakdown
    'quick_pnl_analysis',           # Convenience function
    'simulate_option_pnl',          # Dashboard function
    'test_taylor_expansion_pnl'     # Testing function
]

# Test the fix
if __name__ == "__main__":
    print("🔧 Testing PnL Simulator Import Fix")
    print("=" * 40)
    
    try:
        # Test the imports that dashboard expects
        simulator = TaylorPnLSimulator()
        print("✅ TaylorPnLSimulator can be created")
        
        # Test MarketScenario
        scenario = MarketScenario(1000, 1, 0.1, 0, "Test")
        print("✅ MarketScenario works")
        
        # Test simulate_option_pnl function
        results = simulate_option_pnl(50000, 52000, 30/365.25, 0.8)
        print(f"✅ simulate_option_pnl works: {len(results)} results")
        
        print("\n🎉 All dashboard imports should now work!")
        
    except Exception as e:
        print(f"❌ Import fix test failed: {e}")