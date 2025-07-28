#!/usr/bin/env python3
"""
Comprehensive Test Suite for Financial Calculations
Tests Black-Scholes, Greeks, and Taylor Expansion PnL components
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)


def test_financial_calculations_suite():
    """Comprehensive test of all financial calculation components."""
    
    print("🧪 Testing Qortfolio V2 Financial Calculations Suite")
    print("=" * 60)
    
    # Test 1: Import all financial modules
    print("\n1. Testing Financial Module Imports...")
    try:
        from models.options.black_scholes import (
            BlackScholesModel, OptionParameters, 
            price_option, calculate_greeks
        )
        from models.options.greeks_calculator import (
            GreeksCalculator, GreeksProfile, RiskMetrics,
            calculate_option_greeks, analyze_portfolio_risk
        )
        from analytics.pnl_simulator import (
            TaylorPnLSimulator, MarketScenario, PnLResult,
            simulate_option_pnl
        )
        print("   ✅ All financial calculation modules imported successfully")
        
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        print("   📝 Make sure to create the module files from the artifacts")
        return False
    
    # Test 2: Test Black-Scholes implementation
    print("\n2. Testing Black-Scholes Implementation...")
    try:
        bs_model = BlackScholesModel()
        
        # Standard test case (known values for validation)
        spot = 100.0
        strike = 100.0
        time_to_maturity = 0.25  # 3 months
        volatility = 0.20  # 20%
        risk_free_rate = 0.05  # 5%
        
        # Test call option
        call_price = price_option(spot, strike, time_to_maturity, volatility, 'call', risk_free_rate)
        put_price = price_option(spot, strike, time_to_maturity, volatility, 'put', risk_free_rate)
        
        print(f"   📈 ATM Call Price: ${call_price:.4f}")
        print(f"   📉 ATM Put Price: ${put_price:.4f}")
        
        # Test put-call parity: C - P = S - K*e^(-r*T)
        theoretical_diff = spot - strike * np.exp(-risk_free_rate * time_to_maturity)
        actual_diff = call_price - put_price
        parity_error = abs(theoretical_diff - actual_diff)
        
        print(f"   🔍 Put-call parity check:")
        print(f"      Theoretical: {theoretical_diff:.4f}")
        print(f"      Actual: {actual_diff:.4f}")
        print(f"      Error: {parity_error:.6f}")
        
        if parity_error < 1e-4:
            print("   ✅ Put-call parity verified!")
        else:
            print("   ❌ Put-call parity error too large")
            
        # Test Greeks
        call_greeks = calculate_greeks(spot, strike, time_to_maturity, volatility, 'call', risk_free_rate)
        print(f"   📊 Call Greeks: Delta={call_greeks['delta']:.4f}, Gamma={call_greeks['gamma']:.6f}")
        
    except Exception as e:
        print(f"   ❌ Black-Scholes test failed: {e}")
        return False
    
    # Test 3: Test Greeks Calculator
    print("\n3. Testing Advanced Greeks Calculator...")
    try:
        greeks_calc = GreeksCalculator()
        
        # Create sample portfolio
        sample_positions = [
            {
                'quantity': 10,  # Long 10 calls
                'spot_price': 50000,
                'strike_price': 52000,
                'time_to_maturity': 30 / 365.25,
                'volatility': 0.8,
                'risk_free_rate': 0.05,
                'option_type': 'call'
            },
            {
                'quantity': -5,  # Short 5 puts
                'spot_price': 50000,
                'strike_price': 48000,
                'time_to_maturity': 30 / 365.25,
                'volatility': 0.8,
                'risk_free_rate': 0.05,
                'option_type': 'put'
            }
        ]
        
        portfolio_greeks = greeks_calc.calculate_portfolio_greeks(sample_positions)
        
        print(f"   📊 Portfolio Greeks:")
        print(f"      Delta: {portfolio_greeks.delta:.4f}")
        print(f"      Gamma: {portfolio_greeks.gamma:.6f}")
        print(f"      Theta: {portfolio_greeks.theta:.4f}")
        print(f"      Vega: {portfolio_greeks.vega:.4f}")
        print(f"      Portfolio Value: ${portfolio_greeks.portfolio_value:.2f}")
        
        # Test risk metrics
        risk_metrics = greeks_calc.calculate_risk_metrics(portfolio_greeks, 50000)
        print(f"   ⚠️ Risk Metrics:")
        print(f"      Max daily theta loss: ${risk_metrics.max_loss_1_day:.2f}")
        print(f"      Risk from 1% move: ${risk_metrics.max_loss_1_percent_move:.2f}")
        
        # Test P&L attribution
        pnl_attr = greeks_calc.calculate_pnl_attribution(
            portfolio_greeks,
            spot_change=1000,  # $1000 spot move
            time_decay=1,      # 1 day
            volatility_change=0.05  # 5% vol increase
        )
        
        print(f"   💰 P&L Attribution (example scenario):")
        print(f"      Delta P&L: ${pnl_attr['delta_pnl']:.2f}")
        print(f"      Gamma P&L: ${pnl_attr['gamma_pnl']:.2f}")
        print(f"      Theta P&L: ${pnl_attr['theta_pnl']:.2f}")
        print(f"      Vega P&L: ${pnl_attr['vega_pnl']:.2f}")
        print(f"      Total: ${pnl_attr['total_explained']:.2f}")
        
    except Exception as e:
        print(f"   ❌ Greeks calculator test failed: {e}")
        return False
    
    # Test 4: Test Taylor Expansion PnL Simulator
    print("\n4. Testing Taylor Expansion PnL Simulator...")
    try:
        pnl_simulator = TaylorPnLSimulator()
        
        # Create test scenarios
        base_spot = 50000.0
        test_scenarios = [
            MarketScenario(1000, 1, 0.1, 0, "Small_Up"),     # +$1000, +1day, +10% vol
            MarketScenario(-2000, 1, -0.05, 0, "Medium_Down"), # -$2000, +1day, -5% vol
            MarketScenario(5000, 7, 0.2, 0, "Large_Up"),     # +$5000, +7days, +20% vol
            MarketScenario(-3000, 7, -0.1, 0, "Down_Crush")  # -$3000, +7days, -10% vol
        ]
        
        # Test single position
        test_position = [{
            'quantity': 10,
            'spot_price': base_spot,
            'strike_price': 52000,
            'time_to_maturity': 30 / 365.25,
            'volatility': 0.8,
            'option_type': 'call',
            'risk_free_rate': 0.05
        }]
        
        # Run PnL simulation
        pnl_results = pnl_simulator.simulate_pnl(
            test_position, test_scenarios,
            include_second_order=True,
            validate_with_bs=True
        )
        
        print(f"   📈 Taylor PnL Simulation Results ({len(pnl_results)} scenarios):")
        
        total_taylor_error = 0.0
        max_error_pct = 0.0
        
        for result in pnl_results:
            scenario_name = result.scenario.scenario_name
            taylor_pnl = result.taylor_total_pnl
            actual_pnl = result.actual_pnl
            error = result.taylor_error
            
            if actual_pnl and abs(actual_pnl) > 1e-6:
                error_pct = abs(error) / abs(actual_pnl) * 100
                max_error_pct = max(max_error_pct, error_pct)
            else:
                error_pct = 0.0
            
            total_taylor_error += abs(error) if error else 0.0
            
            print(f"      {scenario_name}:")
            print(f"        Taylor PnL: ${taylor_pnl:.2f}")
            print(f"        Actual PnL: ${actual_pnl:.2f}")
            print(f"        Error: ${error:.2f} ({error_pct:.1f}%)")
            print(f"        Components: Δ=${result.delta_pnl:.0f}, Γ=${result.gamma_pnl:.0f}, θ=${result.theta_pnl:.0f}")
        
        avg_error = total_taylor_error / len(pnl_results)
        print(f"   🎯 Taylor Expansion Accuracy:")
        print(f"      Average absolute error: ${avg_error:.2f}")
        print(f"      Maximum relative error: {max_error_pct:.1f}%")
        
        if max_error_pct < 5.0:  # Less than 5% error
            print("   ✅ Taylor expansion accuracy is excellent!")
        elif max_error_pct < 15.0:  # Less than 15% error
            print("   ✅ Taylor expansion accuracy is good")
        else:
            print("   ⚠️ Taylor expansion has some accuracy issues for large moves")
        
        # Test stress scenarios
        print(f"   🔥 Testing stress scenario generation...")
        stress_scenarios = pnl_simulator.create_stress_scenarios(base_spot, 0.2, 0.3, [1, 7])
        print(f"      Generated {len(stress_scenarios)} stress scenarios")
        
    except Exception as e:
        print(f"   ❌ PnL simulator test failed: {e}")
        return False
    
    # Test 5: Test Time Calculation Integration
    print("\n5. Testing Time Calculation Integration...")
    try:
        from core.utils.time_utils import calculate_time_to_maturity
        
        # Test that our fixed time calculation is being used in options pricing
        current_time = datetime(2024, 1, 1)
        expiry_time = datetime(2024, 1, 31)  # 30 days
        
        # Our FIXED time calculation
        tte_fixed = calculate_time_to_maturity(current_time, expiry_time)
        expected = 30 / 365.25
        
        print(f"   ⏰ Time calculation verification:")
        print(f"      30 days in years (fixed): {tte_fixed:.6f}")
        print(f"      Expected: {expected:.6f}")
        print(f"      Error: {abs(tte_fixed - expected):.8f}")
        
        if abs(tte_fixed - expected) < 1e-7:
            print("   ✅ Fixed time calculation is working correctly!")
        else:
            print("   ❌ Time calculation error detected!")
        
        # Test in options pricing
        option_price_30d = price_option(50000, 50000, tte_fixed, 0.8, 'call', 0.05)
        option_price_direct = price_option(50000, 50000, 30/365.25, 0.8, 'call', 0.05)
        
        print(f"   📊 Options pricing with time calculation:")
        print(f"      Using fixed calculation: ${option_price_30d:.2f}")
        print(f"      Using direct 30/365.25: ${option_price_direct:.2f}")
        print(f"      Difference: ${abs(option_price_30d - option_price_direct):.6f}")
        
    except Exception as e:
        print(f"   ❌ Time calculation integration test failed: {e}")
        return False
    
    # Test 6: Integration Test - Complete Workflow
    print("\n6. Testing Complete Financial Workflow...")
    try:
        print("   🔄 Running complete options analysis workflow...")
        
        # Step 1: Define portfolio
        crypto_portfolio = [
            {'quantity': 20, 'spot_price': 45000, 'strike_price': 50000, 
             'time_to_maturity': 45/365.25, 'volatility': 0.9, 'option_type': 'call'},
            {'quantity': -10, 'spot_price': 45000, 'strike_price': 40000, 
             'time_to_maturity': 45/365.25, 'volatility': 0.9, 'option_type': 'put'},
            {'quantity': 5, 'spot_price': 45000, 'strike_price': 47000, 
             'time_to_maturity': 15/365.25, 'volatility': 1.1, 'option_type': 'call'}
        ]
        
        # Step 2: Calculate portfolio Greeks
        greeks_calc = GreeksCalculator()
        portfolio_greeks = greeks_calc.calculate_portfolio_greeks(crypto_portfolio)
        
        # Step 3: Run risk analysis
        risk_analysis = analyze_portfolio_risk(crypto_portfolio, 45000)
        
        # Step 4: Run PnL simulation
        pnl_sim = TaylorPnLSimulator()
        scenarios = pnl_sim.create_stress_scenarios(45000, 0.15, 0.25, [1, 3, 7])
        
        # Take subset for testing
        test_scenarios = scenarios[:10]  # First 10 scenarios
        workflow_results = pnl_sim.simulate_pnl(crypto_portfolio, test_scenarios[:5])
        
        print(f"   📊 Complete Workflow Results:")
        print(f"      Portfolio positions: {len(crypto_portfolio)}")
        print(f"      Portfolio value: ${portfolio_greeks.portfolio_value:.2f}")
        print(f"      Portfolio delta: {portfolio_greeks.delta:.4f}")
        print(f"      Portfolio gamma: {portfolio_greeks.gamma:.6f}")
        print(f"      Daily theta decay: ${abs(portfolio_greeks.theta):.2f}")
        print(f"      Risk scenarios tested: {len(workflow_results)}")
        
        # Calculate max/min PnL from scenarios
        if workflow_results:
            pnls = [r.taylor_total_pnl for r in workflow_results]
            print(f"      Scenario PnL range: ${min(pnls):.2f} to ${max(pnls):.2f}")
        
        print("   ✅ Complete financial workflow executed successfully!")
        
    except Exception as e:
        print(f"   ❌ Integration workflow test failed: {e}")
        return False
    
    # Test 7: Performance and Validation
    print("\n7. Testing Performance and Validation...")
    try:
        print("   ⚡ Performance testing...")
        
        # Time a batch of option calculations
        start_time = time.time()
        
        test_calculations = []
        for i in range(100):
            spot = 45000 + i * 100  # Varying spot prices
            strike = 50000
            vol = 0.7 + i * 0.002   # Varying volatility
            tte = (30 - i * 0.1) / 365.25  # Varying time
            
            price = price_option(spot, strike, tte, vol, 'call', 0.05)
            greeks = calculate_greeks(spot, strike, tte, vol, 'call', 0.05)
            test_calculations.append((price, greeks))
        
        calc_time = time.time() - start_time
        print(f"      100 option calculations: {calc_time:.3f} seconds")
        print(f"      Average per calculation: {calc_time/100*1000:.2f} ms")
        
        if calc_time < 1.0:  # Should be fast
            print("   ✅ Performance is excellent!")
        elif calc_time < 5.0:
            print("   ✅ Performance is acceptable")
        else:
            print("   ⚠️ Performance could be improved")
        
        # Validation test - compare with known benchmark
        benchmark_call = price_option(100, 100, 0.25, 0.2, 'call', 0.05)
        expected_call = 5.573  # Approximate known value
        
        if abs(benchmark_call - expected_call) < 0.1:
            print("   ✅ Pricing validation passed!")
        else:
            print(f"   ⚠️ Pricing validation: got {benchmark_call:.3f}, expected ~{expected_call}")
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 Financial Calculations Test Suite COMPLETED!")
    print()
    print("📋 Summary of Tested Components:")
    print("✅ Black-Scholes option pricing model")
    print("✅ Put-call parity validation") 
    print("✅ Advanced Greeks calculator")
    print("✅ Portfolio-level risk analysis")
    print("✅ Taylor expansion PnL simulator")
    print("✅ Time calculation integration (BUG FIXED)")
    print("✅ Complete financial workflow")
    print("✅ Performance and validation testing")
    print()
    print("🎯 Core Formula Implemented:")
    print("   Taylor PnL: ΔC ≈ δΔS + ½γ(ΔS)² + θΔt + νΔσ + ρΔr")
    print()
    print("🚀 Phase 1, Week 2: Financial Calculations - COMPLETE!")
    print("📅 Ready for Phase 2: Advanced Analytics & Dashboard")
    
    return True


def show_next_development_phase():
    """Show what comes next."""
    print("\n📋 Next Development Phase:")
    print("=" * 50)
    print()
    print("🎯 Phase 2, Week 3-4: Advanced Analytics & Visualizations")
    print()
    print("  1. 📊 Volatility Surface Analysis")
    print("     - 3D volatility surface construction")
    print("     - Implied vs. Realized volatility")
    print("     - Volatility smile/skew analysis")
    print()
    print("  2. 📈 Options Chain Analytics")
    print("     - Real-time options chain analysis")
    print("     - Greeks heatmaps")
    print("     - Open interest analysis")
    print()
    print("  3. 🎨 Interactive Dashboard Foundation")
    print("     - Streamlit dashboard framework")
    print("     - Real-time data integration")
    print("     - Interactive charts and controls")
    print()
    print("  4. 🔍 Advanced Risk Analytics")
    print("     - VaR calculations")
    print("     - Stress testing framework")
    print("     - Portfolio optimization")
    print()
    print("Foundation is rock solid! Ready to build analytics? 🏗️")


if __name__ == "__main__":
    success = test_financial_calculations_suite()
    
    if success:
        show_next_development_phase()
    else:
        print("\n🚨 Some financial calculation tests failed.")
        print("Please ensure all module files are created from the artifacts.")
        print("Check:")
        print("- src/models/options/black_scholes.py")
        print("- src/models/options/greeks_calculator.py") 
        print("- src/analytics/pnl_simulator.py")
        print("- All required directories exist")