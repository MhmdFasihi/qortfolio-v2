#!/usr/bin/env python3
"""
Test Greeks Calculator Module
File: tests/test_greeks_calculator.py
Run: python tests/test_greeks_calculator.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def test_greeks_calculator():
    """Test Greeks Calculator with portfolio support."""
    
    print("=" * 60)
    print("Testing Greeks Calculator Module")
    print("=" * 60)
    
    try:
        # Import the modules
        from src.models.options.greeks_calculator import (
            GreeksCalculator, GreeksProfile, PortfolioGreeks, RiskMetrics,
            calculate_option_greeks, analyze_portfolio_risk
        )
        from src.models.options.black_scholes import BlackScholesModel, OptionParameters, OptionType
        
        print("✅ Greeks Calculator module imported successfully")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Make sure:")
        print("  - src/models/options/greeks_calculator.py exists")
        print("  - src/models/options/black_scholes.py exists")
        return False
    
    # Test 1: Initialize calculator
    print("\n1. Testing calculator initialization...")
    try:
        bs_model = BlackScholesModel()
        greeks_calc = GreeksCalculator(bs_model)
        print("   ✅ Calculator initialized with Black-Scholes model")
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        return False
    
    # Test 2: Calculate single option Greeks
    print("\n2. Testing single option Greeks calculation...")
    try:
        params = OptionParameters(
            spot_price=50000,
            strike_price=52000,
            time_to_maturity=30/365.25,
            volatility=0.8,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
            is_coin_based=True  # Deribit style
        )
        
        greeks_profile = greeks_calc.calculate_option_greeks(params)
        
        print(f"   Option: BTC Call, Strike=$52,000, 30 days")
        print(f"   ✅ Delta: {greeks_profile.delta:.6f}")
        print(f"   ✅ Gamma: {greeks_profile.gamma:.9f}")
        print(f"   ✅ Theta: {greeks_profile.theta:.6f} per day")
        print(f"   ✅ Vega: {greeks_profile.vega:.6f} per 1% vol")
        print(f"   ✅ Rho: {greeks_profile.rho:.6f} per 1% rate")
        
        # Check if second-order Greeks are calculated
        if greeks_profile.speed is not None:
            print(f"   ✅ Speed (Gamma derivative): {greeks_profile.speed:.9f}")
        if greeks_profile.charm is not None:
            print(f"   ✅ Charm (Delta decay): {greeks_profile.charm:.9f}")
        if greeks_profile.vanna is not None:
            print(f"   ✅ Vanna (Delta/Vega cross): {greeks_profile.vanna:.9f}")
        if greeks_profile.vomma is not None:
            print(f"   ✅ Vomma (Vega derivative): {greeks_profile.vomma:.9f}")
            
    except Exception as e:
        print(f"   ❌ Single option Greeks failed: {e}")
        return False
    
    # Test 3: Portfolio Greeks calculation
    print("\n3. Testing portfolio Greeks calculation...")
    try:
        # Create a sample portfolio
        portfolio_positions = [
            {
                'quantity': 10,  # Long 10 calls
                'spot_price': 50000,
                'strike_price': 52000,
                'time_to_maturity': 30/365.25,
                'volatility': 0.8,
                'option_type': 'call',
                'underlying': 'BTC',
                'is_coin_based': True
            },
            {
                'quantity': -5,  # Short 5 puts
                'spot_price': 50000,
                'strike_price': 48000,
                'time_to_maturity': 15/365.25,
                'volatility': 0.75,
                'option_type': 'put',
                'underlying': 'BTC',
                'is_coin_based': True
            },
            {
                'quantity': 20,  # Long 20 ETH calls
                'spot_price': 3500,
                'strike_price': 3600,
                'time_to_maturity': 45/365.25,
                'volatility': 0.85,
                'option_type': 'call',
                'underlying': 'ETH',
                'is_coin_based': True
            }
        ]
        
        portfolio_greeks = greeks_calc.calculate_portfolio_greeks(portfolio_positions)
        
        print(f"   Portfolio: {portfolio_greeks.positions_count} positions")
        print(f"   ✅ Portfolio Value: ${portfolio_greeks.portfolio_value:,.2f}")
        print(f"   ✅ Total Delta: {portfolio_greeks.total_delta:.4f}")
        print(f"   ✅ Total Gamma: {portfolio_greeks.total_gamma:.6f}")
        print(f"   ✅ Total Theta: {portfolio_greeks.total_theta:.4f} per day")
        print(f"   ✅ Total Vega: {portfolio_greeks.total_vega:.4f} per 1% vol")
        print(f"   ✅ Delta Dollars: ${portfolio_greeks.delta_dollars:,.2f}")
        print(f"   ✅ Gamma Dollars: ${portfolio_greeks.gamma_dollars:,.2f} per 1% move")
        
        # Check by underlying aggregation
        print("\n   Greeks by Underlying:")
        for underlying, greeks in portfolio_greeks.by_underlying.items():
            print(f"     {underlying}: Δ={greeks.delta:.4f}, Γ={greeks.gamma:.6f}")
            
    except Exception as e:
        print(f"   ❌ Portfolio Greeks failed: {e}")
        return False
    
    # Test 4: Risk metrics calculation
    print("\n4. Testing risk metrics calculation...")
    try:
        risk_metrics = greeks_calc.calculate_risk_metrics(portfolio_positions)
        
        print(f"   ✅ Gamma Exposure: ${risk_metrics.gamma_exposure:,.2f}")
        if risk_metrics.gamma_flip_point:
            print(f"   ✅ Gamma Flip Point: ${risk_metrics.gamma_flip_point:,.2f}")
        print(f"   ✅ Max Gamma Strike: ${risk_metrics.max_gamma_strike:,.2f}")
        print(f"   ✅ Pin Risk: ${risk_metrics.pin_risk:,.2f}")
        print(f"   ✅ Vega Exposure: {risk_metrics.vega_exposure:.4f}")
        print(f"   ✅ Daily Theta Decay: {risk_metrics.theta_decay_daily:.4f}")
        print(f"   ✅ Delta Neutral Hedge: {risk_metrics.delta_neutral_hedge:.4f} shares")
        
    except Exception as e:
        print(f"   ❌ Risk metrics failed: {e}")
        return False
    
    # Test 5: Gamma exposure profile
    print("\n5. Testing gamma exposure profile...")
    try:
        gamma_profile = greeks_calc.calculate_gamma_exposure_profile(
            portfolio_positions,
            price_range=(0.9, 1.1),  # ±10% range
            steps=10
        )
        
        print(f"   Gamma Profile Shape: {gamma_profile.shape}")
        print(f"   Price Range: ${gamma_profile['price'].min():,.0f} - ${gamma_profile['price'].max():,.0f}")
        
        # Show sample points
        print("\n   Sample Gamma Exposure Points:")
        sample_points = gamma_profile.iloc[::3]  # Every 3rd point
        for _, row in sample_points.iterrows():
            print(f"     Price ${row['price']:,.0f}: Gamma Exposure = ${row['gamma_exposure']:,.2f}")
        
        print(f"   ✅ Gamma profile calculated for {len(gamma_profile)} price points")
        
    except Exception as e:
        print(f"   ❌ Gamma profile failed: {e}")
        return False
    
    # Test 6: Quick functions
    print("\n6. Testing quick calculation functions...")
    try:
        # Single option Greeks
        quick_greeks = calculate_option_greeks(
            spot=50000,
            strike=52000,
            time_to_maturity=30/365.25,
            volatility=0.8,
            option_type='call',
            is_coin_based=True
        )
        
        print(f"   ✅ Quick Greeks calculation successful")
        print(f"     Delta: {quick_greeks['delta']:.6f}")
        print(f"     Gamma: {quick_greeks['gamma']:.9f}")
        
        # Portfolio risk analysis
        risk_analysis = analyze_portfolio_risk(portfolio_positions)
        
        print(f"   ✅ Quick portfolio analysis successful")
        print(f"     Portfolio Value: ${risk_analysis['portfolio_summary']['portfolio_value']:,.2f}")
        print(f"     Gamma Exposure: ${risk_analysis['risk_metrics']['gamma_exposure']:,.2f}")
        
    except Exception as e:
        print(f"   ❌ Quick functions failed: {e}")
        return False
    
    # Test 7: Edge cases
    print("\n7. Testing edge cases...")
    try:
        # Empty portfolio
        empty_portfolio = []
        empty_greeks = greeks_calc.calculate_portfolio_greeks(empty_portfolio)
        print(f"   ✅ Empty portfolio handled: Delta={empty_greeks.total_delta}")
        
        # Near expiration
        near_expiry_position = [{
            'quantity': 1,
            'spot_price': 50000,
            'strike_price': 50000,
            'time_to_maturity': 1/365.25,  # 1 day
            'volatility': 0.8,
            'option_type': 'call',
            'underlying': 'BTC',
            'is_coin_based': True
        }]
        
        near_expiry_greeks = greeks_calc.calculate_portfolio_greeks(near_expiry_position)
        print(f"   ✅ Near expiration handled: Gamma={near_expiry_greeks.total_gamma:.6f}")
        
        # Very high volatility
        high_vol_position = [{
            'quantity': 1,
            'spot_price': 50000,
            'strike_price': 50000,
            'time_to_maturity': 30/365.25,
            'volatility': 2.0,  # 200% vol
            'option_type': 'call',
            'underlying': 'BTC',
            'is_coin_based': True
        }]
        
        high_vol_greeks = greeks_calc.calculate_portfolio_greeks(high_vol_position)
        print(f"   ✅ High volatility handled: Vega={high_vol_greeks.total_vega:.6f}")
        
    except Exception as e:
        print(f"   ❌ Edge case testing failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All Greeks Calculator tests passed!")
    print("=" * 60)
    return True


def test_deribit_integration():
    """Test integration with Deribit-style data."""
    print("\n" + "=" * 60)
    print("Testing Deribit Integration")
    print("=" * 60)
    
    try:
        from src.models.options.greeks_calculator import GreeksCalculator
        
        greeks_calc = GreeksCalculator()
        
        # Simulate Deribit position (prices in BTC)
        deribit_position = [{
            'quantity': 100,  # 100 contracts
            'spot_price': 50000,  # BTC at $50k
            'strike_price': 52000,
            'time_to_maturity': 30/365.25,
            'volatility': 0.8,
            'option_type': 'call',
            'underlying': 'BTC',
            'is_coin_based': True,  # Deribit uses coin-based pricing
            'mark_price_btc': 0.0523  # Example Deribit price in BTC
        }]
        
        portfolio_greeks = greeks_calc.calculate_portfolio_greeks(deribit_position)
        
        print(f"Deribit BTC Call Option:")
        print(f"  Contracts: 100")
        print(f"  Strike: $52,000")
        print(f"  Mark Price: 0.0523 BTC")
        print(f"  Portfolio Delta: {portfolio_greeks.total_delta:.4f}")
        print(f"  Portfolio Gamma: {portfolio_greeks.total_gamma:.6f}")
        print(f"  Value in USD: ${portfolio_greeks.portfolio_value:,.2f}")
        print("✅ Deribit integration successful!")
        
    except Exception as e:
        print(f"❌ Deribit integration failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("Greeks Calculator Test Suite")
    print("File: tests/test_greeks_calculator.py")
    print("-" * 60)
    
    # Run main tests
    success = test_greeks_calculator()
    
    # Run Deribit integration test
    if success:
        success = test_deribit_integration()
    
    if success:
        print("\n🎉 All tests passed successfully!")
        print("\nNext steps:")
        print("1. Save Options Chain Processor to: src/models/options/options_chain.py")
        print("2. Run: python tests/test_options_chain.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        print("Common issues:")
        print("- Ensure src/models/options/greeks_calculator.py exists")
        print("- Ensure src/models/options/black_scholes.py exists")
        print("- Check that all imports are correct")
    
    sys.exit(0 if success else 1)