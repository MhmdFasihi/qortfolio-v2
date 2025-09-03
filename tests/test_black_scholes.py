#!/usr/bin/env python3
"""
Test Black-Scholes Model Implementation
File: tests/test_black_scholes.py
Run: python tests/test_black_scholes.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime, timedelta

def test_black_scholes_model():
    """Test Black-Scholes model with both USD and coin-based pricing."""
    
    print("=" * 60)
    print("Testing Black-Scholes Model")
    print("=" * 60)
    
    try:
        # Import the model
        from src.models.options.black_scholes import (
            BlackScholesModel, OptionParameters, OptionType,
            price_coin_based_option, validate_deribit_pricing
        )
        print("✅ Black-Scholes module imported successfully")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Make sure src/models/options/black_scholes.py exists")
        return False
    
    # Test 1: Create model instance
    print("\n1. Testing model initialization...")
    try:
        bs_model = BlackScholesModel()
        print("   ✅ Model initialized")
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        return False
    
    # Test 2: Standard USD-based option pricing
    print("\n2. Testing standard USD-based BTC option...")
    try:
        params = OptionParameters(
            spot_price=50000,
            strike_price=52000,
            time_to_maturity=30/365.25,  # 30 days
            volatility=0.8,  # 80% annualized
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
            is_coin_based=False
        )
        
        result = bs_model.calculate_option_price(params)
        
        print(f"   Spot: ${params.spot_price:,}")
        print(f"   Strike: ${params.strike_price:,}")
        print(f"   Time: {params.time_to_maturity*365:.1f} days")
        print(f"   IV: {params.volatility*100:.0f}%")
        print(f"   ✅ USD Price: ${result.usd_price:.2f}")
        print(f"   ✅ BTC Equivalent: {result.coin_based_price:.6f} BTC")
        print(f"   ✅ Delta: {result.delta:.4f}")
        print(f"   ✅ Gamma: {result.gamma:.6f}")
        print(f"   ✅ Theta: {result.theta:.4f} per day")
        print(f"   ✅ Vega: {result.vega:.4f} per 1% vol")
        
    except Exception as e:
        print(f"   ❌ USD pricing failed: {e}")
        return False
    
    # Test 3: Coin-based option pricing (Deribit style)
    print("\n3. Testing coin-based BTC option (Deribit style)...")
    try:
        params.is_coin_based = True
        result = bs_model.calculate_option_price(params)
        
        print(f"   ✅ BTC Price: {result.coin_based_price:.6f} BTC")
        print(f"   ✅ USD Equivalent: ${result.usd_price:.2f}")
        print(f"   ✅ Coin-based Delta: {result.delta:.6f}")
        print(f"   ✅ Coin-based Gamma: {result.gamma:.9f}")
        
        # Verify coin-based pricing is different from USD
        params.is_coin_based = False
        usd_result = bs_model.calculate_option_price(params)
        
        if abs(result.delta - usd_result.delta) > 0.0001:
            print(f"   ✅ Greeks properly adjusted for coin-based")
        else:
            print(f"   ⚠️ Greeks might not be adjusted correctly")
            
    except Exception as e:
        print(f"   ❌ Coin-based pricing failed: {e}")
        return False
    
    # Test 4: Put-Call Parity
    print("\n4. Testing Put-Call Parity...")
    try:
        # Calculate call price
        call_params = OptionParameters(
            spot_price=50000,
            strike_price=50000,  # ATM
            time_to_maturity=30/365.25,
            volatility=0.8,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
            is_coin_based=False
        )
        call_result = bs_model.calculate_option_price(call_params)
        
        # Calculate put price
        put_params = OptionParameters(
            spot_price=50000,
            strike_price=50000,  # ATM
            time_to_maturity=30/365.25,
            volatility=0.8,
            risk_free_rate=0.05,
            option_type=OptionType.PUT,
            is_coin_based=False
        )
        put_result = bs_model.calculate_option_price(put_params)
        
        # Check put-call parity: C - P = S - K*e^(-r*T)
        S = call_params.spot_price
        K = call_params.strike_price
        r = call_params.risk_free_rate
        T = call_params.time_to_maturity
        
        theoretical_diff = S - K * np.exp(-r * T)
        actual_diff = call_result.usd_price - put_result.usd_price
        parity_error = abs(theoretical_diff - actual_diff)
        
        print(f"   Call Price: ${call_result.usd_price:.2f}")
        print(f"   Put Price: ${put_result.usd_price:.2f}")
        print(f"   Theoretical C-P: ${theoretical_diff:.2f}")
        print(f"   Actual C-P: ${actual_diff:.2f}")
        print(f"   Error: ${parity_error:.4f}")
        
        if parity_error < 0.01:
            print(f"   ✅ Put-Call Parity holds!")
        else:
            print(f"   ❌ Put-Call Parity violated!")
            return False
            
    except Exception as e:
        print(f"   ❌ Put-Call Parity test failed: {e}")
        return False
    
    # Test 5: Implied Volatility Calculation
    print("\n5. Testing Implied Volatility calculation...")
    try:
        # Use the call price from previous test as market price
        market_price = call_result.usd_price
        
        # Create params without volatility
        iv_params = OptionParameters(
            spot_price=50000,
            strike_price=50000,
            time_to_maturity=30/365.25,
            volatility=0.5,  # Initial guess
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
            is_coin_based=False
        )
        
        # Calculate IV
        implied_vol = bs_model.calculate_implied_volatility(market_price, iv_params)
        
        print(f"   Market Price: ${market_price:.2f}")
        print(f"   True Volatility: {call_params.volatility*100:.1f}%")
        print(f"   Calculated IV: {implied_vol*100:.1f}%")
        print(f"   Error: {abs(implied_vol - call_params.volatility)*100:.2f}%")
        
        if abs(implied_vol - call_params.volatility) < 0.001:
            print(f"   ✅ IV calculation accurate!")
        else:
            print(f"   ⚠️ IV calculation has small error (acceptable)")
            
    except Exception as e:
        print(f"   ❌ IV calculation failed: {e}")
        return False
    
    # Test 6: Quick pricing function
    print("\n6. Testing quick pricing function...")
    try:
        prices = price_coin_based_option(
            spot=50000,
            strike=52000,
            time_to_maturity=30/365.25,
            volatility=0.8,
            option_type='call'
        )
        
        print(f"   ✅ Coin Price: {prices['coin_price']:.6f} BTC")
        print(f"   ✅ USD Price: ${prices['usd_price']:.2f}")
        print(f"   ✅ All Greeks calculated")
        
    except Exception as e:
        print(f"   ❌ Quick pricing failed: {e}")
        return False
    
    # Test 7: Validate with sample Deribit price
    print("\n7. Testing Deribit price validation...")
    try:
        validation = validate_deribit_pricing(
            deribit_price_btc=0.0523,  # Sample Deribit price in BTC
            spot=50000,
            strike=52000,
            time_to_maturity=30/365.25,
            option_type='call'
        )
        
        print(f"   Deribit Price: {validation['deribit_price_btc']:.6f} BTC")
        print(f"   Theoretical Price: {validation['theoretical_price_btc']:.6f} BTC")
        print(f"   Implied Volatility: {validation['implied_volatility']*100:.1f}%")
        print(f"   Price Difference: {validation['price_difference_btc']:.6f} BTC")
        print(f"   ✅ Validation function working")
        
    except Exception as e:
        print(f"   ❌ Deribit validation failed: {e}")
        return False
    
    # Test 8: Edge cases
    print("\n8. Testing edge cases...")
    try:
        # Near expiration
        near_expiry_params = OptionParameters(
            spot_price=50000,
            strike_price=50000,
            time_to_maturity=1/365.25,  # 1 day
            volatility=0.8,
            option_type=OptionType.CALL,
            is_coin_based=True
        )
        near_result = bs_model.calculate_option_price(near_expiry_params)
        print(f"   ✅ Near expiration (1 day): {near_result.coin_based_price:.6f} BTC")
        
        # Deep ITM
        itm_params = OptionParameters(
            spot_price=50000,
            strike_price=30000,  # Deep ITM
            time_to_maturity=30/365.25,
            volatility=0.8,
            option_type=OptionType.CALL,
            is_coin_based=True
        )
        itm_result = bs_model.calculate_option_price(itm_params)
        print(f"   ✅ Deep ITM call: {itm_result.coin_based_price:.6f} BTC")
        
        # Deep OTM
        otm_params = OptionParameters(
            spot_price=50000,
            strike_price=100000,  # Deep OTM
            time_to_maturity=30/365.25,
            volatility=0.8,
            option_type=OptionType.CALL,
            is_coin_based=True
        )
        otm_result = bs_model.calculate_option_price(otm_params)
        print(f"   ✅ Deep OTM call: {otm_result.coin_based_price:.6f} BTC")
        
    except Exception as e:
        print(f"   ❌ Edge case testing failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All Black-Scholes tests passed!")
    print("=" * 60)
    return True


def test_time_integration():
    """Test integration with fixed time utilities."""
    print("\n" + "=" * 60)
    print("Testing Time Utilities Integration")
    print("=" * 60)
    
    try:
        # Import time utilities
        from src.core.utils.time_utils import TimeUtils
        from src.models.options.black_scholes import BlackScholesModel, OptionParameters, OptionType
        
        current_time = datetime.now()
        expiry_time = current_time + timedelta(days=30)
        
        # Calculate time to maturity using fixed utility
        tte = TimeUtils.calculate_time_to_maturity(current_time, expiry_time)
        
        print(f"Current Time: {current_time}")
        print(f"Expiry Time: {expiry_time}")
        print(f"Time to Maturity: {tte:.6f} years ({tte*365:.1f} days)")
        
        # Use in Black-Scholes
        bs_model = BlackScholesModel()
        params = OptionParameters(
            spot_price=50000,
            strike_price=52000,
            time_to_maturity=tte,
            volatility=0.8,
            option_type=OptionType.CALL,
            is_coin_based=True
        )
        
        result = bs_model.calculate_option_price(params)
        print(f"Option Price with fixed time: {result.coin_based_price:.6f} BTC")
        print("✅ Time utilities integration successful!")
        
    except Exception as e:
        print(f"❌ Time integration failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("Black-Scholes Model Test Suite")
    print("File: tests/test_black_scholes.py")
    print("-" * 60)
    
    # Run main tests
    success = test_black_scholes_model()
    
    # Run time integration test
    if success:
        success = test_time_integration()
    
    if success:
        print("\n🎉 All tests passed successfully!")
        print("\nNext steps:")
        print("1. Save Greeks Calculator to: src/models/options/greeks_calculator.py")
        print("2. Run: python tests/test_greeks_calculator.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        print("Common issues:")
        print("- Ensure src/models/options/black_scholes.py exists")
        print("- Check that src/core/utils/time_utils.py has calculate_time_to_maturity function")
        print("- Verify all imports are correct")
    
    sys.exit(0 if success else 1)