#!/usr/bin/env python3
'''
Sample usage of Financial Models
File: examples/use_financial_models.py
'''

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.options.black_scholes import price_coin_based_option
from src.models.options.greeks_calculator import calculate_option_greeks
from src.models.options.options_chain import process_deribit_options

# Example 1: Price a BTC option (coin-based)
print("BTC Call Option (Deribit style):")
result = price_coin_based_option(
    spot=50000,
    strike=52000,
    time_to_maturity=30/365.25,
    volatility=0.8,
    option_type='call'
)
print(f"  Price: {result['coin_price']:.6f} BTC (${result['usd_price']:.2f})")
print(f"  Delta: {result['delta']:.6f}")
print(f"  Gamma: {result['gamma']:.9f}")

# Example 2: Calculate Greeks
print("\nGreeks Calculation:")
greeks = calculate_option_greeks(
    spot=50000,
    strike=50000,  # ATM
    time_to_maturity=7/365.25,  # 1 week
    volatility=0.75,
    option_type='put',
    is_coin_based=True
)
for name, value in greeks.items():
    if value is not None:
        print(f"  {name}: {value:.6f}")

print("\n✅ Financial models working!")
