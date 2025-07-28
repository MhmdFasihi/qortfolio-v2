# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Comprehensive Test Runner for Qortfolio V2 - Step 1.3 Fix
Location: qortfolio-v2/run_comprehensive_tests.py

Fixed version that resolves all logging import issues.
"""

import sys
import os
import subprocess
import traceback
import logging as std_logging  # Fix: Use standard logging, not our custom one
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test all critical imports."""
    print("🔍 Testing Critical Imports")
    print("-" * 30)
    
    import_tests = [
        ("Core Config", "from core.config import get_config"),
        ("Core Logging", "from core.logging import setup_logging, get_logger"),
        ("Time Utils", "from core.utils.time_utils import calculate_time_to_maturity"),
        ("Crypto Collector", "from data.collectors.crypto_collector import CryptoCollector"),
        ("Deribit Collector", "from data.collectors.deribit_collector import DeribitCollector"),
        ("Data Manager", "from data.collectors.data_manager import DataManager"),
        ("Black-Scholes", "from models.options.black_scholes import BlackScholesModel"),
        ("PnL Simulator", "from analytics.pnl_simulator import TaylorExpansionPnL"),
        ("Volatility Surface", "from analytics.volatility_surface import VolatilitySurfaceAnalyzer"),
    ]
    
    passed = 0
    total = len(import_tests)
    
    for test_name, import_statement in import_tests:
        try:
            exec(import_statement)
            print(f"  ✅ {test_name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {test_name}: {e}")
    
    print(f"\n📊 Import Tests: {passed}/{total} passed ({passed/total*100:.1f}%)")
    return passed == total

def test_time_calculation_fix():
    """Test the critical time calculation bug fix."""
    print("\n⏰ Testing Time Calculation Bug Fix")
    print("-" * 35)
    
    try:
        from core.utils.time_utils import calculate_time_to_maturity
        from datetime import datetime
        
        # Test case: exactly 30 days
        current = datetime(2024, 1, 1)
        expiry = datetime(2024, 1, 31)
        
        result = calculate_time_to_maturity(current, expiry)
        expected = 30 / 365.25  # Correct calculation
        
        error = abs(result - expected)
        if error < 1e-6:
            print(f"  ✅ Time calculation: {result:.6f} years (expected {expected:.6f})")
            print(f"  ✅ Error: {error:.2e} (acceptable)")
            return True
        else:
            print(f"  ❌ Time calculation: {result:.6f} years (expected {expected:.6f})")
            print(f"  ❌ Error: {error:.2e} (too large)")
            return False
            
    except Exception as e:
        print(f"  ❌ Time calculation test failed: {e}")
        return False

def test_data_collection():
    """Test data collection systems."""
    print("\n📊 Testing Data Collection")
    print("-" * 25)
    
    try:
        from data.collectors.data_manager import DataManager
        from core.config import get_config
        from core.logging import setup_logging  # Fix: Import our logging system
        
        # Setup logging first
        setup_logging()
        logger = std_logging.getLogger("test_data_collection")  # Fix: Use standard logging
        
        # Test configuration
        config = get_config()
        cryptos = config.enabled_cryptocurrencies
        print(f"  ✅ Configuration: {len(cryptos)} cryptocurrencies enabled")
        
        # Test data manager initialization
        dm = DataManager()
        print("  ✅ DataManager initialized successfully")
        
        # Test collectors exist
        if hasattr(dm, 'crypto_collector') and hasattr(dm, 'deribit_collector'):
            print("  ✅ Both collectors initialized")
            return True
        else:
            print("  ❌ Collectors not properly initialized")
            return False
            
    except Exception as e:
        print(f"  ❌ Data collection test failed: {e}")
        std_logging.error(f"Data collection test error: {e}")  # Fix: Use standard logging
        return False

def test_financial_models():
    """Test financial models."""
    print("\n🧮 Testing Financial Models")
    print("-" * 26)
    
    try:
        from models.options.black_scholes import BlackScholesModel, OptionParameters, OptionType
        from core.logging import setup_logging  # Fix: Import our logging system
        
        # Setup logging first
        setup_logging()
        logger = std_logging.getLogger("test_financial_models")  # Fix: Use standard logging
        
        # Test Black-Scholes
        bs_model = BlackScholesModel()
        
        params = OptionParameters(
            spot_price=50000.0,
            strike_price=52000.0,
            time_to_maturity=30/365.25,
            volatility=0.8,
            risk_free_rate=0.05,
            option_type=OptionType.CALL
        )
        
        result = bs_model.calculate_greeks(params)
        
        # Basic sanity checks
        if 0 < result.delta < 1 and result.gamma > 0 and result.vega > 0:
            print(f"  ✅ Black-Scholes: Price=${result.option_price:.2f}")
            print(f"  ✅ Greeks: δ={result.delta:.3f}, γ={result.gamma:.6f}")
            return True
        else:
            print(f"  ❌ Black-Scholes: Unrealistic results")
            return False
            
    except Exception as e:
        print(f"  ❌ Financial models test failed: {e}")
        std_logging.error(f"Financial models test error: {e}")  # Fix: Use standard logging
        return False

def test_analytics():
    """Test analytics modules."""
    print("\n📈 Testing Analytics Modules")
    print("-" * 28)
    
    # Setup logging first
    try:
        from core.logging import setup_logging
        setup_logging()
        logger = std_logging.getLogger("test_analytics")  # Fix: Use standard logging
    except Exception as e:
        print(f"  ⚠️ Logging setup failed: {e}")
    
    # Test PnL Simulator
    try:
        from analytics.pnl_simulator import TaylorExpansionPnL, create_sample_portfolio
        
        simulator = TaylorExpansionPnL()
        positions = create_sample_portfolio()
        
        result = simulator.calculate_portfolio_pnl(
            positions, spot_shock=0.1, vol_shock=0.05, time_decay_days=7
        )
        
        if isinstance(result['portfolio_pnl']['taylor_total'], (int, float)):
            print(f"  ✅ PnL Simulator: Taylor PnL = ${result['portfolio_pnl']['taylor_total']:,.2f}")
            pnl_test_passed = True
        else:
            print("  ❌ PnL Simulator: Invalid result type")
            pnl_test_passed = False
            
    except Exception as e:
        print(f"  ❌ PnL Simulator test failed: {e}")
        std_logging.error(f"PnL Simulator test error: {e}")  # Fix: Use standard logging
        pnl_test_passed = False
    
    # Test Volatility Surface
    try:
        from analytics.volatility_surface import VolatilitySurfaceAnalyzer, create_sample_options_data
        
        analyzer = VolatilitySurfaceAnalyzer()
        options_df = create_sample_options_data()
        surface_data = analyzer.build_volatility_surface(options_df, 50000.0)
        
        if len(surface_data.surface_points) > 0:
            print(f"  ✅ Volatility Surface: {len(surface_data.surface_points)} points")
            vol_test_passed = True
        else:
            print("  ❌ Volatility Surface: No surface points generated")
            vol_test_passed = False
            
    except Exception as e:
        print(f"  ❌ Volatility Surface test failed: {e}")
        std_logging.error(f"Volatility Surface test error: {e}")  # Fix: Use standard logging
        vol_test_passed = False
    
    return pnl_test_passed and vol_test_passed

def test_dashboard_imports():
    """Test dashboard can import required modules."""
    print("\n🖥️ Testing Dashboard Integration")
    print("-" * 31)
    
    try:
        # Test if dashboard can import core modules
        import streamlit as st
        print("  ✅ Streamlit available")
        
        # Test if our modules can be imported by dashboard
        from data.collectors.data_manager import DataManager
        from models.options.black_scholes import BlackScholesModel
        from analytics.pnl_simulator import TaylorExpansionPnL
        from analytics.volatility_surface import VolatilitySurfaceAnalyzer
        
        print("  ✅ All dashboard dependencies available")
        return True
        
    except Exception as e:
        print(f"  ❌ Dashboard integration test failed: {e}")
        std_logging.error(f"Dashboard integration test error: {e}")  # Fix: Use standard logging
        return False

def run_pytest():
    """Run pytest if available."""
    print("\n🧪 Running pytest")
    print("-" * 15)
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print("  ✅ All pytest tests passed")
            return True
        else:
            print("  ⚠️ Some pytest tests failed")
            print("  stdout:", result.stdout[-500:] if result.stdout else "None")
            print("  stderr:", result.stderr[-500:] if result.stderr else "None")
            return False
            
    except subprocess.TimeoutExpired:
        print("  ⚠️ pytest timed out (tests may be slow)")
        return False
    except FileNotFoundError:
        print("  ⚠️ pytest not available")
        return False
    except Exception as e:
        print(f"  ⚠️ pytest error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Qortfolio V2 Comprehensive Test Runner")
    print("=" * 45)
    
    test_results = []
    
    # Run all tests
    test_results.append(("Import Tests", test_imports()))
    test_results.append(("Time Calculation Fix", test_time_calculation_fix()))
    test_results.append(("Data Collection", test_data_collection()))
    test_results.append(("Financial Models", test_financial_models()))
    test_results.append(("Analytics Modules", test_analytics()))
    test_results.append(("Dashboard Integration", test_dashboard_imports()))
    test_results.append(("pytest Suite", run_pytest()))
    
    # Summary
    print("\n" + "=" * 45)
    print("📋 TEST SUMMARY")
    print("=" * 45)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your Qortfolio V2 is ready for Step 2!")
        print("\n📋 Next Steps:")
        print("1. Step 2: Implement real-time ticker configuration")
        print("2. Use your crypto_sectors.json file")
        print("3. Add commodities support")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Checking specific issues...")
        if not test_results[6][1]:  # pytest failed
            print("🔧 pytest failures likely due to config test expectations")
            print("   -> Will fix in Step 1.3B")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)