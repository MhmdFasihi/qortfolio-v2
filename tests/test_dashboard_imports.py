#!/usr/bin/env python3
"""
Test Dashboard Import Fix
Location: qortfolio-v2/test_dashboard_imports.py

Test that all dashboard imports work after applying the fix.
"""

import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

def test_dashboard_imports():
    """Test the exact imports that dashboard expects."""
    
    print("🔧 Testing Dashboard Import Fix")
    print("=" * 40)
    
    try:
        # This is the exact import that was failing
        from analytics.pnl_simulator import TaylorPnLSimulator, MarketScenario
        print("✅ Main dashboard imports successful!")
        print("   - TaylorPnLSimulator: ✅")
        print("   - MarketScenario: ✅")
        
        # Test other expected imports
        from analytics.pnl_simulator import PnLResult, simulate_option_pnl
        print("   - PnLResult: ✅")
        print("   - simulate_option_pnl: ✅")
        
        # Test volatility surface import
        from analytics.volatility_surface import VolatilitySurfaceAnalyzer, analyze_options_volatility
        print("   - VolatilitySurfaceAnalyzer: ✅")
        print("   - analyze_options_volatility: ✅")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n📝 To fix:")
        print("1. Add the missing classes to src/analytics/pnl_simulator.py")
        print("2. Make sure src/analytics/volatility_surface.py exists")
        return False
    except Exception as e:
        print(f"❌ Other Error: {e}")
        return False

def test_dashboard_functionality():
    """Test basic dashboard functionality."""
    
    print("\n🧪 Testing Dashboard Functionality")
    print("=" * 40)
    
    try:
        from analytics.pnl_simulator import TaylorPnLSimulator, MarketScenario, simulate_option_pnl
        
        # Test 1: Create simulator
        simulator = TaylorPnLSimulator()
        print("✅ TaylorPnLSimulator created")
        
        # Test 2: Create scenarios
        scenarios = [
            MarketScenario(1000, 1, 0.1, 0, "Test_Up"),
            MarketScenario(-1000, 1, -0.1, 0, "Test_Down")
        ]
        print(f"✅ Created {len(scenarios)} scenarios")
        
        # Test 3: Simulate PnL  
        results = simulate_option_pnl(50000, 52000, 30/365.25, 0.8, "call", 1, scenarios)
        print(f"✅ PnL simulation: {len(results)} results")
        
        # Test 4: Check results structure
        if results:
            result = results[0]
            print(f"   - Scenario: {result.scenario.scenario_name}")
            print(f"   - Taylor PnL: ${result.taylor_total_pnl:.2f}")
            print(f"   - Delta PnL: ${result.delta_pnl:.2f}")
            print(f"   - Has expected attributes: ✅")
        
        # Test 5: Accuracy analysis
        accuracy = simulator.analyze_taylor_accuracy(results)
        print(f"✅ Accuracy analysis works")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def test_original_classes_still_work():
    """Test that original implementation still works."""
    
    print("\n🔄 Testing Original Classes Still Work")
    print("=" * 40)
    
    try:
        # Test original TaylorExpansionPnL
        from analytics.pnl_simulator import TaylorExpansionPnL
        
        original = TaylorExpansionPnL()
        print("✅ Original TaylorExpansionPnL still works")
        
        # Test it can still do analysis
        from models.options.black_scholes import OptionParameters, OptionType
        
        params = OptionParameters(
            spot_price=50000, 
            strike_price=52000,
            time_to_expiry=30/365.25,
            volatility=0.8,
            risk_free_rate=0.05,
            option_type=OptionType.CALL
        )
        
        results = original.analyze_scenarios(params)
        print(f"✅ Original analysis works: {len(results)} results")
        
        return True
        
    except Exception as e:
        print(f"❌ Original classes test failed: {e}")
        return False

def simulate_dashboard_startup():
    """Simulate what happens when dashboard starts up."""
    
    print("\n🖥️ Simulating Dashboard Startup")
    print("=" * 40)
    
    try:
        # This simulates the exact import sequence in main_dashboard.py
        from data import get_data_manager, collect_market_data, get_spot_price
        print("✅ Data modules imported")
        
        from models.options.black_scholes import price_option, calculate_greeks
        print("✅ Black-Scholes modules imported")
        
        from models.options.greeks_calculator import GreeksCalculator, analyze_portfolio_risk
        print("✅ Greeks calculator imported")
        
        from analytics.pnl_simulator import TaylorPnLSimulator, MarketScenario
        print("✅ PnL simulator imported")
        
        from analytics.volatility_surface import VolatilitySurfaceAnalyzer, analyze_options_volatility
        print("✅ Volatility surface imported")
        
        from core.config import get_config
        print("✅ Config imported")
        
        from core.logging import setup_logging, get_logger
        print("✅ Logging imported")
        
        print("\n🎉 All dashboard imports successful!")
        print("Dashboard should now start without import errors!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Dashboard simulation failed: {e}")
        print(f"Missing module: {str(e).split()[-1] if 'No module named' in str(e) else 'Unknown'}")
        return False
    except Exception as e:
        print(f"❌ Other dashboard error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing Dashboard Import Fix")
    print("=" * 50)
    
    # Run all tests
    tests = [
        ("Dashboard Imports", test_dashboard_imports),
        ("Dashboard Functionality", test_dashboard_functionality), 
        ("Original Classes", test_original_classes_still_work),
        ("Dashboard Startup", simulate_dashboard_startup)
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n📋 Next Steps:")
        print("1. Add the missing classes to src/analytics/pnl_simulator.py")
        print("2. Make sure src/analytics/volatility_surface.py exists")
        print("3. Run: streamlit run src/dashboard/main_dashboard.py")
        print("4. Dashboard should start without import errors!")
    else:
        print("\n⚠️ Some tests failed.")
        print("Apply the fixes above and re-run this test.")