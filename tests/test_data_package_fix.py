# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Test Data Package Fix - Verify Dashboard Import Compatibility
Location: tests/test_data_package_fix.py

This test validates that the data package provides all functions
the dashboard expects to import.
"""

import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir.parent / "src"
sys.path.insert(0, str(src_path))

def test_data_package_imports():
    """Test that data package imports work correctly."""
    
    print("🔧 Testing Data Package Import Fix")
    print("=" * 45)
    
    try:
        print("\n1. Testing data package import...")
        import data
        print("   ✅ data package imported successfully")
        
        print("\n2. Testing dashboard-expected functions...")
        
        # These are the exact imports the dashboard tries to make
        expected_functions = [
            'get_data_manager',
            'collect_market_data', 
            'get_spot_price'
        ]
        
        for func_name in expected_functions:
            try:
                func = getattr(data, func_name)
                if callable(func):
                    print(f"   ✅ {func_name} function available and callable")
                else:
                    print(f"   ❌ {func_name} exists but not callable")
                    return False
            except AttributeError:
                print(f"   ❌ {func_name} function missing from data package")
                return False
        
        print("\n3. Testing direct imports (like dashboard does)...")
        try:
            from data import get_data_manager, collect_market_data, get_spot_price
            print("   ✅ Direct import successful (dashboard style)")
        except ImportError as e:
            print(f"   ❌ Direct import failed: {e}")
            return False
        
        print("\n4. Testing function calls...")
        try:
            # Test get_data_manager
            data_manager = get_data_manager()
            print(f"   ✅ get_data_manager() returned: {type(data_manager).__name__}")
            
            # Test get_spot_price (should return a number or fallback)
            btc_price = get_spot_price('BTC')
            print(f"   ✅ get_spot_price('BTC') returned: ${btc_price:,.2f}")
            
            # Test collect_market_data (should return DataFrame)
            market_data = collect_market_data('BTC')
            print(f"   ✅ collect_market_data('BTC') returned: DataFrame with {len(market_data)} rows")
            
        except Exception as e:
            print(f"   ❌ Function call failed: {e}")
            return False
        
        print("\n✅ Data package fix SUCCESSFUL!")
        print("   📋 Dashboard should now import data functions without errors")
        return True
        
    except ImportError as e:
        print(f"   ❌ Data package import failed: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def test_deribit_collector_direct():
    """Test that DeribitCollector can be imported and used directly."""
    
    print("\n🔧 Testing Direct DeribitCollector Access")
    print("=" * 45)
    
    try:
        print("\n1. Testing direct collector import...")
        from data.collectors.deribit_collector import DeribitCollector, get_deribit_collector
        print("   ✅ Direct collector imports successful")
        
        print("\n2. Testing collector creation...")
        collector = get_deribit_collector()
        print(f"   ✅ Collector created: {type(collector).__name__}")
        
        print("\n3. Testing collector methods...")
        required_methods = ['get_options_data', 'get_spot_price']
        for method_name in required_methods:
            if hasattr(collector, method_name) and callable(getattr(collector, method_name)):
                print(f"   ✅ {method_name} method available")
            else:
                print(f"   ❌ {method_name} method missing or not callable")
                return False
        
        print("\n✅ Direct DeribitCollector access working!")
        return True
        
    except Exception as e:
        print(f"   ❌ Direct collector test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚨 Data Package Fix Validation")
    print("This test verifies the fix for data package import issues")
    
    success1 = test_data_package_imports()
    success2 = test_deribit_collector_direct()
    
    if success1 and success2:
        print("\n🎉 ALL DATA PACKAGE TESTS PASSED!")
        print("\n📋 What's Fixed:")
        print("   • data package provides expected functions")
        print("   • Dashboard can import: get_data_manager, collect_market_data, get_spot_price")
        print("   • DeribitCollector works directly")
        print("   • No more 'cannot import name' errors")
        print("\n🎯 Ready to run: python tests/step_by_step_testing.py deribit")
    else:
        print("\n⚠️ Some data package tests failed")
        print("Please check the issues above")