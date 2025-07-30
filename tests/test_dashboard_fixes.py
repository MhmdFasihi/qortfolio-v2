# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Test Dashboard Fixes - Verify Parameter and Method Issues
Location: tests/test_dashboard_fixes.py
"""

import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir.parent / "src"
sys.path.insert(0, str(src_path))

def test_collect_market_data_parameters():
    """Test that collect_market_data now accepts symbols parameter."""
    
    print("🔧 Testing collect_market_data Parameter Fix")
    print("=" * 50)
    
    try:
        from data import collect_market_data
        
        print("\n1. Testing old parameter (symbol)...")
        result1 = collect_market_data(symbol="BTC")
        print(f"   ✅ collect_market_data(symbol='BTC') returned DataFrame with {len(result1)} rows")
        
        print("\n2. Testing new parameter (symbols)...")
        result2 = collect_market_data(symbols=["BTC"])
        print(f"   ✅ collect_market_data(symbols=['BTC']) returned DataFrame with {len(result2)} rows")
        
        print("\n3. Testing new parameter with multiple symbols...")
        result3 = collect_market_data(symbols=["BTC", "ETH"])
        print(f"   ✅ collect_market_data(symbols=['BTC', 'ETH']) returned DataFrame with {len(result3)} rows")
        
        print("\n4. Testing with both parameters...")
        result4 = collect_market_data(symbol="ETH", symbols=["BTC"])
        print(f"   ✅ collect_market_data(symbol='ETH', symbols=['BTC']) returned DataFrame with {len(result4)} rows")
        
        print("\n✅ Parameter fix successful!")
        print("   📋 Dashboard should no longer get 'unexpected keyword argument' error")
        return True
        
    except Exception as e:
        print(f"   ❌ Parameter test failed: {e}")
        return False

def test_dashboard_methods():
    """Test that dashboard methods are available."""
    
    print("\n🔧 Testing Dashboard Methods")
    print("=" * 40)
    
    try:
        from dashboard.dashboard_methods import (
            system_status_page,
            market_overview_page,
            options_chain_page,
            volatility_surface_page
        )
        
        print("   ✅ system_status_page imported")
        print("   ✅ market_overview_page imported") 
        print("   ✅ options_chain_page imported")
        print("   ✅ volatility_surface_page imported")
        
        # Test that they're callable
        for method_name, method in [
            ("system_status_page", system_status_page),
            ("market_overview_page", market_overview_page),
            ("options_chain_page", options_chain_page),
            ("volatility_surface_page", volatility_surface_page)
        ]:
            if callable(method):
                print(f"   ✅ {method_name} is callable")
            else:
                print(f"   ❌ {method_name} is not callable")
                return False
        
        print("\n✅ Dashboard methods available!")
        print("   📋 QortfolioDashboard can be patched with missing methods")
        return True
        
    except ImportError as e:
        print(f"   ❌ Dashboard methods import failed: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Dashboard methods test failed: {e}")
        return False

def test_dashboard_patch():
    """Test the dashboard patching system."""
    
    print("\n🔧 Testing Dashboard Patch System")
    print("=" * 45)
    
    try:
        from dashboard.dashboard_patch import patch_dashboard_class
        
        # Create a mock dashboard class
        class MockDashboard:
            pass
        
        # Patch it
        patched_class = patch_dashboard_class(MockDashboard)
        
        # Test that methods were added
        required_methods = [
            'system_status_page',
            'market_overview_page',
            'options_chain_page',
            'volatility_surface_page',
            'portfolio_management_page',
            'pnl_simulation_page',
            'risk_management_page'
        ]
        
        for method_name in required_methods:
            if hasattr(patched_class, method_name):
                print(f"   ✅ {method_name} added to class")
            else:
                print(f"   ❌ {method_name} missing from class")
                return False
        
        print("\n✅ Dashboard patch system working!")
        print("   📋 Can add missing methods to QortfolioDashboard")
        return True
        
    except Exception as e:
        print(f"   ❌ Dashboard patch test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚨 Dashboard Fixes Validation")
    print("This test verifies fixes for dashboard parameter and method issues")
    
    tests = [
        ("Parameter Fix", test_collect_market_data_parameters),
        ("Dashboard Methods", test_dashboard_methods),
        ("Dashboard Patch", test_dashboard_patch)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"   ❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 DASHBOARD FIXES TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 ALL DASHBOARD FIXES WORKING!")
        print(f"\n📋 Fixed Issues:")
        print(f"   • collect_market_data() now accepts 'symbols' parameter")
        print(f"   • system_status_page and other methods available")
        print(f"   • Dashboard patching system ready")
        print(f"\n🎯 Dashboard should now work without parameter/method errors!")
    else:
        print(f"\n⚠️ Some dashboard fixes failed")
        print(f"Please check the issues above")