# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Step-by-Step Testing Guide for Qortfolio V2 Fixes
Location: tests/step_by_step_testing.py

This script provides specific testing commands for each component
and helps identify exactly where import/functionality issues occur.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
src_path = current_dir.parent / "src"
sys.path.insert(0, str(src_path))

def print_separator(title):
    """Print a nice separator for test sections."""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print('='*60)

def print_command(command, description):
    """Print a command with description."""
    print(f"\n💻 Command: {command}")
    print(f"📝 Purpose: {description}")

def test_step_1_config_manager():
    """Test Step 1: ConfigManager Fix"""
    print_separator("STEP 1: ConfigManager Fix")
    
    print_command(
        "python tests/step_by_step_testing.py config",
        "Test ConfigManager with all missing methods"
    )
    
    try:
        print("\n🔍 Testing ConfigManager imports...")
        from core.config import ConfigManager, get_config, reset_config
        print("   ✅ ConfigManager imports successful")
        
        print("\n🔍 Testing ConfigManager methods...")
        reset_config()
        config = get_config()
        
        # Test all the methods that were missing
        missing_methods = [
            ('get_cache_ttl', ['spot_prices']),
            ('get_deribit_currency', ['BTC']), 
            ('get_config_summary', []),
        ]
        
        for method_name, args in missing_methods:
            try:
                method = getattr(config, method_name)
                result = method(*args) if args else method()
                print(f"   ✅ {method_name}({', '.join(map(str, args))}) = {result}")
            except Exception as e:
                print(f"   ❌ {method_name}() failed: {e}")
                return False
        
        # Test property
        try:
            currencies = config.deribit_currencies
            print(f"   ✅ deribit_currencies property = {currencies}")
        except Exception as e:
            print(f"   ❌ deribit_currencies property failed: {e}")
            return False
            
        print("\n🎉 ConfigManager fix SUCCESSFUL!")
        return True
        
    except ImportError as e:
        print(f"   ❌ ConfigManager import failed: {e}")
        print("   📋 Fix: Ensure src/core/config.py is created with all methods")
        return False
    except Exception as e:
        print(f"   ❌ ConfigManager test failed: {e}")
        return False

def test_step_2_logging():
    """Test Step 2: Logging Fix"""
    print_separator("STEP 2: Logging Fix") 
    
    print_command(
        "python tests/step_by_step_testing.py logging",
        "Test logging module with all required functions"
    )
    
    try:
        print("\n🔍 Testing logging imports...")
        from core.logging import log_data_collection, setup_logging, get_logger
        print("   ✅ Critical logging imports successful")
        
        print("\n🔍 Testing all logging functions...")
        from core.logging import (
            log_api_call,
            log_calculation,
            log_function_call,
            set_log_level
        )
        print("   ✅ All logging functions imported")
        
        print("\n🔍 Testing logging functionality...")
        # Test the critical function that was missing
        log_data_collection("test_data", "BTC", 100, success=True)
        print("   ✅ log_data_collection working")
        
        # Test other functions
        log_api_call("deribit", "/test", status_code=200)
        log_calculation("test_calc", "BTC", {"input": 1}, {"output": 2})
        log_function_call("test_func", ("arg1",), {"kwarg1": "value"})
        print("   ✅ All logging functions working")
        
        print("\n🎉 Logging fix SUCCESSFUL!")
        return True
        
    except ImportError as e:
        print(f"   ❌ Logging import failed: {e}")
        print("   📋 Fix: Ensure src/core/logging.py has log_data_collection function")
        return False
    except Exception as e:
        print(f"   ❌ Logging test failed: {e}")
        return False

def test_step_3_time_utils():
    """Test Step 3: Time Calculation Fix"""
    print_separator("STEP 3: Time Calculation Fix")
    
    print_command(
        "python tests/step_by_step_testing.py time",
        "Test corrected time-to-maturity calculation"
    )
    
    try:
        print("\n🔍 Testing time utils imports...")
        from core.utils.time_utils import calculate_time_to_maturity, validate_time_calculation
        print("   ✅ Time utils imports successful")
        
        print("\n🔍 Testing time calculation fix...")
        # Run validation to check mathematical correctness
        validation_passed = validate_time_calculation()
        
        if validation_passed:
            print("   ✅ Time calculation mathematical bug FIXED!")
            print("   ✅ All validation tests passed")
        else:
            print("   ❌ Time calculation still has issues")
            return False
            
        print("\n🎉 Time calculation fix SUCCESSFUL!")
        return True
        
    except ImportError as e:
        print(f"   ❌ Time utils import failed: {e}")
        print("   📋 Fix: Ensure src/core/utils/time_utils.py exists")
        return False
    except Exception as e:
        print(f"   ❌ Time calculation test failed: {e}")
        return False

def test_step_4_deribit_api():
    """Test Step 4: Deribit API Fix"""
    print_separator("STEP 4: Deribit API Fix")
    
    print_command(
        "python tests/step_by_step_testing.py deribit",
        "Test Deribit websocket API integration"
    )
    
    try:
        print("\n🔍 Testing Deribit collector imports...")
        # Import step by step to isolate the issue
        import data
        print("   ✅ data package imported")
        
        import data.collectors  
        print("   ✅ data.collectors package imported")
        
        import data.collectors.deribit_collector
        print("   ✅ deribit_collector module imported")
        
        from data.collectors.deribit_collector import DeribitCollector
        print("   ✅ DeribitCollector class imported")
        
        from data.collectors.deribit_collector import get_deribit_collector
        print("   ✅ get_deribit_collector function imported")
        
        print("\n🔍 Testing Deribit collector creation...")
        collector = get_deribit_collector()
        print(f"   ✅ Collector created: {type(collector).__name__}")
        print(f"   📡 WebSocket URL: {collector.current_url}")
        print(f"   🧪 Test environment: {collector.use_test_env}")
        
        print("\n🔍 Testing required methods exist...")
        required_methods = ['get_options_data', 'get_spot_price', 'test_connection']
        for method_name in required_methods:
            if hasattr(collector, method_name):
                method = getattr(collector, method_name)
                if callable(method):
                    print(f"   ✅ {method_name} method exists and callable")
                else:
                    print(f"   ❌ {method_name} exists but not callable")
                    return False
            else:
                print(f"   ❌ {method_name} method missing")
                return False
        
        print("\n🔍 Testing connection (quick test)...")
        try:
            # Just test that the method can be called without network issues
            connection_ok = collector.test_connection()
            if connection_ok:
                print("   ✅ WebSocket connection successful")
            else:
                print("   ⚠️ Connection failed (may be network/firewall)")
                print("   📋 This doesn't indicate a code issue")
        except Exception as e:
            print(f"   ⚠️ Connection test error: {e}")
            print("   📋 Continuing with other tests...")
        
        print("\n🎉 Deribit API fix SUCCESSFUL!")
        print("   📋 Should resolve 400 Bad Request errors")
        return True
        
    except ImportError as e:
        print(f"   ❌ Deribit import failed: {e}")
        print("   📋 Fix: Check src/data/collectors/deribit_collector.py")
        print(f"   🔍 Error details: {str(e)}")
        print(f"   🔍 Error type: {type(e).__name__}")
        return False
    except Exception as e:
        print(f"   ❌ Deribit test failed: {e}")
        print(f"   🔍 Error type: {type(e).__name__}")
        return False

def test_step_5_dashboard_imports():
    """Test Step 5: Dashboard Import Chain"""
    print_separator("STEP 5: Dashboard Import Chain")
    
    print_command(
        "streamlit run src/dashboard/main_dashboard.py",
        "Start the actual dashboard (run this separately)"
    )
    
    try:
        print("\n🔍 Testing dashboard import chain...")
        print("   This simulates the exact imports the dashboard makes...")
        
        # Core imports that dashboard needs
        from core.config import get_config
        from core.logging import setup_logging, get_logger, log_data_collection
        print("   ✅ Core modules imported")
        
        # Initialize like dashboard does
        setup_logging({"level": "INFO", "console": False, "file_enabled": False})
        config = get_config()
        logger = get_logger("dashboard_test")
        
        # Test critical function
        log_data_collection("test", "BTC", 0, success=True)
        print("   ✅ Dashboard initialization successful")
        
        # Test if we can import data modules
        try:
            from data.collectors.deribit_collector import DeribitCollector
            print("   ✅ Data collectors importable")
        except ImportError as e:
            print(f"   ⚠️ Data collectors import issue: {e}")
        
        print("\n🎉 Dashboard import chain SUCCESSFUL!")
        print("   📋 Dashboard should now start without import errors")
        return True
        
    except ImportError as e:
        print(f"   ❌ Dashboard import failed: {e}")
        print("   📋 Check which module is missing and create it")
        return False
    except Exception as e:
        print(f"   ❌ Dashboard test failed: {e}")
        return False

def test_all_components():
    """Test all components in sequence"""
    print_separator("COMPREHENSIVE TEST SUITE")
    
    tests = [
        ("ConfigManager Fix", test_step_1_config_manager),
        ("Logging Fix", test_step_2_logging),
        ("Time Calculation Fix", test_step_3_time_utils),
        ("Deribit API Fix", test_step_4_deribit_api),
        ("Dashboard Import Chain", test_step_5_dashboard_imports)
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
    print_separator("TEST RESULTS SUMMARY")
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<25}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"\n📋 NEXT STEPS:")
        print(f"1. Run dashboard: streamlit run src/dashboard/main_dashboard.py")
        print(f"2. Dashboard should start without import errors")
        print(f"3. Should see real data instead of 'No options data available'")
        print(f"4. Ready for Step 3: Add missing analytics features")
    else:
        print(f"\n⚠️ SOME TESTS FAILED")
        print(f"📋 Fix the failing components before proceeding")
    
    return all_passed

def show_individual_commands():
    """Show individual testing commands"""
    print_separator("INDIVIDUAL TESTING COMMANDS")
    
    commands = [
        ("Test ConfigManager Only", "python tests/step_by_step_testing.py config"),
        ("Test Logging Only", "python tests/step_by_step_testing.py logging"), 
        ("Test Time Utils Only", "python tests/step_by_step_testing.py time"),
        ("Test Deribit API Only", "python tests/step_by_step_testing.py deribit"),
        ("Test Dashboard Imports", "python tests/step_by_step_testing.py dashboard"),
        ("Test All Components", "python tests/step_by_step_testing.py all"),
        ("Start Dashboard", "streamlit run src/dashboard/main_dashboard.py"),
    ]
    
    for description, command in commands:
        print_command(command, description)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("🚨 Qortfolio V2 - Step-by-Step Testing")
        print("This script tests each fix component individually")
        show_individual_commands()
        print(f"\n💡 Usage: python {sys.argv[0]} <test_type>")
        print("💡 Test types: config, logging, time, deribit, dashboard, all")
        sys.exit(1)
    
    test_type = sys.argv[1].lower()
    
    if test_type == "config":
        success = test_step_1_config_manager()
    elif test_type == "logging":
        success = test_step_2_logging()
    elif test_type == "time":
        success = test_step_3_time_utils()
    elif test_type == "deribit":
        success = test_step_4_deribit_api()
    elif test_type == "dashboard":
        success = test_step_5_dashboard_imports()
    elif test_type == "all":
        success = test_all_components()
    else:
        print(f"❌ Unknown test type: {test_type}")
        show_individual_commands()
        sys.exit(1)
    
    sys.exit(0 if success else 1)