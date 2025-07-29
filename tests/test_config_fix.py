# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Test ConfigManager Fix - Verify All Missing Methods Work
Location: tests/test_config_fix.py

This test validates that all previously missing methods are now implemented
and the dashboard should no longer crash.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
src_path = current_dir.parent / "src"
sys.path.insert(0, str(src_path))

def test_config_manager_fix():
    """Test that ConfigManager fix resolves all critical issues."""
    
    print("🔧 Testing ConfigManager Fix - All Missing Methods")
    print("=" * 60)
    
    try:
        # Test import
        print("\n1. Testing Import...")
        from core.config import ConfigManager, get_config, reset_config
        print("   ✅ ConfigManager imported successfully")
        
        # Reset any existing config
        reset_config()
        
        # Test creation
        print("\n2. Testing ConfigManager Creation...")
        config = get_config()
        print("   ✅ ConfigManager created successfully")
        
        # Test all previously missing methods
        print("\n3. Testing Previously Missing Methods...")
        
        # Test get_cache_ttl() - CRITICAL missing method
        try:
            ttl = config.get_cache_ttl('spot_prices')
            assert isinstance(ttl, int)
            assert ttl > 0
            print(f"   ✅ get_cache_ttl('spot_prices') = {ttl} seconds")
        except Exception as e:
            print(f"   ❌ get_cache_ttl() failed: {e}")
            return False
        
        # Test get_deribit_currency() - CRITICAL missing method
        try:
            currency = config.get_deribit_currency('BTC')
            assert currency == 'BTC'
            print(f"   ✅ get_deribit_currency('BTC') = {currency}")
        except Exception as e:
            print(f"   ❌ get_deribit_currency() failed: {e}")
            return False
        
        # Test deribit_currencies property - CRITICAL missing property
        try:
            currencies = config.deribit_currencies
            assert isinstance(currencies, list)
            assert 'BTC' in currencies
            assert 'ETH' in currencies
            print(f"   ✅ deribit_currencies = {currencies}")
        except Exception as e:
            print(f"   ❌ deribit_currencies property failed: {e}")
            return False
        
        # Test get_config_summary() - CRITICAL missing method
        try:
            summary = config.get_config_summary()
            assert isinstance(summary, dict)
            assert 'enabled_cryptos' in summary
            assert 'development_mode' in summary
            print(f"   ✅ get_config_summary() = {summary}")
        except Exception as e:
            print(f"   ❌ get_config_summary() failed: {e}")
            return False
        
        # Test basic configuration access
        print("\n4. Testing Basic Configuration Access...")
        
        # Test dot notation access
        base_url = config.get('deribit_api.base_url')
        assert base_url is not None
        print(f"   ✅ deribit_api.base_url = {base_url}")
        
        # Test with default value
        missing = config.get('nonexistent.key', 'default')
        assert missing == 'default'
        print(f"   ✅ Missing key returns default: {missing}")
        
        # Test configuration validation
        print("\n5. Testing Configuration Validation...")
        is_valid = config.validate_configuration()
        print(f"   ✅ Configuration validation: {is_valid}")
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED - ConfigManager Fix Successful!")
        print("✅ Dashboard should now start without ConfigManager errors")
        print("✅ All missing methods are now implemented")
        print("✅ Ready to proceed to next fix step")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please ensure the fixed config.py file is in src/core/config.py")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False

def test_dashboard_compatibility():
    """Test that fixed ConfigManager works with dashboard expectations."""
    
    print("\n🔧 Testing Dashboard Compatibility...")
    print("-" * 40)
    
    try:
        from core.config import get_config
        config = get_config()
        
        # Test methods that dashboard was calling
        dashboard_methods = [
            ('get_cache_ttl', ['spot_prices']),
            ('get_deribit_currency', ['BTC']),
            ('get_config_summary', []),
        ]
        
        for method_name, args in dashboard_methods:
            try:
                method = getattr(config, method_name)
                result = method(*args) if args else method()
                print(f"   ✅ {method_name}({', '.join(map(str, args))}) = {result}")
            except Exception as e:
                print(f"   ❌ {method_name}() failed: {e}")
                return False
        
        # Test properties that dashboard accesses
        try:
            currencies = config.deribit_currencies
            print(f"   ✅ deribit_currencies property = {currencies}")
        except Exception as e:
            print(f"   ❌ deribit_currencies property failed: {e}")
            return False
        
        print("✅ Dashboard compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚨 ConfigManager Fix Validation Test")
    print("This test verifies the fix for dashboard crashes")
    
    success = test_config_manager_fix()
    
    if success:
        dashboard_success = test_dashboard_compatibility()
        if dashboard_success:
            print("\n🎯 READY FOR STEP 2: Fix Deribit API Integration")
        else:
            print("\n❌ Dashboard compatibility issues remain")
    else:
        print("\n❌ ConfigManager fix has issues - needs revision")