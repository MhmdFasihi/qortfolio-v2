# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Diagnose Deribit Collector Import Issue
"""

import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir.parent / "src"
sys.path.insert(0, str(src_path))

def diagnose_deribit_import():
    """Diagnose the Deribit import issue step by step."""
    
    print("🔍 Diagnosing Deribit Collector Import Issue")
    print("=" * 50)
    
    # Step 1: Check if file exists
    deribit_file = src_path / "data" / "collectors" / "deribit_collector.py"
    print(f"\n1. File exists: {deribit_file.exists()}")
    print(f"   Path: {deribit_file}")
    
    if not deribit_file.exists():
        print("❌ File missing! Create src/data/collectors/deribit_collector.py")
        return False
    
    # Step 2: Try to read file and check for syntax errors
    print("\n2. Checking file syntax...")
    try:
        with open(deribit_file, 'r') as f:
            content = f.read()
        
        # Try to compile
        compile(content, str(deribit_file), 'exec')
        print("   ✅ File syntax is valid")
        
        # Check if it has the expected class
        if 'class DeribitCollector' in content:
            print("   ✅ DeribitCollector class found")
        else:
            print("   ❌ DeribitCollector class missing")
            
        if 'def get_options_data' in content:
            print("   ✅ get_options_data method found")
        else:
            print("   ❌ get_options_data method missing")
            
    except SyntaxError as e:
        print(f"   ❌ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ File read error: {e}")
        return False
    
    # Step 3: Try importing step by step
    print("\n3. Testing imports step by step...")
    
    try:
        print("   Testing: import data")
        import data
        print("   ✅ data package imports ok")
    except Exception as e:
        print(f"   ❌ data package import failed: {e}")
        return False
    
    try:
        print("   Testing: import data.collectors")
        import data.collectors
        print("   ✅ data.collectors package imports ok")
    except Exception as e:
        print(f"   ❌ data.collectors import failed: {e}")
        return False
    
    try:
        print("   Testing: import data.collectors.deribit_collector")
        import data.collectors.deribit_collector
        print("   ✅ deribit_collector module imports ok")
        
        # Check what's actually in the module
        module_attrs = dir(data.collectors.deribit_collector)
        print(f"   📋 Module attributes: {[attr for attr in module_attrs if not attr.startswith('_')]}")
        
    except Exception as e:
        print(f"   ❌ deribit_collector module import failed: {e}")
        print(f"   🔍 Error type: {type(e).__name__}")
        print(f"   🔍 Error details: {str(e)}")
        return False
    
    # Step 4: Try importing specific classes
    print("\n4. Testing specific imports...")
    try:
        from data.collectors.deribit_collector import DeribitCollector
        print("   ✅ DeribitCollector class imports ok")
    except Exception as e:
        print(f"   ❌ DeribitCollector import failed: {e}")
        return False
    
    try:
        from data.collectors.deribit_collector import get_deribit_collector
        print("   ✅ get_deribit_collector function imports ok")
    except Exception as e:
        print(f"   ❌ get_deribit_collector import failed: {e}")
        return False
    
    # Step 5: Test creating collector
    print("\n5. Testing collector creation...")
    try:
        collector = get_deribit_collector()
        print("   ✅ Collector created successfully")
        print(f"   📋 Collector type: {type(collector).__name__}")
        
        # Check methods
        methods = [method for method in dir(collector) if not method.startswith('_')]
        print(f"   📋 Available methods: {methods}")
        
        if hasattr(collector, 'get_options_data'):
            print("   ✅ get_options_data method exists")
        else:
            print("   ❌ get_options_data method missing")
            
    except Exception as e:
        print(f"   ❌ Collector creation failed: {e}")
        return False
    
    print("\n✅ All diagnostics passed!")
    return True

if __name__ == "__main__":
    success = diagnose_deribit_import()
    
    if success:
        print("\n🎯 Deribit collector should work fine!")
        print("The import error might be in the test code itself.")
    else:
        print("\n❌ Found issues with Deribit collector")
        print("These need to be fixed before the collector will work.")