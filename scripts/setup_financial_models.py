#!/usr/bin/env python3
"""
Setup and Test Script for Financial Models
File: scripts/setup_financial_models.py
Run: python scripts/setup_financial_models.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}! {text}{Colors.END}")

def print_info(text):
    """Print info message."""
    print(f"{Colors.BLUE}→ {text}{Colors.END}")

def check_project_root():
    """Check if we're in the project root."""
    if not os.path.exists("requirements.txt"):
        print_error("Not in project root directory!")
        print_info("Please run this script from the qortfolio-v2 directory")
        return False
    return True

def create_directories():
    """Create necessary directories."""
    print_header("Step 1: Creating Directory Structure")
    
    directories = [
        "src/models/options",
        "tests",
        "scripts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print_success(f"Directory created: {directory}")
    
    # Create __init__ files
    init_files = [
        "src/models/__init__.py",
        "src/models/options/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch(exist_ok=True)
        print_success(f"Init file created: {init_file}")

def check_model_files():
    """Check if model files exist."""
    print_header("Step 2: Checking Model Files")
    
    model_files = [
        ("src/models/options/black_scholes.py", "Black-Scholes Model"),
        ("src/models/options/greeks_calculator.py", "Greeks Calculator"),
        ("src/models/options/options_chain.py", "Options Chain Processor")
    ]
    
    all_exist = True
    for file_path, description in model_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print_success(f"{description}: {file_path} ({file_size} bytes)")
        else:
            print_error(f"{description}: {file_path} NOT FOUND")
            print_info("  Please save the artifact content to this file")
            all_exist = False
    
    return all_exist

def check_test_files():
    """Check if test files exist."""
    print_header("Step 3: Checking Test Files")
    
    test_files = [
        ("tests/test_black_scholes.py", "Black-Scholes Test"),
        ("tests/test_greeks_calculator.py", "Greeks Calculator Test"),
        ("tests/test_options_chain.py", "Options Chain Test"),
        ("tests/test_complete_integration.py", "Integration Test")
    ]
    
    all_exist = True
    for file_path, description in test_files:
        if os.path.exists(file_path):
            print_success(f"{description}: {file_path}")
        else:
            print_error(f"{description}: {file_path} NOT FOUND")
            print_info("  Please save the test artifact to this file")
            all_exist = False
    
    return all_exist

def check_dependencies():
    """Check Python dependencies."""
    print_header("Step 4: Checking Python Dependencies")
    
    dependencies = [
        "numpy",
        "pandas",
        "scipy",
        "pymongo",
        "yaml"
    ]
    
    all_installed = True
    for dep in dependencies:
        try:
            __import__(dep)
            print_success(f"{dep} installed")
        except ImportError:
            print_error(f"{dep} NOT installed")
            print_info(f"  Run: pip install {dep}")
            all_installed = False
    
    return all_installed

def check_mongodb():
    """Check MongoDB status."""
    print_header("Step 5: Checking MongoDB")
    
    try:
        result = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if "qortfolio-mongo" in result.stdout:
            print_success("MongoDB container is running")
            return True
        else:
            print_warning("MongoDB container not running")
            print_info("  Run: docker-compose up -d mongodb")
            return False
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_warning("Docker not available or not running")
        print_info("  Make sure Docker is installed and running")
        return False

def run_tests():
    """Run all tests."""
    print_header("Step 6: Running Tests")
    
    tests = [
        ("tests/test_black_scholes.py", "Black-Scholes Model"),
        ("tests/test_greeks_calculator.py", "Greeks Calculator"),
        ("tests/test_options_chain.py", "Options Chain Processor"),
        ("tests/test_complete_integration.py", "Complete Integration")
    ]
    
    all_passed = True
    
    for test_file, description in tests:
        if not os.path.exists(test_file):
            print_warning(f"Skipping {description} - test file not found")
            continue
        
        print(f"\n{Colors.YELLOW}Testing {description}...{Colors.END}")
        
        try:
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                print_success(f"{description} tests passed")
                # Show key results
                if "✅" in result.stdout:
                    for line in result.stdout.split('\n'):
                        if "✅" in line and not "All" in line:
                            print(f"    {line.strip()}")
            else:
                print_error(f"{description} tests failed")
                # Show error details
                if result.stderr:
                    print(f"    Error: {result.stderr.split('Error:')[-1].strip()[:100]}")
                all_passed = False
                
        except Exception as e:
            print_error(f"Failed to run {description}: {e}")
            all_passed = False
    
    return all_passed

def create_sample_script():
    """Create a sample usage script."""
    print_header("Creating Sample Usage Script")
    
    sample_script = """#!/usr/bin/env python3
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
print("\\nGreeks Calculation:")
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

print("\\n✅ Financial models working!")
"""
    
    # Create examples directory
    Path("examples").mkdir(exist_ok=True)
    
    # Save sample script
    sample_file = "examples/use_financial_models.py"
    with open(sample_file, "w") as f:
        f.write(sample_script)
    
    print_success(f"Sample script created: {sample_file}")
    print_info("Run: python examples/use_financial_models.py")

def main():
    """Main setup function."""
    print("=" * 60)
    print("QORTFOLIO V2 - FINANCIAL MODELS SETUP")
    print("=" * 60)
    
    # Check project root
    if not check_project_root():
        return False
    
    # Create directories
    create_directories()
    
    # Check files
    models_exist = check_model_files()
    tests_exist = check_test_files()
    
    if not models_exist:
        print("\n" + "=" * 60)
        print_error("Model files are missing!")
        print_info("Please save the artifact contents to the following files:")
        print("  1. src/models/options/black_scholes.py")
        print("  2. src/models/options/greeks_calculator.py")
        print("  3. src/models/options/options_chain.py")
        print("\nThen run this script again.")
        return False
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check MongoDB
    mongo_ok = check_mongodb()
    
    # Run tests if files exist
    if models_exist and tests_exist:
        tests_passed = run_tests()
    else:
        tests_passed = False
        print_warning("Skipping tests - files missing")
    
    # Create sample script
    create_sample_script()
    
    # Final summary
    print_header("SETUP SUMMARY")
    
    status = {
        "Directory Structure": True,
        "Model Files": models_exist,
        "Test Files": tests_exist,
        "Dependencies": deps_ok,
        "MongoDB": mongo_ok,
        "Tests": tests_passed
    }
    
    all_good = all(status.values())
    
    for item, success in status.items():
        if success:
            print_success(item)
        else:
            print_error(item)
    
    if all_good:
        print("\n" + "=" * 60)
        print(f"{Colors.GREEN}🎉 ALL COMPONENTS READY! 🎉{Colors.END}")
        print("=" * 60)
        
        print("\n📋 NEXT STEPS:")
        print("1. Process your 582 BTC options:")
        print("   from src.data.collectors.deribit_collector import DeribitCollector")
        print("   from src.models.options.options_chain import OptionsChainProcessor")
        print("   ")
        print("   collector = DeribitCollector()")
        print("   processor = OptionsChainProcessor()")
        print("   btc_options = collector.get_options_data('BTC')")
        print("   chain = processor.process_deribit_chain(btc_options)")
        print("")
        print("2. Store in MongoDB:")
        print("   from src.core.database.operations import DatabaseOperations")
        print("   db_ops = DatabaseOperations()")
        print("   db_ops.store_options_data(chain.to_dict('records'))")
        print("")
        print("3. Create Reflex dashboard pages")
        
        return True
    else:
        print("\n" + "=" * 60)
        print(f"{Colors.YELLOW}⚠️ SETUP INCOMPLETE{Colors.END}")
        print("Please address the issues above and run again.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
    