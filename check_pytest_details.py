# Quick check to see what config tests expect
import sys
from pathlib import Path

current_dir = Path(__file__).parent  
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

def check_config_expectations():
    """Check what the config tests expect."""
    print("🔍 Checking Config Test Expectations...")
    
    try:
        from core.config import get_config, reset_config
        
        # Reset and get fresh config
        reset_config()
        config = get_config()
        
        print(f"Config type: {type(config)}")
        print(f"Has deribit_currencies: {hasattr(config, 'deribit_currencies')}")
        print(f"deribit_currencies value: {config.deribit_currencies}")
        print(f"deribit_currencies type: {type(config.deribit_currencies)}")
        
        # Test dot notation
        try:
            base_url = config.get('deribit_api.base_url')
            print(f"deribit_api.base_url: {base_url}")
        except Exception as e:
            print(f"dot notation error: {e}")
        
        # Test yfinance ticker
        try:
            btc_ticker = config.get_yfinance_ticker('BTC')
            print(f"BTC ticker: {btc_ticker}")
        except Exception as e:
            print(f"yfinance ticker error: {e}")
            
        # Test summary
        try:
            summary = config.get_config_summary()
            print(f"Config summary: {summary}")
        except Exception as e:
            print(f"config summary error: {e}")
            
        # Test development mode
        try:
            dev_mode = config.is_development_mode()
            print(f"Development mode: {dev_mode}")
        except Exception as e:
            print(f"development mode error: {e}")
            
    except Exception as e:
        print(f"Overall error: {e}")

if __name__ == "__main__":
    check_config_expectations()