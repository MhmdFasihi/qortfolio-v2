# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""

Tests for configuration management system
"""

import pytest
import tempfile
import os
from pathlib import Path
import yaml
import sys

# Add src to path for testing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_path)

from core.config import ConfigManager, CryptoCurrency, ConfigurationError


class TestConfigManager:
    """Test the configuration manager."""
    
    def test_basic_initialization(self):
        """Test basic configuration manager initialization."""
        # This should work even without config files
        try:
            config = ConfigManager(config_dir="nonexistent")
            assert config is not None
        except ConfigurationError:
            # Expected if no config files exist
            pass
    
    def test_crypto_mappings_structure(self):
        """Test cryptocurrency mappings structure."""
        # Create temporary config
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir)
            
            # Create sample crypto mapping
            crypto_config = {
                "crypto_mappings": {
                    "bitcoin": {
                        "name": "Bitcoin",
                        "symbol": "BTC",
                        "yfinance_ticker": "BTC-USD",
                        "deribit_currency": "BTC",
                        "enabled": True,
                        "priority": 1
                    }
                }
            }
            
            # Write config file
            with open(config_path / "crypto_mapping.yaml", 'w') as f:
                yaml.dump(crypto_config, f)
            
            # Test loading
            config = ConfigManager(config_dir=str(config_path))
            
            # Test access
            cryptos = config.crypto_mappings
            assert "bitcoin" in cryptos
            
            btc = cryptos["bitcoin"]
            assert isinstance(btc, CryptoCurrency)
            assert btc.symbol == "BTC"
            assert btc.yfinance_ticker == "BTC-USD"
            assert btc.enabled is True
    
    def test_enabled_cryptocurrencies(self):
        """Test filtering of enabled cryptocurrencies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir)
            
            crypto_config = {
                "crypto_mappings": {
                    "bitcoin": {
                        "name": "Bitcoin", "symbol": "BTC", "yfinance_ticker": "BTC-USD",
                        "deribit_currency": "BTC", "enabled": True, "priority": 1
                    },
                    "ethereum": {
                        "name": "Ethereum", "symbol": "ETH", "yfinance_ticker": "ETH-USD", 
                        "deribit_currency": "ETH", "enabled": False, "priority": 2
                    }
                }
            }
            
            with open(config_path / "crypto_mapping.yaml", 'w') as f:
                yaml.dump(crypto_config, f)
            
            config = ConfigManager(config_dir=str(config_path))
            
            enabled = config.enabled_cryptocurrencies
            assert len(enabled) == 1
            assert enabled[0].symbol == "BTC"
    
    def test_deribit_currencies(self):
        """Test Deribit currency filtering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir)
            
            crypto_config = {
                "crypto_mappings": {
                    "bitcoin": {
                        "name": "Bitcoin", "symbol": "BTC", "yfinance_ticker": "BTC-USD",
                        "deribit_currency": "BTC", "enabled": True, "priority": 1
                    },
                    "ripple": {
                        "name": "XRP", "symbol": "XRP", "yfinance_ticker": "XRP-USD",
                        "deribit_currency": None, "enabled": True, "priority": 2
                    }
                }
            }
            
            with open(config_path / "crypto_mapping.yaml", 'w') as f:
                yaml.dump(crypto_config, f)
            
            config = ConfigManager(config_dir=str(config_path))
            
            deribit_currencies = config.deribit_currencies
            assert "BTC" in deribit_currencies
            assert len(deribit_currencies) == 1  # XRP should be excluded (None currency)
    
    def test_dot_notation_access(self):
        """Test configuration access using dot notation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir)
            
            api_config = {
                "deribit_api": {
                    "base_url": "https://test.deribit.com",
                    "rate_limits": {
                        "requests_per_second": 10
                    }
                }
            }
            
            with open(config_path / "api_config.yaml", 'w') as f:
                yaml.dump(api_config, f)
            
            config = ConfigManager(config_dir=str(config_path))
            
            # Test nested access
            assert config.get("deribit_api.base_url") == "https://test.deribit.com"
            assert config.get("deribit_api.rate_limits.requests_per_second") == 10
            
            # Test default values
            assert config.get("nonexistent.key", "default") == "default"
    
    def test_yfinance_ticker_lookup(self):
        """Test yfinance ticker lookup by symbol."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir)
            
            crypto_config = {
                "crypto_mappings": {
                    "bitcoin": {
                        "name": "Bitcoin", "symbol": "BTC", "yfinance_ticker": "BTC-USD",
                        "deribit_currency": "BTC", "enabled": True, "priority": 1
                    }
                }
            }
            
            with open(config_path / "crypto_mapping.yaml", 'w') as f:
                yaml.dump(crypto_config, f)
            
            config = ConfigManager(config_dir=str(config_path))
            
            # Test ticker lookup
            ticker = config.get_yfinance_ticker("BTC")
            assert ticker == "BTC-USD"
            
            # Test case insensitive
            ticker = config.get_yfinance_ticker("btc")
            assert ticker == "BTC-USD"
            
            # Test nonexistent
            ticker = config.get_yfinance_ticker("NONEXISTENT")
            assert ticker is None


class TestConfigurationIntegration:
    """Test configuration integration."""
    
    def test_config_summary(self):
        """Test configuration summary generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir)
            
            # Create minimal config
            crypto_config = {"crypto_mappings": {}}
            with open(config_path / "crypto_mapping.yaml", 'w') as f:
                yaml.dump(crypto_config, f)
            
            config = ConfigManager(config_dir=str(config_path))
            summary = config.get_config_summary()
            
            assert "config_dir" in summary
            assert "loaded_files" in summary
            assert "enabled_cryptos" in summary
            assert "loaded_at" in summary
    
    def test_development_mode_detection(self):
        """Test development mode detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir)
            
            # Test environment variable
            original_debug = os.environ.get("QORTFOLIO_DEBUG")
            os.environ["QORTFOLIO_DEBUG"] = "true"
            
            try:
                config = ConfigManager(config_dir=str(config_path))
                assert config.is_development_mode() is True
            finally:
                # Cleanup
                if original_debug is None:
                    os.environ.pop("QORTFOLIO_DEBUG", None)
                else:
                    os.environ["QORTFOLIO_DEBUG"] = original_debug


def test_configuration_files_exist():
    """Test that configuration files exist in the project."""
    project_root = Path(__file__).parent.parent
    config_dir = project_root / "config"
    
    # Check if config directory exists
    if config_dir.exists():
        # Check for expected files
        expected_files = ["crypto_mapping.yaml", "api_config.yaml"]
        for filename in expected_files:
            file_path = config_dir / filename
            if file_path.exists():
                # Verify file can be loaded
                with open(file_path, 'r') as f:
                    content = yaml.safe_load(f)
                    assert content is not None, f"{filename} should contain valid YAML"


if __name__ == "__main__":
    # Run basic tests
    print("🧪 Testing Configuration System")
    print("=" * 40)
    
    # Test configuration files exist
    try:
        test_configuration_files_exist()
        print("✅ Configuration files validation passed")
    except Exception as e:
        print(f"⚠️ Configuration files issue: {e}")
    
    # Test basic functionality
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir)
            
            # Create test config
            test_config = {
                "crypto_mappings": {
                    "bitcoin": {
                        "name": "Bitcoin", "symbol": "BTC", "yfinance_ticker": "BTC-USD",
                        "deribit_currency": "BTC", "enabled": True, "priority": 1
                    }
                }
            }
            
            with open(config_path / "crypto_mapping.yaml", 'w') as f:
                yaml.dump(test_config, f)
            
            config = ConfigManager(config_dir=str(config_path))
            
            # Test basic functionality
            assert len(config.crypto_mappings) == 1
            assert config.get_yfinance_ticker("BTC") == "BTC-USD"
            
        print("✅ Configuration manager basic tests passed")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        raise
    
    print("\n🎉 Configuration system tests completed!")