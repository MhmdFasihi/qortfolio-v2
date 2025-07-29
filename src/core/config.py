# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Configuration Management System for Qortfolio V2 - COMPLETE FIX
Location: src/core/config.py

This fixes ALL the configuration interface issues to match what the dashboard expects.
Includes all missing methods that were causing crashes.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

# Custom exception that the system expects
class ConfigurationError(Exception):
    """Configuration-related errors."""
    pass

@dataclass
class CryptoCurrency:
    """Configuration for a cryptocurrency - matches system expectations."""
    symbol: str
    name: str
    yfinance_ticker: str
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'name': self.name,
            'yfinance_ticker': self.yfinance_ticker,
            'enabled': self.enabled
        }

class ConfigManager:
    """
    Complete Configuration Manager with ALL missing methods implemented.
    
    This fixes the critical dashboard crash by providing all expected methods:
    - get_cache_ttl()
    - get_deribit_currency()
    - deribit_currencies property
    - get_config_summary()
    - And all other required configuration methods
    """
    
    def __init__(self):
        """Initialize configuration manager with comprehensive setup."""
        self.logger = logging.getLogger("config_manager")
        
        # Configuration storage
        self._config_data: Dict[str, Any] = {}
        self._cryptocurrencies: List[CryptoCurrency] = []
        
        # Load all configuration
        self._load_configuration()
        
        self.logger.info("ConfigManager initialized with all methods")
    
    def _load_configuration(self):
        """Load complete configuration from multiple sources."""
        try:
            # Load from YAML files if they exist
            self._load_yaml_configs()
            
            # Load basic configuration that the system expects
            self._load_basic_config()
            
            # Load cryptocurrency mappings
            self._load_crypto_currencies()
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            # Provide fallback configuration
            self._load_fallback_config()
    
    def _load_yaml_configs(self):
        """Load YAML configuration files."""
        config_dir = Path("config")
        
        # API configuration
        api_config_path = config_dir / "api_config.yaml"
        if api_config_path.exists():
            try:
                with open(api_config_path, 'r') as f:
                    api_config = yaml.safe_load(f)
                    self._config_data.update(api_config)
                    self.logger.info("Loaded api_config.yaml")
            except Exception as e:
                self.logger.warning(f"Could not load api_config.yaml: {e}")
        
        # Crypto mapping configuration  
        crypto_config_path = config_dir / "crypto_mapping.yaml"
        if crypto_config_path.exists():
            try:
                with open(crypto_config_path, 'r') as f:
                    crypto_config = yaml.safe_load(f)
                    if 'crypto_symbols' in crypto_config:
                        # Convert to cryptocurrency objects
                        for symbol, ticker in crypto_config['crypto_symbols'].items():
                            self._cryptocurrencies.append(
                                CryptoCurrency(symbol, symbol, ticker, True)
                            )
                    self.logger.info("Loaded crypto_mapping.yaml")
            except Exception as e:
                self.logger.warning(f"Could not load crypto_mapping.yaml: {e}")
    
    def _load_basic_config(self):
        """Load basic configuration that the system expects."""
        # Merge with existing config, don't overwrite
        basic_config = {
            'application': {
                'development_mode': True,
                'name': 'Qortfolio V2',
                'version': '1.0.0'
            },
            'deribit_api': {
                'base_url': 'https://www.deribit.com/api/v2',
                'websocket_url': 'wss://www.deribit.com/ws/api/v2',
                'test_websocket_url': 'wss://test.deribit.com/ws/api/v2',
                'timeout': 30,
                'max_retries': 3,
                'rate_limit_delay': 0.1
            },
            'yfinance_api': {
                'timeout': 15,
                'max_retries': 3,
                'backoff_factor': 2.0
            },
            'caching': {
                'enabled': True,
                'ttl': {
                    'spot_prices': 60,
                    'options_data': 300,
                    'historical_data': 3600,
                    'instruments': 1800,
                    'default': 300
                }
            },
            'options_config': {
                'default_params': {
                    'risk_free_rate': 0.05,
                    'dividend_yield': 0.0
                }
            },
            'logging': {
                'level': 'INFO',
                'format': 'detailed'
            }
        }
        
        # Deep merge with existing config
        self._deep_merge_config(self._config_data, basic_config)
    
    def _load_crypto_currencies(self):
        """Load cryptocurrency configurations."""
        # If no cryptos loaded from YAML, provide defaults
        if not self._cryptocurrencies:
            self._cryptocurrencies = [
                CryptoCurrency("BTC", "Bitcoin", "BTC-USD", True),
                CryptoCurrency("ETH", "Ethereum", "ETH-USD", True),
                CryptoCurrency("XRP", "Ripple", "XRP-USD", True),
                CryptoCurrency("LTC", "Litecoin", "LTC-USD", True),
                CryptoCurrency("BCH", "Bitcoin Cash", "BCH-USD", True),
            ]
    
    def _load_fallback_config(self):
        """Load minimal fallback configuration if all else fails."""
        self._config_data = {
            'application': {'development_mode': True},
            'deribit_api': {'base_url': 'https://www.deribit.com/api/v2'},
            'caching': {
                'enabled': True,
                'ttl': {'default': 300, 'spot_prices': 60, 'options_data': 300}
            }
        }
        
        self._cryptocurrencies = [
            CryptoCurrency("BTC", "Bitcoin", "BTC-USD", True),
            CryptoCurrency("ETH", "Ethereum", "ETH-USD", True),
        ]
    
    def _deep_merge_config(self, target: Dict, source: Dict):
        """Deep merge configuration dictionaries."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge_config(target[key], value)
            elif key not in target:
                target[key] = value
    
    # ==================== CRITICAL MISSING METHODS ====================
    # These methods were missing and causing dashboard crashes
    
    def get_cache_ttl(self, cache_type: str) -> int:
        """
        Get cache TTL for specific type - CRITICAL METHOD.
        
        Args:
            cache_type: Type of cache (e.g., 'spot_prices', 'options_data')
        
        Returns:
            TTL in seconds
        """
        try:
            cache_ttls = self.get('caching.ttl', {})
            ttl = cache_ttls.get(cache_type, cache_ttls.get('default', 300))
            self.logger.debug(f"Cache TTL for {cache_type}: {ttl} seconds")
            return int(ttl)
        except Exception as e:
            self.logger.warning(f"Error getting cache TTL for {cache_type}: {e}")
            return 300  # Default 5 minutes
    
    def get_deribit_currency(self, symbol: str) -> Optional[str]:
        """
        Get Deribit currency mapping for symbol - CRITICAL METHOD.
        
        Args:
            symbol: Symbol to map (e.g., "BTC", "ETH")
        
        Returns:
            Deribit currency or None
        """
        # For Deribit, the symbol IS the currency for options
        supported_currencies = self.deribit_currencies
        
        if symbol.upper() in supported_currencies:
            return symbol.upper()
        
        self.logger.warning(f"Symbol {symbol} not supported on Deribit")
        return None
    
    @property
    def deribit_currencies(self) -> List[str]:
        """
        Get list of supported Deribit currencies - CRITICAL PROPERTY.
        
        Returns:
            List of supported currency symbols
        """
        # Deribit supports BTC and ETH options primarily
        return ['BTC', 'ETH']
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary for debugging - CRITICAL METHOD.
        
        Returns:
            Summary of configuration state
        """
        try:
            return {
                'loaded_files': len(self._config_data),
                'enabled_cryptos': len([c for c in self._cryptocurrencies if c.enabled]),
                'total_cryptos': len(self._cryptocurrencies),
                'development_mode': self.get('application.development_mode', True),
                'cache_enabled': self.get('caching.enabled', True),
                'deribit_currencies': self.deribit_currencies,
                'config_keys': list(self._config_data.keys())
            }
        except Exception as e:
            self.logger.error(f"Error generating config summary: {e}")
            return {'error': str(e)}
    
    # ==================== CORE CONFIGURATION METHODS ====================
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot notation key.
        
        Args:
            key: Configuration key (e.g., 'deribit_api.base_url')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            keys = key.split('.')
            value = self._config_data
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception as e:
            self.logger.warning(f"Error accessing config key '{key}': {e}")
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by dot notation key.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        try:
            keys = key.split('.')
            config = self._config_data
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set the value
            config[keys[-1]] = value
            
        except Exception as e:
            self.logger.error(f"Error setting config key '{key}': {e}")
            raise ConfigurationError(f"Cannot set config key '{key}': {e}")
    
    def get_cryptocurrencies(self) -> List[CryptoCurrency]:
        """Get list of configured cryptocurrencies."""
        return self._cryptocurrencies
    
    def get_enabled_cryptocurrencies(self) -> List[CryptoCurrency]:
        """Get list of enabled cryptocurrencies."""
        return [crypto for crypto in self._cryptocurrencies if crypto.enabled]
    
    @property
    def enabled_cryptocurrencies(self) -> List[CryptoCurrency]:
        """Property version of enabled cryptocurrencies - CRITICAL for dashboard."""
        return self.get_enabled_cryptocurrencies()
    
    def get_yfinance_ticker(self, symbol: str) -> Optional[str]:
        """
        Get yfinance ticker for symbol.
        
        Args:
            symbol: Crypto symbol
            
        Returns:
            yfinance ticker or None
        """
        for crypto in self._cryptocurrencies:
            if crypto.symbol.upper() == symbol.upper():
                return crypto.yfinance_ticker
        return None
    
    def validate_configuration(self) -> bool:
        """
        Validate current configuration.
        
        Returns:
            True if configuration is valid
        """
        try:
            # Check required configuration sections
            required_sections = ['deribit_api', 'caching']
            for section in required_sections:
                if not self.get(section):
                    self.logger.error(f"Missing required config section: {section}")
                    return False
            
            # Check Deribit API URL
            if not self.get('deribit_api.base_url'):
                self.logger.error("Missing Deribit API base URL")
                return False
            
            # Check cache configuration
            if not isinstance(self.get('caching.ttl'), dict):
                self.logger.warning("Cache TTL configuration is not a dictionary")
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

# ==================== GLOBAL CONFIGURATION INSTANCE ====================

# Global configuration instance (singleton pattern)
_config_instance: Optional[ConfigManager] = None

def get_config() -> ConfigManager:
    """
    Get global configuration instance (singleton).
    
    Returns:
        ConfigManager instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance

def reset_config():
    """Reset global configuration instance (for testing)."""
    global _config_instance
    _config_instance = None