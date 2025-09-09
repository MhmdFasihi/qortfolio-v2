# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Configuration management module for Qortfolio V2.
Handles environment variables, API settings, and crypto mappings.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """MongoDB database configuration."""
    host: str
    port: int
    username: str
    password: str
    database: str
    auth_source: str
    uri: Optional[str] = None
    # Pooling / timeouts
    max_pool_size: int = 50
    min_pool_size: int = 10
    server_selection_timeout_ms: int = 5000
    connect_timeout_ms: int = 10000
    socket_timeout_ms: int = 10000
    wait_queue_timeout_ms: int = 5000
    heartbeat_frequency_ms: int = 10000
    
    @property
    def connection_string(self) -> str:
        """Generate MongoDB connection string.

        Priority:
        1) If `uri` (env `MONGODB_URL`) is set, return it as-is.
        2) If username/password are provided, include them and use `auth_source`.
        3) Otherwise build a no-auth local URI.
        """
        if self.uri:
            return self.uri

        user = (self.username or "").strip()
        pwd = (self.password or "").strip()
        if user and pwd:
            return (
                f"mongodb://{user}:{pwd}@{self.host}:{self.port}/"
                f"{self.database}?authSource={self.auth_source}"
            )
        # No credentials -> no-auth local MongoDB
        return f"mongodb://{self.host}:{self.port}/{self.database}"

@dataclass
class APIConfig:
    """API configuration for external services."""
    deribit_client_id: str
    deribit_client_secret: str
    yfinance_rate_limit: int
    deribit_rate_limit: int
    data_update_interval: int
    
    @property
    def deribit_testnet_url(self) -> str:
        """Deribit testnet URL."""
        return "wss://test.deribit.com/ws/api/v2"
    
    @property
    def deribit_mainnet_url(self) -> str:
        """Deribit mainnet URL."""
        return "wss://www.deribit.com/ws/api/v2"
    
    @property
    def deribit_rest_url(self) -> str:
        """Deribit REST API URL."""
        env = os.getenv("ENVIRONMENT", "development")
        if env == "production":
            return "https://www.deribit.com/api/v2"
        return "https://test.deribit.com/api/v2"

@dataclass
class RedisConfig:
    """Redis cache configuration."""
    host: str
    port: int
    password: str
    db: int = 0
    
    @property
    def connection_string(self) -> str:
        """Generate Redis connection string."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

class Config:
    """Main configuration class for Qortfolio V2."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.config_dir = self.project_root / "config"
        
        # Load configurations
        self.database = self._load_database_config()
        self.redis = self._load_redis_config()
        self.api = self._load_api_config()
        self.app_settings = self._load_app_settings()
        
        # Load crypto sectors configuration
        try:
            from src.core.crypto.crypto_sectors import crypto_sectors
            self.crypto_sectors = crypto_sectors
            logger.info("✅ Crypto sectors configuration loaded")
        except ImportError as e:
            logger.error(f"❌ Failed to load crypto sectors: {e}")
            raise ImportError(f"Crypto sectors module is required: {e}")

    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration from environment variables."""
        return DatabaseConfig(
            host=os.getenv("MONGO_HOST", "localhost"),
            port=int(os.getenv("MONGO_PORT", 27017)),
            username=os.getenv("MONGO_USER", "admin"),
            # Default to docker-compose default for consistency
            password=os.getenv("MONGO_PASSWORD", "secure_password_123"),
            database=os.getenv("MONGO_DATABASE", "qortfolio"),
            auth_source=os.getenv("MONGO_AUTH_SOURCE", "admin"),
            uri=os.getenv("MONGODB_URL"),
            max_pool_size=int(os.getenv("MONGO_MAX_POOL_SIZE", 50)),
            min_pool_size=int(os.getenv("MONGO_MIN_POOL_SIZE", 10)),
            server_selection_timeout_ms=int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", 5000)),
            connect_timeout_ms=int(os.getenv("MONGO_CONNECT_TIMEOUT_MS", 10000)),
            socket_timeout_ms=int(os.getenv("MONGO_SOCKET_TIMEOUT_MS", 10000)),
            wait_queue_timeout_ms=int(os.getenv("MONGO_WAIT_QUEUE_TIMEOUT_MS", 5000)),
            heartbeat_frequency_ms=int(os.getenv("MONGO_HEARTBEAT_FREQUENCY_MS", 10000)),
        )
    
    def _load_redis_config(self) -> RedisConfig:
        """Load Redis configuration from environment variables."""
        return RedisConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD", ""),
            db=int(os.getenv("REDIS_DB", 0))
        )
    
    def _load_api_config(self) -> APIConfig:
        """Load API configuration from environment variables."""
        return APIConfig(
            deribit_client_id=os.getenv("DERIBIT_CLIENT_ID", ""),
            deribit_client_secret=os.getenv("DERIBIT_CLIENT_SECRET", ""),
            yfinance_rate_limit=int(os.getenv("YFINANCE_RATE_LIMIT", 100)),
            deribit_rate_limit=int(os.getenv("DERIBIT_RATE_LIMIT", 50)),
            data_update_interval=int(os.getenv("DATA_UPDATE_INTERVAL", 60))
        )
    
    def _load_crypto_mapping(self) -> Dict[str, str]:
        """Load cryptocurrency ticker mappings (legacy method)."""
        # Use crypto_sectors if available
        if getattr(self, "crypto_sectors", None):
            return self.crypto_sectors.ticker_mapping
        
        # Otherwise load from file or use defaults
        mapping_file = self.config_dir / "crypto_mapping.yaml"
        
        # Default mappings if file doesn't exist
        default_mappings = {
            "BTC": "BTC-USD",
            "ETH": "ETH-USD",
            "SOL": "SOL-USD",
            "LINK": "LINK-USD",
            "AVAX": "AVAX-USD",
            "UNI": "UNI7083-USD",
            "AAVE": "AAVE-USD",
        }
        
        if mapping_file.exists():
            try:
                with open(mapping_file, 'r') as f:
                    return yaml.safe_load(f) or default_mappings
            except Exception as e:
                logger.warning(f"Failed to load crypto mapping: {e}")
                return default_mappings
        else:
            # Create the file with defaults
            self.config_dir.mkdir(exist_ok=True)
            with open(mapping_file, 'w') as f:
                yaml.dump(default_mappings, f, default_flow_style=False)
            return default_mappings
    
    def _load_app_settings(self) -> Dict[str, Any]:
        """Load application settings."""
        return {
            # Environment
            "environment": os.getenv("ENVIRONMENT", "development"),
            "debug": os.getenv("DEBUG", "true").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            
            # Reflex settings
            "reflex_host": os.getenv("REFLEX_HOST", "0.0.0.0"),
            "reflex_port": int(os.getenv("REFLEX_PORT", 3000)),
            
            # Portfolio settings
            "max_portfolio_size": int(os.getenv("MAX_PORTFOLIO_SIZE", 100)),
            "default_risk_free_rate": float(os.getenv("DEFAULT_RISK_FREE_RATE", 0.05)),
            "default_confidence_level": float(os.getenv("DEFAULT_CONFIDENCE_LEVEL", 0.95)),
            
            # Data settings
            "default_lookback_days": int(os.getenv("DEFAULT_LOOKBACK_DAYS", 365)),
            "cache_ttl_seconds": int(os.getenv("CACHE_TTL_SECONDS", 300)),
            
            # Options settings
            "options_chain_depth": int(os.getenv("OPTIONS_CHAIN_DEPTH", 10)),
            "min_option_volume": int(os.getenv("MIN_OPTION_VOLUME", 1)),
            "max_option_spread_pct": float(os.getenv("MAX_OPTION_SPREAD_PCT", 0.1)),
            
            # Performance settings
            "enable_caching": os.getenv("ENABLE_CACHING", "true").lower() == "true",
            "enable_monitoring": os.getenv("ENABLE_MONITORING", "true").lower() == "true",
        }
    
    # === Public Methods ===
    
    def get_yfinance_ticker(self, symbol: str) -> str:
        """
        Get yfinance ticker for a given crypto symbol.
        Uses crypto_sectors if available, otherwise falls back to mapping.
        """
        if getattr(self, "crypto_sectors", None):
            return self.crypto_sectors.get_yfinance_ticker(symbol)
        return self._load_crypto_mapping().get(symbol, f"{symbol}-USD")
    
    def get_all_sectors(self) -> List[str]:
        """Get list of all available sectors."""
        if getattr(self, "crypto_sectors", None):
            return self.crypto_sectors.get_all_sectors()
        return ["DeFi", "Infrastructure", "AI", "Digital Assets", "Privacy", "Services", "DApps"]
    
    def get_sector_tickers(self, sector: str) -> List[str]:
        """Get tickers for a specific sector."""
        if getattr(self, "crypto_sectors", None):
            return self.crypto_sectors.get_sector_tickers(sector)
        return []
    
    def validate_ticker(self, ticker: str) -> bool:
        """Validate if a ticker exists in our configuration."""
        if getattr(self, "crypto_sectors", None):
            is_valid, _ = self.crypto_sectors.validate_ticker(ticker)
            return is_valid
        return ticker in self._load_crypto_mapping()
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_settings["environment"] == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app_settings["environment"] == "development"
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration."""
        return {
            "enabled": self.app_settings["enable_caching"],
            "ttl": self.app_settings["cache_ttl_seconds"],
            "redis": self.redis.connection_string if self.app_settings["enable_caching"] else None
        }
    
    def get_data_sources_config(self) -> Dict[str, Any]:
        """Get configuration for all data sources."""
        return {
            "yfinance": {
                "rate_limit": self.api.yfinance_rate_limit,
                "enabled": True
            },
            "deribit": {
                "rate_limit": self.api.deribit_rate_limit,
                "url": self.api.deribit_rest_url,
                "ws_url": self.api.deribit_mainnet_url if self.is_production() else self.api.deribit_testnet_url,
                "client_id": self.api.deribit_client_id,
                "enabled": bool(self.api.deribit_client_id)
            }
        }
    
    def get_options_config(self) -> Dict[str, Any]:
        """Get options-specific configuration."""
        return {
            "chain_depth": self.app_settings["options_chain_depth"],
            "min_volume": self.app_settings["min_option_volume"],
            "max_spread_pct": self.app_settings["max_option_spread_pct"],
            "currencies": ["BTC", "ETH"],  # Supported option currencies
            "update_interval": self.api.data_update_interval
        }
    
    def get_portfolio_config(self) -> Dict[str, Any]:
        """Get portfolio management configuration."""
        return {
            "max_size": self.app_settings["max_portfolio_size"],
            "risk_free_rate": self.app_settings["default_risk_free_rate"],
            "confidence_level": self.app_settings["default_confidence_level"],
            "lookback_days": self.app_settings["default_lookback_days"],
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform configuration health check."""
        health = {
            "config_loaded": True,
            "environment": self.app_settings["environment"],
            "database_configured": bool(self.database.username),
            "redis_configured": bool(self.redis.host),
            "deribit_configured": bool(self.api.deribit_client_id),
            "crypto_sectors_loaded": getattr(self, "crypto_sectors", None) is not None,
        }
        
        if getattr(self, "crypto_sectors", None):
            health["total_sectors"] = len(self.get_all_sectors())
            health["total_tickers"] = len(self.crypto_sectors.get_all_tickers())
        
        return health
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(env={self.app_settings['environment']}, db={self.database.database})"
    
    def __repr__(self) -> str:
        """Detailed representation of configuration."""
        return (
            f"Config("
            f"environment={self.app_settings['environment']}, "
            f"database={self.database.database}, "
            f"debug={self.app_settings['debug']}, "
            f"sectors_loaded={getattr(self, 'crypto_sectors', None) is not None}"
            f")"
        )

# Global config instance
config = Config()

if __name__ == "__main__":
    # Test configuration loading
    print("\n🔧 Qortfolio V2 Configuration")
    print("=" * 50)
    
    # Basic settings
    print(f"\n📊 Environment: {config.app_settings['environment']}")
    print(f"📊 Debug Mode: {config.app_settings['debug']}")
    print(f"📊 Log Level: {config.app_settings['log_level']}")
    
    # Database
    print(f"\n💾 Database: {config.database.database}")
    print(f"💾 MongoDB Host: {config.database.host}:{config.database.port}")
    
    # Redis
    print(f"\n📦 Redis Cache: {config.redis.host}:{config.redis.port}")
    print(f"📦 Cache Enabled: {config.app_settings['enable_caching']}")
    
    # API Configuration
    print(f"\n🌐 Deribit Configured: {bool(config.api.deribit_client_id)}")
    print(f"🌐 Deribit URL: {config.api.deribit_rest_url}")
    
    # Crypto Sectors
    if getattr(config, "crypto_sectors", None):
        print(f"\n📈 Crypto Sectors Loaded: ✅")
        print(f"📈 Total Sectors: {len(config.get_all_sectors())}")
        print(f"📈 Available Sectors: {', '.join(config.get_all_sectors())}")
        
        # Test ticker mapping
        test_tickers = ["BTC", "ETH", "UNI", "RENDER"]
        print("\n📊 Sample Ticker Mappings:")
        for ticker in test_tickers:
            yf_ticker = config.get_yfinance_ticker(ticker)
            print(f"   {ticker} -> {yf_ticker}")
    else:
        print(f"\n📈 Crypto Sectors: Not loaded")
    
    # Health check
    print("\n🏥 Health Check:")
    health = config.health_check()
    for key, value in health.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key}: {value}")
