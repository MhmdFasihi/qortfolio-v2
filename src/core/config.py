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
from typing import Dict, Any, Optional
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
    
    @property
    def connection_string(self) -> str:
        """Generate MongoDB connection string."""
        return f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?authSource=admin"

@dataclass
class APIConfig:
    """API configuration for external services."""
    deribit_client_id: str
    deribit_client_secret: str
    yfinance_rate_limit: int
    deribit_rate_limit: int
    data_update_interval: int

class Config:
    """Main configuration class for Qortfolio V2."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.config_dir = self.project_root / "config"
        
        # Load configurations
        self.database = self._load_database_config()
        self.api = self._load_api_config()
        self.crypto_mapping = self._load_crypto_mapping()
        self.app_settings = self._load_app_settings()
        
    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration from environment variables."""
        return DatabaseConfig(
            host=os.getenv("MONGO_HOST", "localhost"),
            port=int(os.getenv("MONGO_PORT", 27017)),
            username=os.getenv("MONGO_USER", "admin"),
            password=os.getenv("MONGO_PASSWORD", "password123"),
            database=os.getenv("MONGO_DATABASE", "qortfolio")
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
        """Load cryptocurrency ticker mappings."""
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
            "environment": os.getenv("ENVIRONMENT", "development"),
            "debug": os.getenv("DEBUG", "true").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "reflex_host": os.getenv("REFLEX_HOST", "0.0.0.0"),
            "reflex_port": int(os.getenv("REFLEX_PORT", 3000)),
            "max_portfolio_size": int(os.getenv("MAX_PORTFOLIO_SIZE", 100)),
            "default_risk_free_rate": float(os.getenv("DEFAULT_RISK_FREE_RATE", 0.05)),
            "default_confidence_level": float(os.getenv("DEFAULT_CONFIDENCE_LEVEL", 0.95)),
        }
    
    def get_yfinance_ticker(self, symbol: str) -> str:
        """Get yfinance ticker for a given crypto symbol."""
        return self.crypto_mapping.get(symbol, f"{symbol}-USD")

# Global config instance
config = Config()

if __name__ == "__main__":
    # Test configuration loading
    print("Database Config:", config.database.connection_string)
    print("API Config:", config.api.__dict__)
    print("Crypto Mappings:", config.crypto_mapping)
    print("App Settings:", config.app_settings)
