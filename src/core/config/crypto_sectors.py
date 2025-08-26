# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Cryptocurrency sectors configuration management.
Handles sector classifications and ticker mappings for portfolio analysis.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SectorInfo:
    """Information about a crypto sector."""
    name: str
    tickers: List[str]
    yfinance_tickers: List[str]
    
    @property
    def size(self) -> int:
        """Number of assets in the sector."""
        return len(self.tickers)
    
    def get_ticker_mapping(self) -> Dict[str, str]:
        """Get ticker to yfinance mapping for this sector."""
        return dict(zip(self.tickers, self.yfinance_tickers))

class CryptoSectorsManager:
    """
    Manages cryptocurrency sector classifications and mappings.
    Integrates with crypto_sectors.json and provides utility methods.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the crypto sectors manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "config"
        
        self.config_dir = config_dir
        self.sectors_file = config_dir / "crypto_sectors.json"
        self.mapping_file = config_dir / "crypto_mapping.yaml"
        
        # Load data
        self.sectors_data = self._load_sectors_data()
        self.ticker_mapping = self._load_ticker_mapping()
        
        # Create inverse mappings
        self.ticker_to_sector = self._create_ticker_to_sector_map()
        self.yfinance_to_ticker = self._create_inverse_mapping()
        
        # Cache for performance
        self._all_tickers: Optional[Set[str]] = None
        self._all_sectors: Optional[List[str]] = None
        
    def _load_sectors_data(self) -> Dict[str, Dict]:
        """Load sectors data from JSON file."""
        try:
            with open(self.sectors_file, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} sectors from {self.sectors_file}")
                return data
        except FileNotFoundError:
            logger.error(f"Sectors file not found: {self.sectors_file}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing sectors JSON: {e}")
            return {}
    
    def _load_ticker_mapping(self) -> Dict[str, str]:
        """Load ticker to yfinance mapping from YAML."""
        try:
            with open(self.mapping_file, 'r') as f:
                mapping = yaml.safe_load(f) or {}
                logger.info(f"Loaded {len(mapping)} ticker mappings")
                return mapping
        except FileNotFoundError:
            logger.warning(f"Mapping file not found: {self.mapping_file}")
            # Generate from sectors data
            return self._generate_ticker_mapping()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing mapping YAML: {e}")
            return self._generate_ticker_mapping()
    
    def _generate_ticker_mapping(self) -> Dict[str, str]:
        """Generate ticker mapping from sectors data."""
        mapping = {}
        for sector_data in self.sectors_data.values():
            tickers = sector_data.get('tickers', [])
            yfinance = sector_data.get('yfinance', [])
            
            for ticker, yf_ticker in zip(tickers, yfinance):
                # Add -USD suffix for yfinance
                mapping[ticker] = f"{yf_ticker}-USD"
        
        # Add major cryptocurrencies if not present
        if 'BTC' not in mapping:
            mapping['BTC'] = 'BTC-USD'
        if 'ETH' not in mapping:
            mapping['ETH'] = 'ETH-USD'
            
        return mapping
    
    def _create_ticker_to_sector_map(self) -> Dict[str, str]:
        """Create mapping from ticker to sector name."""
        ticker_to_sector = {}
        
        for sector_name, sector_data in self.sectors_data.items():
            for ticker in sector_data.get('tickers', []):
                ticker_to_sector[ticker] = sector_name
        
        return ticker_to_sector
    
    def _create_inverse_mapping(self) -> Dict[str, str]:
        """Create yfinance to ticker mapping."""
        return {v: k for k, v in self.ticker_mapping.items()}
    
    # === Public Methods ===
    
    def get_all_sectors(self) -> List[str]:
        """Get list of all sector names."""
        if self._all_sectors is None:
            self._all_sectors = list(self.sectors_data.keys())
        return self._all_sectors
    
    def get_all_tickers(self) -> Set[str]:
        """Get set of all unique tickers across all sectors."""
        if self._all_tickers is None:
            tickers = set()
            for sector_data in self.sectors_data.values():
                tickers.update(sector_data.get('tickers', []))
            self._all_tickers = tickers
        return self._all_tickers
    
    def get_sector_info(self, sector_name: str) -> Optional[SectorInfo]:
        """
        Get detailed information about a specific sector.
        
        Args:
            sector_name: Name of the sector
            
        Returns:
            SectorInfo object or None if sector not found
        """
        if sector_name not in self.sectors_data:
            return None
        
        sector_data = self.sectors_data[sector_name]
        return SectorInfo(
            name=sector_name,
            tickers=sector_data.get('tickers', []),
            yfinance_tickers=sector_data.get('yfinance', [])
        )
    
    def get_sector_tickers(self, sector_name: str) -> List[str]:
        """Get list of tickers for a specific sector."""
        if sector_name in self.sectors_data:
            return self.sectors_data[sector_name].get('tickers', [])
        return []
    
    def get_ticker_sector(self, ticker: str) -> Optional[str]:
        """Get sector name for a specific ticker."""
        return self.ticker_to_sector.get(ticker)
    
    def get_yfinance_ticker(self, ticker: str) -> str:
        """
        Get yfinance ticker for a given crypto symbol.
        
        Args:
            ticker: Crypto ticker symbol
            
        Returns:
            yfinance ticker symbol
        """
        return self.ticker_mapping.get(ticker, f"{ticker}-USD")
    
    def get_ticker_from_yfinance(self, yf_ticker: str) -> Optional[str]:
        """Convert yfinance ticker back to standard ticker."""
        # Remove -USD suffix if present
        if yf_ticker.endswith('-USD'):
            yf_ticker = yf_ticker[:-4]
        
        # Check inverse mapping
        if yf_ticker in self.yfinance_to_ticker:
            return self.yfinance_to_ticker[yf_ticker]
        
        # Check if it's a direct match
        if yf_ticker in self.get_all_tickers():
            return yf_ticker
        
        return None
    
    def get_sectors_by_tickers(self, tickers: List[str]) -> Dict[str, List[str]]:
        """
        Group tickers by their sectors.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping sector names to lists of tickers
        """
        sectors_dict = {}
        
        for ticker in tickers:
            sector = self.get_ticker_sector(ticker)
            if sector:
                if sector not in sectors_dict:
                    sectors_dict[sector] = []
                sectors_dict[sector].append(ticker)
        
        return sectors_dict
    
    def validate_ticker(self, ticker: str) -> Tuple[bool, str]:
        """
        Validate if a ticker exists in our database.
        
        Args:
            ticker: Ticker to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        if ticker in self.get_all_tickers():
            sector = self.get_ticker_sector(ticker)
            return True, f"Valid ticker in {sector} sector"
        return False, f"Unknown ticker: {ticker}"
    
    def get_sector_statistics(self) -> Dict[str, Dict]:
        """Get statistics about all sectors."""
        stats = {}
        
        for sector_name in self.get_all_sectors():
            info = self.get_sector_info(sector_name)
            if info:
                stats[sector_name] = {
                    'count': info.size,
                    'tickers': info.tickers[:5],  # First 5 as sample
                    'has_yfinance_mapping': len(info.yfinance_tickers) == len(info.tickers)
                }
        
        return stats
    
    def export_full_mapping(self, output_file: Optional[Path] = None) -> Dict[str, str]:
        """
        Export complete ticker to yfinance mapping.
        
        Args:
            output_file: Optional file path to save mapping
            
        Returns:
            Complete mapping dictionary
        """
        mapping = self.ticker_mapping.copy()
        
        if output_file:
            with open(output_file, 'w') as f:
                yaml.dump(mapping, f, default_flow_style=False, sort_keys=True)
                logger.info(f"Exported {len(mapping)} mappings to {output_file}")
        
        return mapping

# Global instance
crypto_sectors = CryptoSectorsManager()

if __name__ == "__main__":
    # Test the crypto sectors manager
    manager = CryptoSectorsManager()
    
    print("\n📊 Crypto Sectors Configuration")
    print("=" * 50)
    
    # Show all sectors
    sectors = manager.get_all_sectors()
    print(f"\nTotal Sectors: {len(sectors)}")
    for sector in sectors:
        info = manager.get_sector_info(sector)
        print(f"  - {sector}: {info.size} assets")
    
    # Show total tickers
    all_tickers = manager.get_all_tickers()
    print(f"\nTotal Unique Tickers: {len(all_tickers)}")
    
    # Test ticker lookup
    test_tickers = ["BTC", "ETH", "UNI", "AAVE", "RENDER"]
    print("\nTicker Mappings:")
    for ticker in test_tickers:
        yf_ticker = manager.get_yfinance_ticker(ticker)
        sector = manager.get_ticker_sector(ticker)
        print(f"  {ticker} -> {yf_ticker} (Sector: {sector})")
    
    # Show sector statistics
    stats = manager.get_sector_statistics()
    print("\nSector Statistics:")
    for sector, stat in stats.items():
        print(f"  {sector}: {stat['count']} assets")
