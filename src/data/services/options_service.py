"""Options Data Service for MongoDB Operations"""

from typing import List, Dict, Optional
from datetime import datetime
from src.core.database.connection import db_connection
import logging

logger = logging.getLogger(__name__)

class OptionsDataService:
    """Service for managing options data in MongoDB"""
    
    def __init__(self):
        self.db = db_connection
        self.collection = self.db.get_collection('options_data')
    
    async def get_options_by_currency(self, currency: str = "BTC") -> List[Dict]:
        """Fetch options data for a specific currency"""
        try:
            if not self.collection:
                logger.warning("No MongoDB connection, returning sample data")
                return self._get_sample_data(currency)
            
            # Query MongoDB for options data
            query = {"underlying_currency": currency}
            cursor = self.collection.find(query).limit(100)
            
            # Convert cursor to list
            options = list(cursor)
            
            # If no data, return sample
            if not options:
                logger.info(f"No options data found for {currency}, using sample data")
                return self._get_sample_data(currency)
            
            # Format for display
            return self._format_options_data(options)
            
        except Exception as e:
            logger.error(f"Error fetching options data: {e}")
            return self._get_sample_data(currency)
    
    def _format_options_data(self, options: List[Dict]) -> List[Dict]:
        """Format MongoDB data for UI display"""
        formatted = []
        for opt in options:
            formatted.append({
                "strike": opt.get("strike_price", 0),
                "option_type": opt.get("option_type", ""),
                "expiry": str(opt.get("expiration_date", ""))[:10],
                "bid": f"{opt.get('bid', 0):.4f}",
                "ask": f"{opt.get('ask', 0):.4f}",
                "iv": f"{opt.get('implied_volatility', 0):.1%}",
                "volume": opt.get('volume', 0),
                "open_interest": opt.get('open_interest', 0),
                "delta": f"{opt.get('delta', 0):.3f}",
                "gamma": f"{opt.get('gamma', 0):.4f}",
                "theta": f"{opt.get('theta', 0):.4f}",
                "vega": f"{opt.get('vega', 0):.4f}",
            })
        return formatted
    
    def _get_sample_data(self, currency: str) -> List[Dict]:
        """Return sample data when DB is unavailable"""
        base_strike = 45000 if currency == "BTC" else 3000
        return [
            {
                "strike": base_strike,
                "option_type": "CALL",
                "expiry": "2024-09-27",
                "bid": "0.0234",
                "ask": "0.0245",
                "iv": "65.3%",
                "volume": 125,
                "open_interest": 890,
                "delta": "0.523",
                "gamma": "0.0012",
                "theta": "-0.0234",
                "vega": "0.1234",
            },
            {
                "strike": base_strike + 1000,
                "option_type": "PUT",
                "expiry": "2024-09-27",
                "bid": "0.0156",
                "ask": "0.0162",
                "iv": "62.1%",
                "volume": 89,
                "open_interest": 567,
                "delta": "-0.477",
                "gamma": "0.0011",
                "theta": "-0.0198",
                "vega": "0.1123",
            },
        ]
    
    async def get_metrics(self, currency: str = "BTC") -> Dict:
        """Calculate metrics for dashboard"""
        options = await self.get_options_by_currency(currency)
        
        if not options:
            return {
                "total_contracts": 0,
                "avg_iv": 0.0,
                "max_oi": 0,
                "total_volume": 0
            }
        
        # Calculate metrics
        ivs = []
        for opt in options:
            iv_str = opt.get('iv', '0%').replace('%', '')
            try:
                ivs.append(float(iv_str))
            except:
                pass
        
        avg_iv = sum(ivs) / len(ivs) / 100 if ivs else 0.0
        max_oi = max([opt.get('open_interest', 0) for opt in options], default=0)
        total_volume = sum([opt.get('volume', 0) for opt in options])
        
        return {
            "total_contracts": len(options),
            "avg_iv": avg_iv,
            "max_oi": max_oi,
            "total_volume": total_volume
        }

# Create service instance
options_service = OptionsDataService()
