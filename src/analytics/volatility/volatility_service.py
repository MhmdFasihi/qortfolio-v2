"""Volatility Analytics Service"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from src.core.database.connection import db_connection
import logging

logger = logging.getLogger(__name__)

class VolatilityService:
    """Service for volatility calculations and analysis"""
    
    def __init__(self):
        self.db = db_connection
        
    async def get_volatility_metrics(self, currency: str = "BTC") -> Dict:
        """Calculate volatility metrics from options data"""
        try:
            # Get options collection
            options_col = self.db.get_collection('options_data')
            price_col = self.db.get_collection('price_data')
            
            if not options_col:
                return self._get_default_metrics()
            
            # Fetch options data
            options_cursor = options_col.find(
                {"underlying_currency": currency}
            ).sort("timestamp", -1).limit(500)
            
            options_data = list(options_cursor)
            
            if not options_data:
                return self._get_default_metrics()
            
            # Calculate average IV (ATM options)
            atm_ivs = []
            for opt in options_data:
                if opt.get('moneyness') and 0.95 < opt.get('moneyness', 0) < 1.05:
                    iv = opt.get('implied_volatility', 0)
                    if iv > 0:
                        atm_ivs.append(iv)
            
            current_iv = np.mean(atm_ivs) if atm_ivs else 0.65
            
            # Calculate realized volatility from price data
            current_rv = await self._calculate_realized_vol(currency, price_col)
            
            # Calculate IV rank and percentile
            iv_rank, iv_percentile = await self._calculate_iv_metrics(currency, options_col)
            
            return {
                "current_iv": current_iv,
                "current_rv": current_rv,
                "iv_rank": iv_rank,
                "iv_percentile": iv_percentile
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {e}")
            return self._get_default_metrics()
    
    async def _calculate_realized_vol(self, currency: str, price_col, days: int = 30) -> float:
        """Calculate realized volatility from price data"""
        try:
            if not price_col:
                return 0.58
            
            # Get historical prices
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            prices = list(price_col.find(
                {
                    "symbol": f"{currency}-USD",
                    "timestamp": {"$gte": start_date, "$lte": end_date}
                }
            ).sort("timestamp", 1))
            
            if len(prices) < 2:
                return 0.58
            
            # Calculate returns
            price_values = [p['price'] for p in prices]
            returns = np.diff(np.log(price_values))
            
            # Annualized volatility
            rv = np.std(returns) * np.sqrt(365)
            return rv
            
        except Exception as e:
            logger.error(f"Error calculating RV: {e}")
            return 0.58
    
    async def _calculate_iv_metrics(self, currency: str, options_col) -> tuple:
        """Calculate IV rank and percentile"""
        try:
            # Get 1 year of IV history
            one_year_ago = datetime.now() - timedelta(days=365)
            
            iv_history = list(options_col.aggregate([
                {
                    "$match": {
                        "underlying_currency": currency,
                        "timestamp": {"$gte": one_year_ago}
                    }
                },
                {
                    "$group": {
                        "_id": {
                            "$dateToString": {
                                "format": "%Y-%m-%d",
                                "date": "$timestamp"
                            }
                        },
                        "avg_iv": {"$avg": "$implied_volatility"}
                    }
                },
                {"$sort": {"avg_iv": 1}}
            ]))
            
            if not iv_history:
                return 50.0, 50.0
            
            ivs = [h['avg_iv'] for h in iv_history]
            current_iv = ivs[-1] if ivs else 0.65
            
            # IV Rank = (Current IV - Min IV) / (Max IV - Min IV) * 100
            iv_rank = ((current_iv - min(ivs)) / (max(ivs) - min(ivs)) * 100) if max(ivs) != min(ivs) else 50
            
            # IV Percentile = % of days with IV below current
            iv_percentile = (len([iv for iv in ivs if iv < current_iv]) / len(ivs) * 100) if ivs else 50
            
            return iv_rank, iv_percentile
            
        except Exception as e:
            logger.error(f"Error calculating IV metrics: {e}")
            return 50.0, 50.0
    
    async def get_term_structure(self, currency: str = "BTC") -> List[Dict]:
        """Get volatility term structure"""
        try:
            options_col = self.db.get_collection('options_data')
            if not options_col:
                return self._get_default_term_structure()
            
            # Get unique expiries
            pipeline = [
                {"$match": {"underlying_currency": currency}},
                {"$group": {
                    "_id": "$expiration_date",
                    "avg_iv": {"$avg": "$implied_volatility"},
                    "count": {"$sum": 1}
                }},
                {"$sort": {"_id": 1}},
                {"$limit": 10}
            ]
            
            expiries = list(options_col.aggregate(pipeline))
            
            if not expiries:
                return self._get_default_term_structure()
            
            # Format for display
            term_structure = []
            for exp in expiries:
                exp_date = exp['_id']
                days_to_exp = (exp_date - datetime.now()).days if isinstance(exp_date, datetime) else 30
                
                if days_to_exp <= 7:
                    label = "1W"
                elif days_to_exp <= 30:
                    label = f"{days_to_exp}D"
                elif days_to_exp <= 90:
                    label = f"{days_to_exp//30}M"
                else:
                    label = f"{days_to_exp//30}M"
                
                term_structure.append({
                    "expiry": label,
                    "iv": exp['avg_iv'],
                    "days": days_to_exp
                })
            
            return term_structure
            
        except Exception as e:
            logger.error(f"Error getting term structure: {e}")
            return self._get_default_term_structure()
    
    async def get_volatility_smile(self, currency: str = "BTC", expiry_days: int = 30) -> List[Dict]:
        """Get volatility smile for specific expiry"""
        try:
            options_col = self.db.get_collection('options_data')
            if not options_col:
                return self._get_default_smile()
            
            # Find target expiry
            target_date = datetime.now() + timedelta(days=expiry_days)
            
            # Get options for that expiry
            pipeline = [
                {
                    "$match": {
                        "underlying_currency": currency,
                        "expiration_date": {
                            "$gte": target_date - timedelta(days=5),
                            "$lte": target_date + timedelta(days=5)
                        }
                    }
                },
                {
                    "$group": {
                        "_id": "$strike_price",
                        "avg_iv": {"$avg": "$implied_volatility"}
                    }
                },
                {"$sort": {"_id": 1}}
            ]
            
            strikes = list(options_col.aggregate(pipeline))
            
            if not strikes:
                return self._get_default_smile()
            
            # Format for display
            smile = [
                {
                    "strike": s['_id'],
                    "iv": s['avg_iv']
                }
                for s in strikes
            ]
            
            return smile
            
        except Exception as e:
            logger.error(f"Error getting volatility smile: {e}")
            return self._get_default_smile()
    
    async def get_iv_history(self, currency: str = "BTC", days: int = 30) -> List[Dict]:
        """Get historical IV data"""
        try:
            options_col = self.db.get_collection('options_data')
            if not options_col:
                return []
            
            start_date = datetime.now() - timedelta(days=days)
            
            # Daily average IV
            pipeline = [
                {
                    "$match": {
                        "underlying_currency": currency,
                        "timestamp": {"$gte": start_date}
                    }
                },
                {
                    "$group": {
                        "_id": {
                            "$dateToString": {
                                "format": "%Y-%m-%d",
                                "date": "$timestamp"
                            }
                        },
                        "value": {"$avg": "$implied_volatility"}
                    }
                },
                {"$sort": {"_id": 1}}
            ]
            
            iv_history = list(options_col.aggregate(pipeline))
            
            return [{"date": h['_id'], "value": h['value']} for h in iv_history]
            
        except Exception as e:
            logger.error(f"Error getting IV history: {e}")
            return []
    
    def _get_default_metrics(self) -> Dict:
        """Default metrics when no data available"""
        return {
            "current_iv": 0.65,
            "current_rv": 0.58,
            "iv_rank": 50.0,
            "iv_percentile": 50.0
        }
    
    def _get_default_term_structure(self) -> List[Dict]:
        """Default term structure"""
        return [
            {"expiry": "1W", "iv": 0.62, "days": 7},
            {"expiry": "2W", "iv": 0.64, "days": 14},
            {"expiry": "1M", "iv": 0.65, "days": 30},
            {"expiry": "2M", "iv": 0.66, "days": 60},
        ]
    
    def _get_default_smile(self) -> List[Dict]:
        """Default volatility smile"""
        return [
            {"strike": 40000, "iv": 0.72},
            {"strike": 42500, "iv": 0.68},
            {"strike": 45000, "iv": 0.65},
            {"strike": 47500, "iv": 0.68},
            {"strike": 50000, "iv": 0.72},
        ]

# Create service instance
volatility_service = VolatilityService()
