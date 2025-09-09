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
            db = await self.db.get_database_async()
            options_col = db.options_data
            price_col = db.price_data

            # Fetch recent options data for the underlying
            cursor = options_col.find({"underlying": currency}).sort("timestamp", -1).limit(1000)
            options_data = await cursor.to_list(length=1000)
            
            if not options_data:
                return self._get_default_metrics()
            
            # Calculate average IV (ATM options)
            iv_values = []
            atm_ivs = []
            for opt in options_data:
                iv = opt.get('mark_iv')
                if iv is None:
                    iv = opt.get('implied_volatility')
                try:
                    iv = float(iv)
                    if iv > 1:
                        iv = iv / 100.0
                except Exception:
                    iv = None
                if iv is not None:
                    iv_values.append(iv)
                    m = opt.get('moneyness')
                    if m is None:
                        try:
                            strike = float(opt.get('strike', 0))
                            spot = float(opt.get('underlying_price', 0) or 0)
                            m = strike / spot if spot > 0 else None
                        except Exception:
                            m = None
                    if m is not None and 0.95 < m < 1.05:
                        atm_ivs.append(iv)

            current_iv = float(np.mean(atm_ivs)) if atm_ivs else (float(np.mean(iv_values)) if iv_values else 0.65)
            
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
            # Get historical prices
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)

            cursor = price_col.find({
                "symbol": currency,
                "timestamp": {"$gte": start_date, "$lte": end_date}
            }).sort("timestamp", 1)
            prices = await cursor.to_list(length=None)

            if len(prices) < 2:
                return 0.58

            closes = [float(p.get('close', 0)) for p in prices]
            closes = [c for c in closes if c > 0]
            if len(closes) < 2:
                return 0.58

            returns = np.diff(np.log(closes))
            rv = float(np.std(returns) * np.sqrt(365))
            return rv
            
        except Exception as e:
            logger.error(f"Error calculating RV: {e}")
            return 0.58
    
    async def _calculate_iv_metrics(self, currency: str, options_col) -> tuple:
        """Calculate IV rank and percentile"""
        try:
            # Get 1 year of IV history
            one_year_ago = datetime.utcnow() - timedelta(days=365)

            pipeline = [
                {"$match": {"underlying": currency, "timestamp": {"$gte": one_year_ago}}},
                {"$group": {
                    "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                    "avg_iv": {"$avg": {"$ifNull": ["$mark_iv", "$implied_volatility"]}}
                }},
                {"$sort": {"_id": 1}}
            ]
            iv_history = await options_col.aggregate(pipeline).to_list(length=None)
            
            if not iv_history:
                return 50.0, 50.0
            
            ivs = [float(h.get('avg_iv', 0) or 0) for h in iv_history]
            # Normalize if stored as percent.
            ivs = [iv/100.0 if iv > 1 else iv for iv in ivs]
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
            db = await self.db.get_database_async()
            options_col = db.options_data
            
            # Get unique expiries
            pipeline = [
                {"$match": {"underlying": currency}},
                {"$group": {
                    "_id": "$expiry",
                    "avg_iv": {"$avg": {"$ifNull": ["$mark_iv", "$implied_volatility"]}},
                    "count": {"$sum": 1}
                }},
                {"$sort": {"_id": 1}},
                {"$limit": 12}
            ]
            expiries = await options_col.aggregate(pipeline).to_list(length=None)
            
            if not expiries:
                return self._get_default_term_structure()
            
            # Format for display
            term_structure = []
            for exp in expiries:
                exp_date = exp['_id']
                # exp_date might be stored as datetime or string
                if isinstance(exp_date, str):
                    try:
                        exp_date = datetime.fromisoformat(exp_date)
                    except Exception:
                        exp_date = datetime.utcnow() + timedelta(days=30)
                days_to_exp = (exp_date - datetime.utcnow()).days
                
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
                    "iv": (exp.get('avg_iv')/100.0 if exp.get('avg_iv', 0) and exp.get('avg_iv') > 1 else exp.get('avg_iv', 0) or 0),
                    "days": days_to_exp
                })
            
            return term_structure
            
        except Exception as e:
            logger.error(f"Error getting term structure: {e}")
            return self._get_default_term_structure()
    
    async def get_volatility_smile(self, currency: str = "BTC", expiry_days: int = 30) -> List[Dict]:
        """Get volatility smile for specific expiry"""
        try:
            db = await self.db.get_database_async()
            options_col = db.options_data
            
            # Find target expiry
            target_date = datetime.now() + timedelta(days=expiry_days)
            
            # Get options for that expiry
            pipeline = [
                {"$match": {
                    "underlying": currency,
                    "expiry": {"$gte": target_date - timedelta(days=5), "$lte": target_date + timedelta(days=5)}
                }},
                {"$group": {
                    "_id": "$strike",
                    "avg_iv": {"$avg": {"$ifNull": ["$mark_iv", "$implied_volatility"]}}
                }},
                {"$sort": {"_id": 1}}
            ]
            strikes = await options_col.aggregate(pipeline).to_list(length=None)
            
            if not strikes:
                return self._get_default_smile()
            
            # Format for display
            smile = [
                {"strike": s['_id'], "iv": (s.get('avg_iv')/100.0 if s.get('avg_iv', 0) and s.get('avg_iv') > 1 else s.get('avg_iv', 0) or 0)}
                for s in strikes
            ]
            
            return smile
            
        except Exception as e:
            logger.error(f"Error getting volatility smile: {e}")
            return self._get_default_smile()
    
    async def get_iv_history(self, currency: str = "BTC", days: int = 30) -> List[Dict]:
        """Get historical IV data"""
        try:
            db = await self.db.get_database_async()
            options_col = db.options_data

            start_date = datetime.utcnow() - timedelta(days=days)

            pipeline = [
                {"$match": {"underlying": currency, "timestamp": {"$gte": start_date}}},
                {"$group": {
                    "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                    "value": {"$avg": {"$ifNull": ["$mark_iv", "$implied_volatility"]}}
                }},
                {"$sort": {"_id": 1}}
            ]
            iv_history = await options_col.aggregate(pipeline).to_list(length=None)

            out = []
            for h in iv_history:
                v = h.get('value', 0) or 0
                v = v/100.0 if v > 1 else v
                out.append({"date": h['_id'], "value": v})
            return out
            
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
