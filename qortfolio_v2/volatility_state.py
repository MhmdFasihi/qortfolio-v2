"""Volatility Analysis State with Real Data"""

import reflex as rx
from typing import Dict, List
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class VolatilityState(rx.State):
    """Volatility analysis state with MongoDB data"""
    
    # Selection
    selected_currency: str = "BTC"
    selected_period: str = "30d"
    
    # Data
    iv_data: List[Dict] = []
    rv_data: List[Dict] = []  
    term_structure: List[Dict] = []
    volatility_smile: List[Dict] = []
    
    # Metrics
    current_iv: float = 0.0
    current_rv: float = 0.0
    iv_rank: float = 0.0
    iv_percentile: float = 0.0
    
    # Loading states
    loading: bool = False
    db_status: str = "Not Connected"
    
    @rx.var
    def iv_display(self) -> str:
        return f"{self.current_iv * 100:.1f}%"
    
    @rx.var
    def rv_display(self) -> str:
        return f"{self.current_rv * 100:.1f}%"
    
    @rx.var
    def iv_rank_display(self) -> str:
        return f"{self.iv_rank:.1f}%"
    
    @rx.var
    def iv_premium(self) -> str:
        premium = (self.current_iv - self.current_rv) * 100
        return f"{premium:+.1f}%"
    
    def set_currency(self, currency: str):
        """Set currency and fetch data"""
        self.selected_currency = currency
        yield from self.fetch_volatility_data()
    
    def set_period(self, period: str):
        """Set period and fetch data"""
        self.selected_period = period
        yield from self.fetch_volatility_data()
    
    async def fetch_volatility_data(self):
        """Fetch real volatility data from MongoDB"""
        self.loading = True
        
        try:
            # Import the service
            from src.analytics.volatility.volatility_service import volatility_service
            from src.core.database.connection import db_connection
            
            # Check DB connection
            if db_connection.check_connection():
                self.db_status = "Connected"
                
                # Get period in days
                period_days = {
                    "7d": 7,
                    "30d": 30,
                    "90d": 90,
                    "1y": 365
                }.get(self.selected_period, 30)
                
                # Fetch metrics
                metrics = await volatility_service.get_volatility_metrics(self.selected_currency)
                self.current_iv = metrics['current_iv']
                self.current_rv = metrics['current_rv']
                self.iv_rank = metrics['iv_rank']
                self.iv_percentile = metrics['iv_percentile']
                
                # Fetch term structure
                self.term_structure = await volatility_service.get_term_structure(self.selected_currency)
                
                # Fetch volatility smile
                self.volatility_smile = await volatility_service.get_volatility_smile(self.selected_currency)
                
                # Fetch IV history
                self.iv_data = await volatility_service.get_iv_history(self.selected_currency, period_days)
                
                # For now, RV data same as IV (you can calculate separately)
                self.rv_data = [
                    {"date": d["date"], "value": d["value"] * 0.9}  # Mock RV as 90% of IV
                    for d in self.iv_data
                ]
                
            else:
                self.db_status = "Using Sample Data"
                self._load_sample_data()
                
        except Exception as e:
            print(f"Error fetching volatility data: {e}")
            self.db_status = f"Error: {str(e)[:30]}"
            self._load_sample_data()
            
        finally:
            self.loading = False
            yield
    
    def _load_sample_data(self):
        """Load sample data as fallback"""
        # Sample term structure
        self.term_structure = [
            {"expiry": "1W", "iv": 0.62},
            {"expiry": "2W", "iv": 0.64},
            {"expiry": "1M", "iv": 0.65},
            {"expiry": "2M", "iv": 0.66},
        ]
        
        # Sample smile
        base_strike = 45000 if self.selected_currency == "BTC" else 3000
        self.volatility_smile = [
            {"strike": base_strike - 5000, "iv": 0.72},
            {"strike": base_strike - 2500, "iv": 0.68},
            {"strike": base_strike, "iv": 0.65},
            {"strike": base_strike + 2500, "iv": 0.68},
            {"strike": base_strike + 5000, "iv": 0.72},
        ]
        
        # Sample metrics
        self.current_iv = 0.65
        self.current_rv = 0.58
        self.iv_rank = 50.0
        self.iv_percentile = 50.0
