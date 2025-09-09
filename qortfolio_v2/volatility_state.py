"""Volatility Analysis State Management"""

import reflex as rx
from typing import Dict, List
from datetime import datetime, timedelta
import numpy as np

class VolatilityState(rx.State):
    """Volatility analysis state"""
    
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
        self.selected_currency = currency
        return self.fetch_volatility_data
    
    def set_period(self, period: str):
        self.selected_period = period
        return self.fetch_volatility_data
    
    async def fetch_volatility_data(self):
        """Fetch volatility data"""
        self.loading = True
        
        # Generate sample data
        dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") 
                 for i in range(30, 0, -1)]
        
        # IV vs RV time series
        self.iv_data = [
            {"date": date, "value": 0.5 + np.random.random() * 0.3}
            for date in dates
        ]
        
        self.rv_data = [
            {"date": date, "value": 0.45 + np.random.random() * 0.25}
            for date in dates
        ]
        
        # Term structure (different expiries)
        expiries = ["1W", "2W", "1M", "2M", "3M", "6M"]
        self.term_structure = [
            {"expiry": exp, "iv": 0.5 + i * 0.02 + np.random.random() * 0.1}
            for i, exp in enumerate(expiries)
        ]
        
        # Volatility smile
        strikes = list(range(35000, 55000, 2000)) if self.selected_currency == "BTC" else list(range(2000, 4000, 200))
        self.volatility_smile = [
            {
                "strike": strike,
                "iv": 0.6 + abs(45000 - strike) / 100000 + np.random.random() * 0.05
            }
            for strike in strikes
        ]
        
        # Update metrics
        self.current_iv = 0.652
        self.current_rv = 0.584
        self.iv_rank = 75.3
        self.iv_percentile = 82.1
        
        self.loading = False
