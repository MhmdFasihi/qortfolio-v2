"""Volatility Analysis State with Real Data"""

import reflex as rx
from typing import Dict, List
from datetime import datetime
import asyncio

class VolatilityState(rx.State):
    """Volatility analysis state with MongoDB data"""
    
    # Selection
    selected_currency: str = "BTC"
    selected_period: str = "30d"
    
    # Data
    iv_data: List[Dict] = []
    rv_data: List[Dict] = []  
    iv_rv_data: List[Dict] = []
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
    # Auto refresh
    auto_refresh: bool = False
    refresh_seconds: int = 60
    
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
        # Return an event spec to trigger async fetch (no yield from)
        return VolatilityState.fetch_volatility_data()
    
    def set_period(self, period: str):
        """Set period and fetch data"""
        self.selected_period = period
        return VolatilityState.fetch_volatility_data()

    def set_refresh_seconds(self, secs: str):
        try:
            self.refresh_seconds = int(secs)
        except Exception:
            self.refresh_seconds = 60

    def toggle_auto_refresh(self, value: bool):
        if bool(value):
            return VolatilityState.start_auto_refresh()
        else:
            return VolatilityState.stop_auto_refresh()

    async def start_auto_refresh(self):
        if self.auto_refresh:
            return
        self.auto_refresh = True
        try:
            while self.auto_refresh:
                await self.fetch_volatility_data()
                delay = self.refresh_seconds if self.refresh_seconds and self.refresh_seconds > 0 else 60
                await asyncio.sleep(delay)
        except Exception:
            self.auto_refresh = False

    def stop_auto_refresh(self):
        self.auto_refresh = False
    
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
                # Add percentage convenience field for charting
                try:
                    self.term_structure = [
                        {**row, "iv_pct": round(float(row.get("iv", 0) or 0) * 100.0, 2)}
                        for row in self.term_structure
                    ]
                except Exception:
                    pass
                
                # Fetch volatility smile
                self.volatility_smile = await volatility_service.get_volatility_smile(self.selected_currency)
                
                # Fetch IV history
                self.iv_data = await volatility_service.get_iv_history(self.selected_currency, period_days)
                
                # Fetch RV history with a sensible rolling window
                rv_window = 7 if period_days <= 7 else 30
                self.rv_data = await volatility_service.get_rv_history(
                    self.selected_currency, days=period_days, window=rv_window
                )

                # Build combined IV vs RV series for charting
                iv_map = {d["date"]: d["value"] for d in self.iv_data}
                rv_map = {d["date"]: d["value"] for d in self.rv_data}
                all_dates = sorted(set(iv_map.keys()) | set(rv_map.keys()))
                self.iv_rv_data = []
                for dt in all_dates:
                    ivv = iv_map.get(dt, None)
                    rvv = rv_map.get(dt, None)
                    entry = {"date": dt, "iv": ivv, "rv": rvv}
                    try:
                        entry["iv_pct"] = round(ivv * 100.0, 2) if ivv is not None else None
                        entry["rv_pct"] = round(rvv * 100.0, 2) if rvv is not None else None
                    except Exception:
                        entry["iv_pct"] = None
                        entry["rv_pct"] = None
                    self.iv_rv_data.append(entry)
                
            else:
                self.db_status = "Using Sample Data"
                self._load_sample_data()
                
        except Exception as e:
            print(f"Error fetching volatility data: {e}")
            self.db_status = f"Error: {str(e)[:30]}"
            self._load_sample_data()
            
        finally:
            self.loading = False
    
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

        # Sample IV/RV history
        self.iv_data = [
            {"date": "2024-09-01", "value": 0.60},
            {"date": "2024-09-05", "value": 0.62},
            {"date": "2024-09-10", "value": 0.64},
            {"date": "2024-09-15", "value": 0.66},
        ]
        self.rv_data = [
            {"date": d["date"], "value": d["value"] * 0.9}
            for d in self.iv_data
        ]
        self.iv_rv_data = []
        for d in self.iv_data:
            ivv = d["value"]
            rvv = d["value"] * 0.9
            self.iv_rv_data.append({
                "date": d["date"],
                "iv": ivv,
                "rv": rvv,
                "iv_pct": round(ivv * 100.0, 2),
                "rv_pct": round(rvv * 100.0, 2),
            })

        # Add iv_pct to term structure sample
        try:
            self.term_structure = [
                {**row, "iv_pct": round(float(row.get("iv", 0) or 0) * 100.0, 2)}
                for row in self.term_structure
            ]
        except Exception:
            pass
