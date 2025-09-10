"""Risk Dashboard State Management"""

import reflex as rx
from typing import Dict, List
from datetime import datetime, timedelta
import numpy as np

class RiskState(rx.State):
    """Risk management state"""
    
    # Risk Metrics
    portfolio_var: float = 0.0  # Value at Risk
    portfolio_cvar: float = 0.0  # Conditional VaR
    beta: float = 0.0
    correlation_to_btc: float = 0.0
    
    # Greeks Exposure
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_theta: float = 0.0
    total_vega: float = 0.0
    
    # Risk Levels
    risk_score: int = 0  # 0-100
    risk_level: str = "Medium"  # Low, Medium, High, Critical
    
    # Historical Risk
    var_history: List[Dict] = []
    drawdown_history: List[Dict] = []
    
    # Correlation Matrix
    correlation_data: List[Dict] = []
    
    # Risk Alerts
    alerts: List[Dict] = []
    
    # UI State
    loading: bool = False
    selected_timeframe: str = "1M"
    
    @rx.var
    def var_display(self) -> str:
        return f"${abs(self.portfolio_var):,.0f}"
    
    @rx.var
    def cvar_display(self) -> str:
        return f"${abs(self.portfolio_cvar):,.0f}"
    
    @rx.var
    def risk_score_display(self) -> str:
        return f"{self.risk_score}/100"
    
    @rx.var
    def risk_color(self) -> str:
        if self.risk_score < 30:
            return "green"
        elif self.risk_score < 60:
            return "yellow"
        elif self.risk_score < 80:
            return "orange"
        else:
            return "red"
    
    def set_timeframe(self, timeframe: str):
        """Change risk timeframe"""
        self.selected_timeframe = timeframe
        self.fetch_risk_data()
    
    def fetch_risk_data(self):
        """Fetch risk metrics"""
        self.loading = True
        
        # Sample risk metrics
        self.portfolio_var = -12500  # 95% VaR
        self.portfolio_cvar = -18750  # 95% CVaR
        self.beta = 1.15
        self.correlation_to_btc = 0.82
        
        # Sample Greeks
        self.total_delta = 0.523
        self.total_gamma = 0.0234
        self.total_theta = -156.78
        self.total_vega = 234.56
        
        # Risk scoring
        self.risk_score = 65
        self.risk_level = "Medium-High"
        
        # Historical VaR (30 days)
        dates = [(datetime.now() - timedelta(days=i)).strftime("%m/%d") 
                 for i in range(30, 0, -1)]
        self.var_history = [
            {"date": date, "value": -10000 - np.random.random() * 5000}
            for date in dates
        ]
        
        # Drawdown history
        self.drawdown_history = [
            {"date": date, "value": -np.random.random() * 20}
            for date in dates
        ]
        
        # Correlation matrix data
        self.correlation_data = [
            {"asset1": "BTC", "asset2": "BTC", "correlation": 1.0},
            {"asset1": "BTC", "asset2": "ETH", "correlation": 0.75},
            {"asset1": "BTC", "asset2": "Options", "correlation": 0.45},
            {"asset1": "ETH", "asset2": "BTC", "correlation": 0.75},
            {"asset1": "ETH", "asset2": "ETH", "correlation": 1.0},
            {"asset1": "ETH", "asset2": "Options", "correlation": 0.52},
        ]
        
        # Risk alerts
        self.alerts = [
            {
                "level": "warning",
                "message": "Portfolio gamma exposure exceeds threshold",
                "timestamp": datetime.now().strftime("%H:%M")
            },
            {
                "level": "info",
                "message": "VaR increased by 15% in last 24h",
                "timestamp": datetime.now().strftime("%H:%M")
            },
            {
                "level": "danger",
                "message": "Correlation spike detected in crypto positions",
                "timestamp": datetime.now().strftime("%H:%M")
            }
        ]
        
        self.loading = False
    
    def clear_alerts(self):
        """Clear risk alerts"""
        self.alerts = []
