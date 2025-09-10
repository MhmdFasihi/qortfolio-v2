"""Portfolio Management State"""

import reflex as rx
from typing import Dict, List
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PortfolioState(rx.State):
    """Portfolio management state"""
    
    # Portfolio data
    positions: List[Dict] = []
    total_value: float = 0.0
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    
    # Allocations
    crypto_allocation: List[Dict] = []
    sector_allocation: List[Dict] = []
    
    # Performance metrics
    daily_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    
    # UI state
    loading: bool = False
    selected_view: str = "positions"  # positions, allocation, performance
    
    @rx.var
    def total_value_display(self) -> str:
        return f"${self.total_value:,.2f}"
    
    @rx.var
    def pnl_display(self) -> str:
        sign = "+" if self.total_pnl >= 0 else ""
        return f"{sign}${abs(self.total_pnl):,.2f}"
    
    @rx.var
    def pnl_percent_display(self) -> str:
        sign = "+" if self.total_pnl_percent >= 0 else ""
        return f"{sign}{abs(self.total_pnl_percent):.2f}%"
    
    @rx.var
    def sharpe_display(self) -> str:
        return f"{self.sharpe_ratio:.2f}"
    
    def set_view(self, view: str):
        """Change portfolio view"""
        self.selected_view = view
    
    def fetch_portfolio_data(self):
        """Fetch portfolio data"""
        self.loading = True
        
        # Sample positions data
        self.positions = [
            {
                "symbol": "BTC",
                "type": "Spot",
                "quantity": 0.5,
                "entry_price": 42000,
                "current_price": 45000,
                "value": 22500,
                "pnl": 1500,
                "pnl_percent": 7.14,
                "allocation": 45.0
            },
            {
                "symbol": "ETH",
                "type": "Spot",
                "quantity": 5,
                "entry_price": 2800,
                "current_price": 3000,
                "value": 15000,
                "pnl": 1000,
                "pnl_percent": 7.14,
                "allocation": 30.0
            },
            {
                "symbol": "BTC-45000-CALL",
                "type": "Option",
                "quantity": 10,
                "entry_price": 0.025,
                "current_price": 0.032,
                "value": 3200,
                "pnl": 700,
                "pnl_percent": 28.0,
                "allocation": 6.4
            },
            {
                "symbol": "ETH-3200-PUT",
                "type": "Option",
                "quantity": 20,
                "entry_price": 0.018,
                "current_price": 0.015,
                "value": 1500,
                "pnl": -600,
                "pnl_percent": -16.67,
                "allocation": 3.0
            }
        ]
        
        # Calculate totals
        self.total_value = sum(p["value"] for p in self.positions)
        self.total_pnl = sum(p["pnl"] for p in self.positions)
        self.total_pnl_percent = (self.total_pnl / (self.total_value - self.total_pnl)) * 100 if self.total_value > 0 else 0
        
        # Crypto allocation
        self.crypto_allocation = [
            {"name": "BTC", "value": 45.0, "color": "#f7931a"},
            {"name": "ETH", "value": 30.0, "color": "#627eea"},
            {"name": "Options", "value": 10.0, "color": "#a855f7"},
            {"name": "Cash", "value": 15.0, "color": "#10b981"}
        ]
        
        # Sector allocation
        self.sector_allocation = [
            {"name": "Layer 1", "value": 75.0},
            {"name": "DeFi", "value": 10.0},
            {"name": "Options", "value": 10.0},
            {"name": "Stables", "value": 5.0}
        ]
        
        # Performance metrics
        self.daily_return = 2.3
        self.sharpe_ratio = 1.45
        self.max_drawdown = -12.5
        self.win_rate = 65.0
        
        self.loading = False
    
    def add_position(self):
        """Add new position - placeholder"""
        pass
    
    def close_position(self, symbol: str):
        """Close a position"""
        self.positions = [p for p in self.positions if p["symbol"] != symbol]
        self.fetch_portfolio_data()  # Recalculate
