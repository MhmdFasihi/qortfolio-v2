"""Global State Management"""

import reflex as rx
from typing import Dict, List

class State(rx.State):
    """Global application state"""
    
    # Database status
    db_connected: bool = False
    mongodb_status: str = "Checking..."
    
    # Selected currency
    selected_currency: str = "BTC"
    available_currencies: List[str] = ["BTC", "ETH"]
    
    # Options data
    options_data: List[Dict] = []
    options_loading: bool = False
    last_update: str = ""
    
    # Volatility
    implied_volatility: float = 0.65
    realized_volatility: float = 0.58
    
    def select_currency(self, currency: str):
        """Change selected cryptocurrency"""
        self.selected_currency = currency
