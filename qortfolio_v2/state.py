"""State Management with Computed Properties"""

import reflex as rx
from typing import Dict, List
from datetime import datetime

class OptionsState(rx.State):
    """Options page state with sample data"""
    
    # UI State
    selected_currency: str = "BTC"
    loading: bool = False
    db_status: str = "Not Connected"
    
    # Data - Initialize with defaults
    options_data: List[Dict] = []
    total_contracts: int = 0
    avg_iv: float = 0.0
    max_oi: int = 0
    total_volume: int = 0
    last_update: str = "Never"
    
    @rx.var
    def avg_iv_display(self) -> str:
        """Format IV as percentage string"""
        return f"{self.avg_iv * 100:.1f}%"
    
    def set_currency(self, currency: str):
        """Set selected currency"""
        self.selected_currency = currency
        return self.fetch_options_data
    
    def fetch_options_data(self):
        """Load sample options data"""
        self.loading = True
        
        # Sample data for testing
        self.options_data = [
            {
                "strike": 45000,
                "option_type": "CALL",
                "expiry": "2024-09-27",
                "bid": "0.0234",
                "ask": "0.0245",
                "iv": "65.3%",
                "volume": 125,
                "open_interest": 890,
            },
            {
                "strike": 46000,
                "option_type": "PUT",
                "expiry": "2024-09-27", 
                "bid": "0.0156",
                "ask": "0.0162",
                "iv": "62.1%",
                "volume": 89,
                "open_interest": 567,
            },
            {
                "strike": 47000,
                "option_type": "CALL",
                "expiry": "2024-10-25",
                "bid": "0.0345",
                "ask": "0.0356",
                "iv": "68.7%",
                "volume": 234,
                "open_interest": 1245,
            }
        ]
        
        # Update metrics
        self.total_contracts = len(self.options_data)
        self.avg_iv = 0.652  # Store as decimal
        self.max_oi = 1245
        self.total_volume = 448
        self.last_update = datetime.now().strftime("%H:%M:%S")
        self.db_status = "Sample Data"
        self.loading = False

class State(rx.State):
    """Main app state"""
    pass
