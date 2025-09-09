"""State Management with MongoDB Integration"""

import reflex as rx
from typing import Dict, List
from datetime import datetime
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class OptionsState(rx.State):
    """Options page state with MongoDB integration"""
    
    # UI State
    selected_currency: str = "BTC"
    loading: bool = False
    db_status: str = "Checking..."
    
    # Data
    options_data: List[Dict] = []
    total_contracts: int = 0
    avg_iv: float = 0.0
    max_oi: int = 0
    total_volume: int = 0
    last_update: str = "Never"
    
    async def check_db_connection(self):
        """Check MongoDB connection status"""
        try:
            from src.core.database.connection import db_connection
            if db_connection.check_connection():
                self.db_status = "Connected"
                return True
            else:
                self.db_status = "Disconnected"
                return False
        except Exception as e:
            self.db_status = f"Error: {str(e)[:20]}"
            return False
    
    def set_currency(self, currency: str):
        """Set selected currency and fetch data"""
        self.selected_currency = currency
        return self.fetch_options_data
    
    async def fetch_options_data(self):
        """Fetch options data from MongoDB"""
        self.loading = True
        
        try:
            # Check DB connection first
            db_connected = await self.check_db_connection()
            
            # Import service
            from src.data.services.options_service import options_service
            
            # Fetch data
            self.options_data = await options_service.get_options_by_currency(
                self.selected_currency
            )
            
            # Get metrics
            metrics = await options_service.get_metrics(self.selected_currency)
            self.total_contracts = metrics['total_contracts']
            self.avg_iv = metrics['avg_iv']
            self.max_oi = metrics['max_oi']
            self.total_volume = metrics['total_volume']
            
            # Update status
            if db_connected:
                self.db_status = "Connected"
            else:
                self.db_status = "Using Sample Data"
                
        except Exception as e:
            print(f"Error in fetch_options_data: {e}")
            # Use fallback sample data
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
                    "delta": "0.523",
                    "gamma": "0.0012",
                    "theta": "-0.0234",
                    "vega": "0.1234",
                }
            ]
            self.total_contracts = 1
            self.avg_iv = 0.653
            self.max_oi = 890
            self.total_volume = 125
            self.db_status = "Error - Using Fallback"
            
        finally:
            self.loading = False
            self.last_update = datetime.now().strftime("%H:%M:%S")

class State(rx.State):
    """Main app state"""
    pass
