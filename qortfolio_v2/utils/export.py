"""Working Export Utilities"""

import reflex as rx
import pandas as pd
import json
import io
import base64
from typing import List, Dict
from datetime import datetime

class ExportState(rx.State):
    """State for handling exports"""
    
    download_url: str = ""
    download_filename: str = ""
    
    def export_to_csv(self, data: List[Dict], filename: str = None):
        """Export data to CSV with download"""
        if not data:
            return
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Generate filename
        if filename is None:
            filename = f"qortfolio_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()
        
        # Create data URL for download
        b64 = base64.b64encode(csv_string.encode()).decode()
        self.download_url = f"data:text/csv;base64,{b64}"
        self.download_filename = filename
        
        # Trigger download
        return rx.download(url=self.download_url, filename=self.download_filename)
    
    def export_to_json(self, data: List[Dict], filename: str = None):
        """Export data to JSON with download"""
        if not data:
            return
        
        # Generate filename
        if filename is None:
            filename = f"qortfolio_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to JSON
        json_string = json.dumps(data, indent=2, default=str)
        
        # Create data URL for download
        b64 = base64.b64encode(json_string.encode()).decode()
        self.download_url = f"data:application/json;base64,{b64}"
        self.download_filename = filename
        
        # Trigger download
        return rx.download(url=self.download_url, filename=self.download_filename)

def export_button(data_state_var: str) -> rx.Component:
    """Create working export button"""
    return rx.menu(
        rx.menu_button(
            rx.button(
                rx.icon("download", size=16),
                "Export",
                size="2",
                color_scheme="gray",
            ),
        ),
        rx.menu_content(
            rx.menu_item(
                "Export as CSV",
                on_click=lambda: ExportState.export_to_csv(
                    getattr(rx.State, data_state_var)
                ),
            ),
            rx.menu_item(
                "Export as JSON", 
                on_click=lambda: ExportState.export_to_json(
                    getattr(rx.State, data_state_var)
                ),
            ),
        ),
    )
