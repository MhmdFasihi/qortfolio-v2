"""Export utilities for data download"""

import reflex as rx
import pandas as pd
import json
from typing import List, Dict
from datetime import datetime

class ExportUtils:
    """Utilities for exporting data"""
    
    @staticmethod
    def export_to_csv(data: List[Dict], filename: str = None) -> str:
        """Convert data to CSV format"""
        if not data:
            return ""
        
        df = pd.DataFrame(data)
        if filename is None:
            filename = f"qortfolio_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return df.to_csv(index=False)
    
    @staticmethod
    def export_to_json(data: List[Dict], filename: str = None) -> str:
        """Convert data to JSON format"""
        if not data:
            return "{}"
        
        if filename is None:
            filename = f"qortfolio_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        return json.dumps(data, indent=2, default=str)

def export_button(data: List[Dict], filename: str = "export") -> rx.Component:
    """Create export button with download functionality"""
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
                on_click=lambda: download_csv(data, filename),
            ),
            rx.menu_item(
                "Export as JSON",
                on_click=lambda: download_json(data, filename),
            ),
        ),
    )

def download_csv(data: List[Dict], filename: str):
    """Trigger CSV download"""
    csv_content = ExportUtils.export_to_csv(data, f"{filename}.csv")
    # In a real implementation, this would trigger a download
    # For now, we'll just print to console
    print(f"Downloading {filename}.csv")
    return csv_content

def download_json(data: List[Dict], filename: str):
    """Trigger JSON download"""
    json_content = ExportUtils.export_to_json(data, f"{filename}.json")
    print(f"Downloading {filename}.json")
    return json_content
