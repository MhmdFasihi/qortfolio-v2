"""Options Analytics Page"""

import reflex as rx
from ..state import OptionsState

def options_analytics_page() -> rx.Component:
    """Main options analytics page"""
    return rx.vstack(
        # Header
        rx.hstack(
            rx.heading("Options Analytics", size="8", color="#a855f7"),
            rx.spacer(),
            rx.button(
                "Refresh Data",
                on_click=OptionsState.fetch_options_data,
                loading=OptionsState.loading,
                color_scheme="purple",
                size="3",
            ),
            width="100%",
            padding="2rem",
        ),
        
        # Currency Selector
        rx.hstack(
            rx.text("Select Currency:", size="4", weight="bold"),
            rx.select(
                ["BTC", "ETH"],
                value=OptionsState.selected_currency,
                on_change=OptionsState.set_currency,
            ),
            rx.text(f"Status: ", color="#9ca3af"),
            rx.text(OptionsState.db_status, color="#a855f7"),
            spacing="4",
            padding="1rem 2rem",
        ),
        
        # Metrics Cards
        rx.grid(
            metric_card("Total Contracts", OptionsState.total_contracts),
            metric_card("Avg IV", OptionsState.avg_iv, is_percentage=True),
            metric_card("Max OI", OptionsState.max_oi),
            metric_card("Volume", OptionsState.total_volume),
            columns="4",
            spacing="4",
            width="100%",
            padding="0 2rem",
        ),
        
        # Options Chain Table
        rx.card(
            rx.vstack(
                rx.heading("Options Chain", size="5"),
                rx.cond(
                    OptionsState.loading,
                    rx.center(rx.spinner(color="purple", size="3")),
                    rx.cond(
                        OptionsState.options_data.length() > 0,
                        rx.data_table(
                            data=OptionsState.options_data,
                            columns=[
                                {"name": "Strike", "id": "strike"},
                                {"name": "Type", "id": "option_type"},
                                {"name": "Expiry", "id": "expiry"},
                                {"name": "Bid", "id": "bid"},
                                {"name": "Ask", "id": "ask"},
                                {"name": "IV", "id": "iv"},
                                {"name": "Volume", "id": "volume"},
                                {"name": "OI", "id": "open_interest"},
                            ],
                        ),
                        rx.text("No data loaded. Click 'Refresh Data' button.", color="#9ca3af"),
                    ),
                ),
                spacing="3",
            ),
            width="95%",
            margin="2rem",
        ),
        
        width="100%",
        style={
            "background": "linear-gradient(135deg, #1a0033 0%, #220044 50%, #1a0033 100%)",
            "min_height": "100vh",
        },
    )

def metric_card(label: str, value: rx.Var, is_percentage: bool = False) -> rx.Component:
    """Metric display card"""
    display_value = rx.cond(
        is_percentage,
        rx.text(f"{value:.1%}", size="5", weight="bold"),
        rx.text(value, size="5", weight="bold"),
    )
    
    return rx.card(
        rx.vstack(
            rx.text(label, size="2", color="#a855f7"),
            display_value,
            align="center",
        ),
        style={
            "background": "rgba(45, 27, 61, 0.8)",
            "border": "1px solid #4c1d95",
            "min_height": "100px",
        }
    )
