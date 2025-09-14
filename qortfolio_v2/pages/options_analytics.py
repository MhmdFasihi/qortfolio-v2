"""Options Analytics Page with Navigation and Calls/Puts separation"""

import reflex as rx
from ..state import OptionsState
from ..components.navigation import page_layout

def options_analytics_page() -> rx.Component:
    """Options analytics page with sidebar, expiry filter, calls/puts tables."""
    content = rx.vstack(
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
        
        # Currency / Expiry / Status / Auto-refresh
        rx.hstack(
            rx.text("Currency:", size="4", weight="bold"),
            rx.select(["BTC", "ETH"], value=OptionsState.selected_currency, on_change=OptionsState.set_currency),
            rx.text("Expiry:", size="4", weight="bold", margin_left="2rem"),
            rx.select(OptionsState.expiry_options, value=OptionsState.selected_expiry, on_change=OptionsState.set_expiry, placeholder="Select expiry"),
            rx.text("Auto:", size="3", margin_left="2rem"),
            rx.switch(is_checked=OptionsState.auto_refresh, on_change=OptionsState.toggle_auto_refresh),
            rx.select(["15","30","60","120"], value=OptionsState.refresh_seconds.to_string(), on_change=OptionsState.set_refresh_seconds, width="80px"),
            rx.spacer(),
            rx.text("Spot:", color="#9ca3af"),
            rx.text(OptionsState.spot_display),
            rx.text("DB:", color="#9ca3af", margin_left="1rem"),
            rx.text(OptionsState.db_status, color="#a855f7"),
            spacing="4",
            padding="1rem 2rem",
        ),
        
        # Metrics Cards
        rx.grid(
            metric_card("Total Contracts", OptionsState.total_contracts),
            metric_card("Avg IV", OptionsState.avg_iv_pct),
            metric_card("Max OI", OptionsState.max_oi),
            metric_card("Volume", OptionsState.total_volume),
            columns="4",
            spacing="4",
            width="100%",
            padding="0 2rem",
        ),
        
        # Calls / Puts tables
        rx.grid(
            rx.card(
                rx.vstack(
                    rx.heading("Calls", size="5"),
                    rx.cond(
                        OptionsState.loading,
                        rx.center(rx.spinner(color="purple", size="3")),
                        rx.cond(
                            OptionsState.calls_data,
                            rx.data_table(
                                data=OptionsState.calls_data,
                                columns=[
                                    {"name": "Strike", "id": "strike"},
                                    {"name": "Bid", "id": "bid"},
                                    {"name": "Ask", "id": "ask"},
                                    {"name": "IV", "id": "iv"},
                                    {"name": "Vol", "id": "volume"},
                                    {"name": "OI", "id": "open_interest"},
                                ],
                                pagination=True,
                                page_size=15,
                            ),
                            rx.text("No Calls.", color="#9ca3af"),
                        ),
                    ),
                ),
                style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            ),
            rx.card(
                rx.vstack(
                    rx.heading("Puts", size="5"),
                    rx.cond(
                        OptionsState.loading,
                        rx.center(rx.spinner(color="purple", size="3")),
                        rx.cond(
                            OptionsState.puts_data,
                            rx.data_table(
                                data=OptionsState.puts_data,
                                columns=[
                                    {"name": "Strike", "id": "strike"},
                                    {"name": "Bid", "id": "bid"},
                                    {"name": "Ask", "id": "ask"},
                                    {"name": "IV", "id": "iv"},
                                    {"name": "Vol", "id": "volume"},
                                    {"name": "OI", "id": "open_interest"},
                                ],
                                pagination=True,
                                page_size=15,
                            ),
                            rx.text("No Puts.", color="#9ca3af"),
                        ),
                    ),
                ),
                style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            ),
            columns="2",
            spacing="4",
            width="95%",
            margin="0 2rem 2rem",
        ),
        
        width="100%",
    )
    
    return page_layout(content, "Options Analytics")

def metric_card(label: str, value: rx.Var) -> rx.Component:
    """Metric display card"""
    return rx.card(
        rx.vstack(
            rx.text(label, size="2", color="#a855f7"),
            rx.text(value, size="5", weight="bold"),
            align="center",
        ),
        style={
            "background": "rgba(45, 27, 61, 0.8)",
            "border": "1px solid #4c1d95",
            "min_height": "100px",
        }
    )
