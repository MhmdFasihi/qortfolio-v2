"""Options Analytics Page with MongoDB/Deribit integration"""

import reflex as rx
from ..state import OptionsState


def options_analytics_page() -> rx.Component:
    """Main options analytics page with expiry filtering"""
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

        # Currency / Expiry Selector + DB status
        rx.hstack(
            rx.text("Select Currency:", size="4", weight="bold"),
            rx.select(
                ["BTC", "ETH"],
                value=OptionsState.selected_currency,
                on_change=OptionsState.set_currency,
            ),
            rx.text("Expiry:", size="4", weight="bold", margin_left="2rem"),
            rx.select(
                OptionsState.expiry_options,
                value=OptionsState.selected_expiry,
                on_change=OptionsState.set_expiry,
                placeholder="Select expiry",
            ),
            rx.text("Database:", color="#9ca3af"),
            rx.text(OptionsState.db_status),
            spacing="4",
            padding="1rem 2rem",
        ),

        # Metrics Cards
        rx.grid(
            metric_card("Total Contracts", OptionsState.total_contracts),
            metric_card("Avg IV", OptionsState.avg_iv_pct),
            metric_card("Max OI", OptionsState.max_oi),
            metric_card("Last Update", OptionsState.last_update),
            columns="4",
            spacing="4",
            width="100%",
            padding="0 2rem",
        ),

        # Options Chain Tables (Calls / Puts)
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
                                    {"name": "Volume", "id": "volume"},
                                    {"name": "OI", "id": "open_interest"},
                                ],
                                pagination=True,
                                page_size=15,
                                style={
                                    "background": "transparent",
                                    "color": "#e2e8f0",
                                },
                            ),
                            rx.text("No call options.", color="#9ca3af"),
                        ),
                    ),
                ),
                style={
                    "background": "rgba(45, 27, 61, 0.8)",
                    "border": "1px solid #4c1d95",
                },
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
                                    {"name": "Volume", "id": "volume"},
                                    {"name": "OI", "id": "open_interest"},
                                ],
                                pagination=True,
                                page_size=15,
                                style={
                                    "background": "transparent",
                                    "color": "#e2e8f0",
                                },
                            ),
                            rx.text("No put options.", color="#9ca3af"),
                        ),
                    ),
                ),
                style={
                    "background": "rgba(45, 27, 61, 0.8)",
                    "border": "1px solid #4c1d95",
                },
            ),
            columns="2",
            spacing="4",
            width="100%",
            padding="0 2rem 2rem",
        ),

        width="100%",
        style={
            "background": "linear-gradient(135deg, #1a0033 0%, #220044 50%, #1a0033 100%)",
            "min_height": "100vh",
        },
    )


def metric_card(label: str, value) -> rx.Component:
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
        }
    )
