"""Options Analytics Page"""

import reflex as rx
from ..state import State

def options_page() -> rx.Component:
    """Main options analytics page"""
    return rx.vstack(
        rx.heading(
            "Options Analytics",
            size="8",
            color="#a855f7",
        ),
        
        rx.hstack(
            rx.text("Select Currency:"),
            rx.select(
                State.available_currencies,
                value=State.selected_currency,
                on_change=State.select_currency,
            ),
            rx.text(f"Status: {State.mongodb_status}"),
            spacing="4",
        ),
        
        rx.grid(
            metric_card("IV", f"{State.implied_volatility:.2%}"),
            metric_card("RV", f"{State.realized_volatility:.2%}"),
            columns="2",
            spacing="4",
            width="100%",
        ),
        
        rx.card(
            rx.text("Options chain will appear here"),
            width="100%",
        ),
        
        padding="2rem",
        width="100%",
        style={
            "background": "linear-gradient(135deg, #1a0033 0%, #220044 50%, #1a0033 100%)",
            "min_height": "100vh",
        }
    )

def metric_card(label: str, value: str) -> rx.Component:
    """Metric card component"""
    return rx.card(
        rx.vstack(
            rx.text(label, size="2", color="#a855f7"),
            rx.text(value, size="6"),
        ),
        style={
            "background": "rgba(45, 27, 61, 0.8)",
            "border": "1px solid #4c1d95",
        }
    )
