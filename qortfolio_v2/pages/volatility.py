"""Volatility Analysis Page"""

import reflex as rx
from ..volatility_state import VolatilityState
from ..components.navigation import page_layout
import plotly.graph_objects as go

def volatility_page() -> rx.Component:
    """Volatility analysis page with sidebar"""
    content = rx.vstack(
        # Header
        rx.hstack(
            rx.heading("Volatility Analysis", size="8", color="#a855f7"),
            rx.spacer(),
            rx.button(
                "Refresh Data",
                on_click=VolatilityState.fetch_volatility_data,
                loading=VolatilityState.loading,
                color_scheme="purple",
                size="3",
            ),
            width="100%",
            padding="2rem",
        ),
        
        # Controls + Auto-refresh
        rx.hstack(
            rx.text("Currency:", size="4", weight="bold"),
            rx.select(
                ["BTC", "ETH"],
                value=VolatilityState.selected_currency,
                on_change=VolatilityState.set_currency,
            ),
            rx.text("Period:", size="4", weight="bold"),
            rx.select(
                ["7d", "30d", "90d", "1y"],
                value=VolatilityState.selected_period,
                on_change=VolatilityState.set_period,
            ),
            rx.text("Auto:", size="3", margin_left="2rem"),
            rx.switch(is_checked=VolatilityState.auto_refresh, on_change=VolatilityState.toggle_auto_refresh),
            rx.select(["15","30","60","120"], value=VolatilityState.refresh_seconds.to_string(), on_change=VolatilityState.set_refresh_seconds, width="80px"),
            spacing="4",
            padding="0 2rem",
        ),
        
        # Metrics Cards
        rx.grid(
            vol_metric_card("Current IV", VolatilityState.iv_display, "purple"),
            vol_metric_card("Current RV", VolatilityState.rv_display, "blue"),
            vol_metric_card("IV Rank", VolatilityState.iv_rank_display, "green"),
            vol_metric_card("IV Premium", VolatilityState.iv_premium, "orange"),
            columns="4",
            spacing="4",
            width="100%",
            padding="2rem",
        ),
        
        # Charts Grid
        rx.grid(
            # IV vs RV Chart
            rx.card(
                rx.vstack(
                    rx.heading("IV vs RV History", size="5"),
                    create_iv_rv_chart(),
                ),
                style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            ),
            
            # Term Structure Chart
            rx.card(
                rx.vstack(
                    rx.heading("Term Structure", size="5"),
                    create_term_structure_chart(),
                ),
                style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            ),
            
            columns="2",
            spacing="4",
            width="95%",
            margin="0 2rem",
        ),
        
        # Volatility Smile
        rx.card(
            rx.vstack(
                rx.heading("Volatility Smile", size="5"),
                create_volatility_smile_chart(),
            ),
            width="95%",
            margin="2rem",
            style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
        ),
        
        width="100%",
    )
    return page_layout(content, "Volatility Analysis")

def vol_metric_card(label: str, value: rx.Var, color: str) -> rx.Component:
    """Volatility metric card"""
    return rx.card(
        rx.vstack(
            rx.text(label, size="2", color=f"#{color}.400"),
            rx.text(value, size="6", weight="bold"),
            align="center",
        ),
        style={
            "background": "rgba(45, 27, 61, 0.8)",
            "border": f"1px solid var(--{color}-9)",
            "min_height": "100px",
        }
    )

def create_iv_rv_chart() -> rx.Component:
    """Create IV vs RV comparison chart (interactive, percent axis)."""
    return rx.recharts.line_chart(
        rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
        rx.recharts.line(
            data_key="iv_pct",
            stroke="#a855f7",
            name="IV (%)",
            dot=False,
        ),
        rx.recharts.line(
            data_key="rv_pct",
            stroke="#3b82f6",
            name="RV (%)",
            dot=False,
        ),
        rx.recharts.x_axis(data_key="date"),
        rx.recharts.y_axis(),
        rx.recharts.tooltip(),
        rx.recharts.legend(),
        data=VolatilityState.iv_rv_data,
        height=320,
    )

def create_term_structure_chart() -> rx.Component:
    """Create term structure chart (interactive, percent axis)."""
    return rx.recharts.bar_chart(
        rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
        rx.recharts.bar(
            data_key="iv_pct",
            fill="#a855f7",
            name="Avg IV (%)",
        ),
        rx.recharts.x_axis(data_key="expiry"),
        rx.recharts.y_axis(),
        rx.recharts.tooltip(),
        data=VolatilityState.term_structure,
        height=320,
    )

def create_volatility_smile_chart() -> rx.Component:
    """Create volatility smile chart"""
    return rx.recharts.line_chart(
        rx.recharts.line(
            data_key="iv",
            stroke="#a855f7",
            stroke_width=2,
        ),
        rx.recharts.x_axis(data_key="strike"),
        rx.recharts.y_axis(),
        rx.recharts.tooltip(),
        data=VolatilityState.volatility_smile,
        height=300,
    )
