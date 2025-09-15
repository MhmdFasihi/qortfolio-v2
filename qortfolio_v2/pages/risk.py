"""Risk Dashboard Page"""

import reflex as rx
from typing import Dict
from ..components.navigation import page_layout
from ..state import RiskState

def risk_page() -> rx.Component:
    """Risk dashboard page with sidebar"""
    content = rx.vstack(
        # Header
        rx.hstack(
            rx.heading("Risk Dashboard", size="8", color="#a855f7"),
            rx.spacer(),
            rx.button(
                "Calculate Risk",
                on_click=RiskState.calculate_portfolio_risk,
                loading=RiskState.loading,
                color_scheme="purple",
                size="3",
            ),
            width="100%",
            padding="2rem",
        ),
        
        # Primary Risk Metrics
        rx.grid(
            risk_metric_card("VaR (95%)", RiskState.var_95_display, "red"),
            risk_metric_card("CVaR (95%)", RiskState.cvar_95_display, "orange"),
            risk_metric_card("Max Drawdown", RiskState.max_drawdown_display, "blue"),
            risk_metric_card("Sharpe", RiskState.sharpe_ratio_display, "purple"),
            columns="4",
            spacing="4",
            width="100%",
            padding="0 2rem",
        ),
        
        # Greeks Exposure
        rx.card(
            rx.vstack(
                rx.heading("Greeks Exposure", size="5"),
                rx.grid(
                    greek_card("Delta", RiskState.total_delta),
                    greek_card("Gamma", RiskState.total_gamma),
                    greek_card("Theta", RiskState.total_theta),
                    greek_card("Vega", RiskState.total_vega),
                    columns="4",
                    spacing="3",
                ),
            ),
            width="95%",
            margin="2rem",
            style={
                "background": "rgba(45, 27, 61, 0.8)",
                "border": "1px solid #4c1d95",
            }
        ),
        
        # Charts Grid
        rx.grid(
            # VaR History Chart
            rx.card(
                rx.vstack(
                    rx.heading("VaR History", size="5"),
                    rx.recharts.line_chart(
                        rx.recharts.line(data_key="var_95", stroke="#ef4444", name="VaR 95%"),
                        rx.recharts.line(data_key="var_99", stroke="#f97316", name="VaR 99%"),
                        rx.recharts.x_axis(data_key="date"),
                        rx.recharts.y_axis(),
                        rx.recharts.tooltip(),
                        data=RiskState.var_history_data,
                        height=250,
                    ),
                ),
                style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            ),
            
            # Drawdown Chart
            rx.card(
                rx.vstack(
                    rx.heading("Drawdown History", size="5"),
                    rx.recharts.area_chart(
                        rx.recharts.area(
                            data_key="drawdown",
                            fill="#f87171",
                            stroke="#ef4444",
                        ),
                        rx.recharts.x_axis(data_key="date"),
                        rx.recharts.y_axis(),
                        rx.recharts.tooltip(),
                        data=RiskState.drawdown_data,
                        height=250,
                    ),
                ),
                style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            ),
            
            columns="2",
            spacing="4",
            width="95%",
            margin="0 2rem",
        ),
        
        # Additional analytics can be added here (e.g., correlation matrix)
        
        width="100%",
    )
    return page_layout(content, "Risk Dashboard")

def risk_score_banner() -> rx.Component:
    """Risk score display banner"""
    return rx.card(
        rx.hstack(
            rx.vstack(
                rx.text("Overall Risk Score", size="3", color="#9ca3af"),
                rx.heading(RiskState.risk_score_display, size="8"),
                align="start",
            ),
            rx.spacer(),
            rx.vstack(
                rx.text("Risk Level", size="3", color="#9ca3af"),
                rx.badge(
                    RiskState.risk_level,
                    color_scheme=RiskState.risk_color,
                    size="3",
                ),
                align="end",
            ),
            width="100%",
            padding="1rem",
        ),
        width="95%",
        margin="0 2rem 2rem 2rem",
        style={
            "background": f"linear-gradient(90deg, rgba(168, 85, 247, 0.1), rgba(239, 68, 68, 0.1))",
            "border": "1px solid #4c1d95",
        }
    )

def risk_metric_card(label: str, value: str, color: str) -> rx.Component:
    """Risk metric card"""
    return rx.card(
        rx.vstack(
            rx.text(label, size="2", color=f"var(--{color}-9)"),
            rx.text(value, size="5", weight="bold"),
            align="center",
        ),
        style={
            "background": "rgba(45, 27, 61, 0.8)",
            "border": f"1px solid var(--{color}-9)",
            "min_height": "100px",
        }
    )

def greek_card(name: str, value) -> rx.Component:
    """Greek exposure card (reactive-safe)."""
    # Use rx.cond with a reactive comparison; avoid Python boolean on Var.
    color_var = rx.cond(value > 0, "var(--green-9)", "var(--red-9)")
    return rx.box(
        rx.vstack(
            rx.text(name, size="2", color="#9ca3af"),
            # Display raw reactive value; formatting can be handled in state if desired.
            rx.text(value, size="4", weight="bold", color=color_var),
            align="center",
        ),
        padding="1rem",
        style={
            "background": "rgba(31, 27, 36, 0.5)",
            "border_radius": "0.5rem",
        }
    )

def risk_alert(alert: Dict) -> rx.Component:
    """Risk alert item"""
    colors = {
        "info": "blue",
        "warning": "yellow",
        "danger": "red"
    }
    color = colors.get(alert["level"], "gray")
    
    return rx.hstack(
        rx.icon(
            rx.cond(
                alert["level"] == "warning",
                "alert-triangle",
                "info"
            ),
            size=16,
            color=f"var(--{color}-9)",
        ),
        rx.text(alert["message"], size="2"),
        rx.spacer(),
        rx.text(alert["timestamp"], size="1", color="#9ca3af"),
        width="100%",
        padding="0.75rem",
        style={
            "background": f"var(--{color}-3)",
            "border_left": f"3px solid var(--{color}-9)",
            "border_radius": "0.25rem",
        }
    )
