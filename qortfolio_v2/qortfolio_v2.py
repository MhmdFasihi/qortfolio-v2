"""Qortfolio V2 Main Application"""

import reflex as rx
from .pages.options_analytics import options_analytics_page
from .pages.volatility import volatility_page
from .pages.portfolio import portfolio_page
from .pages.risk import risk_page
from .components.navigation import page_layout
from .state import State, OptionsState
from .volatility_state import VolatilityState
from .portfolio_state import PortfolioState
from .risk_state import RiskState

def index() -> rx.Component:
    """Main dashboard page with sidebar"""
    content = rx.center(
        rx.vstack(
            rx.vstack(
                rx.heading("Welcome to Qortfolio V2", size="9", color="#a855f7"),
                rx.text("Professional Quantitative Finance Platform", size="5", color="#9ca3af"),
                spacing="2",
            ),
            
            rx.vstack(
                rx.text("Quick Stats", size="3", color="#9ca3af", margin_bottom="1rem"),
                rx.grid(
                    stat_card("Active Positions", "12", "briefcase"),
                    stat_card("Total Value", "$45,230", "dollar-sign"),
                    stat_card("Today's P&L", "+$1,250", "trending-up"),
                    stat_card("Risk Score", "65/100", "shield"),
                    columns="4",
                    spacing="4",
                    width="100%",
                ),
                margin_top="3rem",
            ),
            
            rx.vstack(
                rx.text("Quick Actions", size="3", color="#9ca3af", margin_bottom="1rem"),
                rx.grid(
                    rx.link(
                        rx.button(
                            rx.icon("trending-up", size=20),
                            "Options Analytics", 
                            color_scheme="purple", 
                            size="3",
                            width="100%",
                        ),
                        href="/options",
                    ),
                    rx.link(
                        rx.button(
                            rx.icon("activity", size=20),
                            "Volatility Analysis", 
                            color_scheme="purple", 
                            size="3",
                            width="100%",
                        ),
                        href="/volatility",
                    ),
                    rx.link(
                        rx.button(
                            rx.icon("briefcase", size=20),
                            "Portfolio", 
                            color_scheme="purple", 
                            size="3",
                            width="100%",
                        ),
                        href="/portfolio",
                    ),
                    rx.link(
                        rx.button(
                            rx.icon("shield", size=20),
                            "Risk Dashboard", 
                            color_scheme="purple", 
                            size="3",
                            width="100%",
                        ),
                        href="/risk",
                    ),
                    columns="2",
                    spacing="4",
                    width="60%",
                ),
                margin_top="3rem",
            ),
            
            spacing="5",
            max_width="1200px",
        ),
        height="100vh",
    )
    
    return page_layout(content, "Dashboard")

def stat_card(label: str, value: str, icon: str) -> rx.Component:
    """Statistics card for dashboard"""
    return rx.card(
        rx.hstack(
            rx.icon(icon, size=30, color="#a855f7"),
            rx.vstack(
                rx.text(label, size="2", color="#9ca3af"),
                rx.text(value, size="5", weight="bold"),
                align="start",
                spacing="1",
            ),
            spacing="3",
        ),
        style={
            "background": "rgba(45, 27, 61, 0.8)",
            "border": "1px solid #4c1d95",
            "padding": "1.5rem",
        }
    )

# Create app
app = rx.App(
    theme=rx.theme(
        appearance="dark",
        accent_color="purple",
    )
)

# Add pages
app.add_page(index, route="/", title="Qortfolio V2 Dashboard")
app.add_page(options_analytics_page, route="/options", title="Options Analytics")
app.add_page(volatility_page, route="/volatility", title="Volatility Analysis")
app.add_page(portfolio_page, route="/portfolio", title="Portfolio Management")
app.add_page(risk_page, route="/risk", title="Risk Dashboard")
