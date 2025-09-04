"""
Qortfolio V2 - Main Reflex Dashboard Application
Professional Quantitative Finance Platform with Crypto Options Analytics
"""

import reflex as rx
from .state import State
from .pages import dashboard, options_analytics

# Configure the app with dark royal purple theme
app = rx.App(
    theme=rx.theme(
        appearance="dark",
        has_background=True,
        radius="medium",
        accent_color="purple",
    ),
    style={
        "background": "linear-gradient(135deg, #1a0033 0%, #220044 50%, #1a0033 100%)",
        "min_height": "100vh",
        "color": "#e2e8f0",
        "font_family": "'Inter', sans-serif",
    },
)

# Add pages
app.add_page(dashboard.dashboard_page, route="/", title="Qortfolio V2 - Dashboard")
app.add_page(options_analytics.options_page, route="/options", title="Options Analytics")
