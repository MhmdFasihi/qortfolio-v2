"""Qortfolio V2 Main Application"""

import reflex as rx
from .pages.options_analytics import options_analytics_page
from .state import State, OptionsState

def index() -> rx.Component:
    """Main dashboard page"""
    return rx.center(
        rx.vstack(
            rx.heading("Qortfolio V2 Dashboard", size="8", color="#a855f7"),
            rx.text("Professional Quantitative Finance Platform", size="4", color="#9ca3af"),
            rx.link(
                rx.button("Go to Options Analytics", color_scheme="purple", size="3"),
                href="/options",
            ),
            spacing="5",
        ),
        height="100vh",
        style={
            "background": "linear-gradient(135deg, #1a0033 0%, #220044 50%, #1a0033 100%)",
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
app.add_page(index, route="/", title="Qortfolio V2")
app.add_page(
    options_analytics_page,
    route="/options",
    title="Options Analytics",
    on_load=OptionsState.fetch_options_data,
)
