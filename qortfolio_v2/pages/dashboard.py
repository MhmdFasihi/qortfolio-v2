import reflex as rx
from ..state import State

def dashboard_page() -> rx.Component:
    """Main dashboard page"""
    return rx.center(
        rx.vstack(
            rx.heading("Qortfolio V2 Dashboard", size="8", color="#a855f7"),
            rx.text("Professional Quantitative Finance Platform", size="4", color="#9ca3af"),
            rx.link(
                rx.button("Go to Options Analytics", color_scheme="purple"),
                href="/options",
            ),
            spacing="4",
        ),
        height="100vh",
        style={
            "background": "linear-gradient(135deg, #1a0033 0%, #220044 50%, #1a0033 100%)",
        }
    )
