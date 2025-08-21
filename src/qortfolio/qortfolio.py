# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Main Reflex application for Qortfolio V2.
This is the entry point for the dashboard with dark royal purple theme.
"""

import reflex as rx
from typing import Dict, List

# Dark Royal Purple Theme Configuration
theme_config = {
    "appearance": "dark",
    "radius": "medium",
    "accent_color": "purple",
}

# Custom color palette
colors = {
    "primary": "#4C1D95",      # Dark purple
    "secondary": "#7C3AED",    # Medium purple
    "background": "#0F0A17",   # Very dark purple/black
    "surface": "#1A1625",      # Dark purple surface
    "text": "#E5E7EB",         # Light gray text
    "accent": "#A855F7",       # Bright purple accent
    "success": "#10B981",      # Green
    "warning": "#F59E0B",      # Orange
    "error": "#EF4444",        # Red
}

class QortfolioState(rx.State):
    """Main application state."""
    
    # Navigation
    current_page: str = "dashboard"
    
    # Data states
    is_loading: bool = False
    selected_currency: str = "BTC"
    currencies: List[str] = ["BTC", "ETH"]
    
    # Connection status
    mongodb_connected: bool = False
    redis_connected: bool = False
    deribit_connected: bool = False
    
    def check_connections(self):
        """Check all service connections."""
        # This will be implemented with actual connection checks
        pass

def navbar() -> rx.Component:
    """Navigation bar component."""
    return rx.box(
        rx.hstack(
            rx.heading(
                "Qortfolio V2",
                size="7",
                weight="bold",
                color=colors["accent"],
            ),
            rx.spacer(),
            rx.hstack(
                rx.link(
                    rx.button(
                        "Dashboard",
                        variant="ghost",
                        color_scheme="purple",
                    ),
                    href="/",
                ),
                rx.link(
                    rx.button(
                        "Options",
                        variant="ghost",
                        color_scheme="purple",
                    ),
                    href="/options",
                ),
                rx.link(
                    rx.button(
                        "Portfolio",
                        variant="ghost",
                        color_scheme="purple",
                    ),
                    href="/portfolio",
                ),
                rx.link(
                    rx.button(
                        "Risk",
                        variant="ghost",
                        color_scheme="purple",
                    ),
                    href="/risk",
                ),
                spacing="4",
            ),
            width="100%",
            padding="4",
        ),
        bg=colors["surface"],
        border_bottom=f"1px solid {colors['primary']}",
        width="100%",
    )

def index() -> rx.Component:
    """Main dashboard page."""
    return rx.vstack(
        navbar(),
        rx.container(
            rx.vstack(
                rx.heading(
                    "Quantitative Finance Platform",
                    size="8",
                    weight="bold",
                    color=colors["text"],
                ),
                rx.text(
                    "Professional crypto options analytics with real-time data",
                    size="4",
                    color=colors["text"],
                    opacity=0.8,
                ),
                rx.divider(margin_y="4"),
                
                # Status cards
                rx.grid(
                    rx.card(
                        rx.vstack(
                            rx.text("MongoDB", weight="bold"),
                            rx.cond(
                                QortfolioState.mongodb_connected,
                                rx.text("Connected", color=colors["success"]),
                                rx.text("Disconnected", color=colors["error"]),
                            ),
                        ),
                    ),
                    rx.card(
                        rx.vstack(
                            rx.text("Redis Cache", weight="bold"),
                            rx.cond(
                                QortfolioState.redis_connected,
                                rx.text("Connected", color=colors["success"]),
                                rx.text("Disconnected", color=colors["error"]),
                            ),
                        ),
                    ),
                    rx.card(
                        rx.vstack(
                            rx.text("Deribit API", weight="bold"),
                            rx.cond(
                                QortfolioState.deribit_connected,
                                rx.text("Connected", color=colors["success"]),
                                rx.text("Disconnected", color=colors["error"]),
                            ),
                        ),
                    ),
                    columns="3",
                    spacing="4",
                    width="100%",
                ),
                
                spacing="6",
                width="100%",
            ),
            max_width="1200px",
            padding="6",
        ),
        bg=colors["background"],
        min_height="100vh",
        width="100%",
    )

# Create the app
app = rx.App(
    theme=rx.theme(**theme_config),
    stylesheets=[],
)

# Add pages
app.add_page(index, route="/", title="Qortfolio V2 - Dashboard")

# This will be expanded with more pages as we develop
