"""Navigation Components for Qortfolio V2"""

import reflex as rx
from typing import List, Tuple
from ..state import OptionsState

def sidebar() -> rx.Component:
    """Main navigation sidebar with collapse functionality"""
    menu_items = [
        ("Dashboard", "/", "home"),
        ("Options Analytics", "/options", "trending-up"),
        ("Volatility Analysis", "/volatility", "activity"),
        ("Portfolio", "/portfolio", "briefcase"),
        ("Risk Dashboard", "/risk", "shield"),
    ]

    return rx.box(
        rx.vstack(
            # Logo/Title - Responsive
            rx.cond(
                OptionsState.sidebar_visible,
                rx.link(
                    rx.hstack(
                        rx.heading("Qortfolio", size="5", color="#a855f7"),
                        rx.badge("V2", color_scheme="purple"),
                        spacing="2",
                    ),
                    href="/",
                    style={"text_decoration": "none"},
                ),
                rx.link(
                    rx.hstack(
                        rx.heading("Q", size="5", color="#a855f7"),
                        spacing="2",
                    ),
                    href="/",
                    style={"text_decoration": "none"},
                ),
            ),
            rx.divider(margin="1rem 0"),

            # Menu Items - Responsive
            rx.vstack(
                *[nav_item(name, path, icon, OptionsState.sidebar_visible) for name, path, icon in menu_items],
                spacing="2",
                width="100%",
            ),

            rx.spacer(),

            # Bottom section - Responsive
            rx.vstack(
                rx.divider(),
                rx.cond(
                    OptionsState.sidebar_visible,
                    rx.hstack(
                        rx.icon("settings", size=20),
                        rx.text("Settings", size="3"),
                        padding="0.75rem",
                        width="100%",
                        _hover={"background": "rgba(168, 85, 247, 0.1)", "border_radius": "0.5rem"},
                    ),
                    rx.hstack(
                        rx.icon("settings", size=20),
                        padding="0.75rem",
                        width="100%",
                        justify="center",
                        _hover={"background": "rgba(168, 85, 247, 0.1)", "border_radius": "0.5rem"},
                    ),
                ),
                width="100%",
            ),

            height="100%",
            padding=rx.cond(OptionsState.sidebar_visible, "1.5rem", "0.75rem"),
            align="start",
        ),
        position="fixed",
        left="0",
        top="0",
        height="100vh",
        width=rx.cond(OptionsState.sidebar_visible, "250px", "70px"),
        style={
            "background": "rgba(31, 27, 36, 0.95)",
            "border_right": "1px solid #4c1d95",
            "backdrop_filter": "blur(10px)",
            "z_index": "100",
            "transition": "all 0.3s ease",
        }
    )

def nav_item(name: str, path: str, icon: str, visible: rx.Var) -> rx.Component:
    """Individual navigation item with responsive design"""
    return rx.link(
        rx.cond(
            visible,
            # Full navigation item
            rx.hstack(
                rx.icon(icon, size=20),
                rx.text(name, size="3"),
                padding="0.75rem",
                width="100%",
                border_radius="0.5rem",
                _hover={"background": "rgba(168, 85, 247, 0.2)"},
                style={"transition": "all 0.2s"},
            ),
            # Collapsed navigation item (icon only)
            rx.hstack(
                rx.icon(icon, size=20),
                padding="0.75rem",
                width="100%",
                justify="center",
                border_radius="0.5rem",
                _hover={"background": "rgba(168, 85, 247, 0.2)"},
                style={"transition": "all 0.2s"},
                title=name,  # Tooltip for collapsed state
            ),
        ),
        href=path,
        style={"text_decoration": "none", "color": "#e5e7eb"},
        width="100%",
    )

def page_layout(content: rx.Component, title: str = "") -> rx.Component:
    """Standard page layout with responsive sidebar"""
    return rx.hstack(
        sidebar(),
        rx.box(
            # Header with sidebar toggle
            rx.hstack(
                rx.button(
                    rx.icon("menu", size=20),
                    on_click=OptionsState.toggle_sidebar,
                    color_scheme="purple",
                    variant="ghost",
                    size="2",
                ),
                rx.spacer(),
                rx.cond(
                    title != "",
                    rx.heading(title, size="6", color="#e5e7eb"),
                    rx.text(""),
                ),
                width="100%",
                padding="1rem 2rem",
                align="center",
                style={
                    "border_bottom": "1px solid #4c1d95",
                    "background": "rgba(31, 27, 36, 0.8)",
                    "backdrop_filter": "blur(10px)",
                }
            ),
            content,
            margin_left=rx.cond(OptionsState.sidebar_visible, "250px", "70px"),
            width=rx.cond(OptionsState.sidebar_visible, "calc(100% - 250px)", "calc(100% - 70px)"),
            min_height="100vh",
            style={
                "background": "linear-gradient(135deg, #1a0033 0%, #220044 50%, #1a0033 100%)",
                "transition": "all 0.3s ease",
            }
        ),
        spacing="0",
        width="100%",
    )
