"""Navigation Components for Qortfolio V2"""

import reflex as rx
from typing import List, Tuple

def sidebar() -> rx.Component:
    """Main navigation sidebar"""
    menu_items = [
        ("Dashboard", "/", "home"),
        ("Options Analytics", "/options", "trending-up"),
        ("Volatility Analysis", "/volatility", "activity"),
        ("Portfolio", "/portfolio", "briefcase"),
        ("Risk Dashboard", "/risk", "shield"),
    ]
    
    return rx.box(
        rx.vstack(
            # Logo/Title
            rx.link(
                rx.hstack(
                    rx.heading("Qortfolio", size="5", color="#a855f7"),
                    rx.badge("V2", color_scheme="purple"),
                    spacing="2",
                ),
                href="/",
                style={"text_decoration": "none"},
            ),
            rx.divider(margin="1rem 0"),
            
            # Menu Items
            rx.vstack(
                *[nav_item(name, path, icon) for name, path, icon in menu_items],
                spacing="2",
                width="100%",
            ),
            
            rx.spacer(),
            
            # Bottom section
            rx.vstack(
                rx.divider(),
                rx.hstack(
                    rx.icon("settings", size=20),
                    rx.text("Settings", size="3"),
                    padding="0.75rem",
                    width="100%",
                    _hover={"background": "rgba(168, 85, 247, 0.1)", "border_radius": "0.5rem"},
                ),
                width="100%",
            ),
            
            height="100%",
            padding="1.5rem",
            align="start",
        ),
        position="fixed",
        left="0",
        top="0",
        height="100vh",
        width="250px",
        style={
            "background": "rgba(31, 27, 36, 0.95)",
            "border_right": "1px solid #4c1d95",
            "backdrop_filter": "blur(10px)",
            "z_index": "100",
        }
    )

def nav_item(name: str, path: str, icon: str) -> rx.Component:
    """Individual navigation item"""
    return rx.link(
        rx.hstack(
            rx.icon(icon, size=20),
            rx.text(name, size="3"),
            padding="0.75rem",
            width="100%",
            border_radius="0.5rem",
            _hover={"background": "rgba(168, 85, 247, 0.2)"},
            style={
                "transition": "all 0.2s",
            }
        ),
        href=path,
        style={"text_decoration": "none", "color": "#e5e7eb"},
        width="100%",
    )

def page_layout(content: rx.Component, title: str = "") -> rx.Component:
    """Standard page layout with sidebar"""
    return rx.hstack(
        sidebar(),
        rx.box(
            content,
            margin_left="250px",
            width="calc(100% - 250px)",
            min_height="100vh",
            style={
                "background": "linear-gradient(135deg, #1a0033 0%, #220044 50%, #1a0033 100%)",
            }
        ),
        spacing="0",
        width="100%",
    )
