"""Enhanced Options Analytics Page with Volatility Surface and Advanced Analytics"""

import reflex as rx
from ..state import OptionsState
from ..components.navigation import page_layout

def options_analytics_page() -> rx.Component:
    """Options analytics page with sidebar, expiry filter, calls/puts tables."""
    content = rx.vstack(
        # Header
        rx.hstack(
            rx.heading("Options Analytics", size="8", color="#a855f7"),
            rx.spacer(),
            rx.button(
                "Refresh Data",
                on_click=OptionsState.fetch_options_data,
                loading=OptionsState.loading,
                color_scheme="purple",
                size="3",
            ),
            width="100%",
            padding="2rem",
        ),
        
        # Currency / Expiry / Status / Auto-refresh
        rx.hstack(
            rx.text("Currency:", size="4", weight="bold"),
            rx.select(["BTC", "ETH"], value=OptionsState.selected_currency, on_change=OptionsState.set_currency),
            rx.text("Expiry:", size="4", weight="bold", margin_left="2rem"),
            rx.select(OptionsState.expiry_options, value=OptionsState.selected_expiry, on_change=OptionsState.set_expiry, placeholder="Select expiry"),
            rx.text("Auto:", size="3", margin_left="2rem"),
            rx.switch(is_checked=OptionsState.auto_refresh, on_change=OptionsState.toggle_auto_refresh),
            rx.select(["15","30","60","120"], value=OptionsState.refresh_seconds.to_string(), on_change=OptionsState.set_refresh_seconds, width="80px"),
            rx.spacer(),
            rx.text("Spot:", color="#9ca3af"),
            rx.text(OptionsState.spot_display),
            rx.text("DB:", color="#9ca3af", margin_left="1rem"),
            rx.text(OptionsState.db_status, color="#a855f7"),
            spacing="4",
            padding="1rem 2rem",
        ),
        
        # Enhanced Metrics Cards
        rx.grid(
            metric_card("Total Contracts", OptionsState.total_contracts, "briefcase"),
            metric_card("Avg IV", OptionsState.avg_iv_pct, "activity"),
            metric_card("Max OI", OptionsState.max_oi, "users"),
            metric_card("Volume", OptionsState.total_volume, "bar-chart-3"),
            advanced_metric_card("Call/Put Ratio", OptionsState.call_put_ratio_display, "shuffle"),
            advanced_metric_card("Max Pain", OptionsState.max_pain_display, "target"),
            advanced_metric_card("Gamma Exp", OptionsState.gamma_exposure_display, "trending-up"),
            advanced_metric_card("IV Rank", OptionsState.iv_rank_display, "percent"),
            columns="4",
            spacing="4",
            width="100%",
            padding="0 2rem",
        ),

        # Analytics Tabs (using compatible Reflex tabs)
        rx.tabs(
            rx.tabs.list(
                rx.tabs.trigger("Options Chain", value="chain"),
                rx.tabs.trigger("Volatility Surface", value="surface"),
                rx.tabs.trigger("Flow Analysis", value="flow"),
                rx.tabs.trigger("Greeks Analysis", value="greeks"),
            ),
            rx.tabs.content(
                # Options Chain Tab
                options_chain_content(),
                value="chain"
            ),
            rx.tabs.content(
                # Volatility Surface Tab
                volatility_surface_content(),
                value="surface"
            ),
            rx.tabs.content(
                # Flow Analysis Tab
                flow_analysis_content(),
                value="flow"
            ),
            rx.tabs.content(
                # Greeks Analysis Tab
                greeks_analysis_content(),
                value="greeks"
            ),
            default_value="chain",
            width="100%",
            padding="0 2rem",
        ),
        
        width="100%",
    )
    
    return page_layout(content, "Options Analytics")


def options_chain_content() -> rx.Component:
    """Options chain tab content with calls/puts tables."""
    return rx.vstack(
        # Enhanced Calls / Puts tables
        rx.grid(
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.icon("trending-up", color="#22c55e"),
                        rx.heading("Calls", size="5"),
                        rx.spacer(),
                        rx.badge(OptionsState.calls_count, color_scheme="green"),
                        width="100%",
                    ),
                    rx.cond(
                        OptionsState.loading,
                        rx.center(rx.spinner(color="purple", size="3")),
                        rx.cond(
                            OptionsState.calls_data,
                            rx.data_table(
                                data=OptionsState.calls_data,
                                columns=[
                                    {"name": "Strike", "id": "strike"},
                                    {"name": "Bid", "id": "bid"},
                                    {"name": "Ask", "id": "ask"},
                                    {"name": "Mark", "id": "mark_price"},
                                    {"name": "IV", "id": "iv"},
                                    {"name": "Delta", "id": "delta"},
                                    {"name": "Gamma", "id": "gamma"},
                                    {"name": "Vol", "id": "volume"},
                                    {"name": "OI", "id": "open_interest"},
                                ],
                                pagination=True,
                                page_size=15,
                            ),
                            rx.text("No Calls.", color="#9ca3af"),
                        ),
                    ),
                ),
                style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            ),
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.icon("trending-down", color="#ef4444"),
                        rx.heading("Puts", size="5"),
                        rx.spacer(),
                        rx.badge(OptionsState.puts_count, color_scheme="red"),
                        width="100%",
                    ),
                    rx.cond(
                        OptionsState.loading,
                        rx.center(rx.spinner(color="purple", size="3")),
                        rx.cond(
                            OptionsState.puts_data,
                            rx.data_table(
                                data=OptionsState.puts_data,
                                columns=[
                                    {"name": "Strike", "id": "strike"},
                                    {"name": "Bid", "id": "bid"},
                                    {"name": "Ask", "id": "ask"},
                                    {"name": "Mark", "id": "mark_price"},
                                    {"name": "IV", "id": "iv"},
                                    {"name": "Delta", "id": "delta"},
                                    {"name": "Gamma", "id": "gamma"},
                                    {"name": "Vol", "id": "volume"},
                                    {"name": "OI", "id": "open_interest"},
                                ],
                                pagination=True,
                                page_size=15,
                            ),
                            rx.text("No Puts.", color="#9ca3af"),
                        ),
                    ),
                ),
                style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            ),
            columns="2",
            spacing="4",
            width="100%",
        ),
        spacing="4",
        width="100%",
    )


def volatility_surface_content() -> rx.Component:
    """Volatility surface visualization tab."""
    return rx.vstack(
        rx.grid(
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.icon("activity"),
                        rx.heading("3D Volatility Surface", size="5"),
                        rx.spacer(),
                        rx.button(
                            "Update Surface",
                            on_click=OptionsState.build_volatility_surface,
                            loading=OptionsState.surface_loading,
                            size="2",
                            color_scheme="purple",
                        ),
                        width="100%",
                    ),
                    rx.cond(
                        OptionsState.surface_loading,
                        rx.center(rx.spinner(color="purple", size="3")),
                        rx.cond(
                            OptionsState.volatility_surface_data,
                            rx.text("Volatility surface ready (Plotly rendering deferred)", color="#a855f7"),
                            rx.text("No surface data available", color="#9ca3af"),
                        ),
                    ),
                ),
                style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            ),
            rx.card(
                rx.vstack(
                    rx.heading("Term Structure", size="5"),
                    rx.cond(
                        OptionsState.atm_term_structure_data,
                        rx.recharts.line_chart(
                            rx.recharts.line(data_key="iv", stroke="#a855f7"),
                            rx.recharts.x_axis(data_key="expiry"),
                            rx.recharts.y_axis(),
                            rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
                            rx.recharts.tooltip(),
                            data=OptionsState.atm_term_structure_data,
                            height="300px",
                        ),
                        rx.text("No term structure data", color="#9ca3af"),
                    ),
                ),
                style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            ),
            columns="1",
            spacing="4",
            width="100%",
        ),
        spacing="4",
        width="100%",
    )


def flow_analysis_content() -> rx.Component:
    """Options flow analysis tab."""
    return rx.vstack(
        rx.grid(
            rx.card(
                rx.vstack(
                    rx.heading("Flow Direction", size="5"),
                    rx.hstack(
                        rx.icon("trending-up", size="6", color="#22c55e"),
                        rx.vstack(
                            rx.text("Current Flow", size="2", color="#9ca3af"),
                            rx.text(OptionsState.flow_direction, size="4", weight="bold"),
                            align="start",
                        ),
                        rx.spacer(),
                        rx.vstack(
                            rx.text("Confidence", size="2", color="#9ca3af"),
                            rx.text(OptionsState.flow_confidence, size="4", weight="bold"),
                            align="start",
                        ),
                    ),
                ),
                style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            ),
            rx.card(
                rx.vstack(
                    rx.heading("Unusual Activity", size="5"),
                    rx.cond(
                        OptionsState.unusual_activity,
                        rx.data_table(
                            data=OptionsState.unusual_activity,
                            columns=[
                                {"name": "Symbol", "id": "symbol"},
                                {"name": "Type", "id": "option_type"},
                                {"name": "Strike", "id": "strike"},
                                {"name": "Volume", "id": "volume"},
                                {"name": "OI", "id": "open_interest"},
                                {"name": "Reason", "id": "reason"},
                            ],
                            pagination=True,
                            page_size=10,
                        ),
                        rx.text("No unusual activity detected", color="#9ca3af"),
                    ),
                ),
                style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            ),
            columns="1",
            spacing="4",
            width="100%",
        ),
        spacing="4",
        width="100%",
    )


def greeks_analysis_content() -> rx.Component:
    """Greeks analysis tab."""
    return rx.vstack(
        rx.grid(
            greeks_metric_card("Total Delta", OptionsState.total_delta_display, "triangle"),
            greeks_metric_card("Total Gamma", OptionsState.total_gamma_display, "square"),
            greeks_metric_card("Total Theta", OptionsState.total_theta_display, "clock"),
            greeks_metric_card("Total Vega", OptionsState.total_vega_display, "activity"),
            columns="4",
            spacing="4",
            width="100%",
        ),
        rx.card(
            rx.vstack(
                rx.heading("Gamma Exposure Profile", size="5"),
                rx.cond(
                    OptionsState.gamma_profile_data,
                    rx.recharts.area_chart(
                        rx.recharts.area(data_key="gamma_exposure", fill="#a855f7", stroke="#a855f7"),
                        rx.recharts.x_axis(data_key="price"),
                        rx.recharts.y_axis(),
                        rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
                        rx.recharts.tooltip(),
                        data=OptionsState.gamma_profile_data,
                        height="300px",
                    ),
                    rx.text("No gamma profile data", color="#9ca3af"),
                ),
            ),
            style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            width="100%",
        ),
        spacing="4",
        width="100%",
    )


def metric_card(label: str, value: rx.Var, icon: str = "bar-chart") -> rx.Component:
    """Enhanced metric display card with icon."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon(icon, size=24, color="#a855f7"),
                rx.spacer(),
            ),
            rx.text(label, size="2", color="#a855f7"),
            rx.text(value, size="5", weight="bold"),
            align="center",
            spacing="1",
        ),
        style={
            "background": "rgba(45, 27, 61, 0.8)",
            "border": "1px solid #4c1d95",
            "min_height": "120px",
        }
    )


def advanced_metric_card(label: str, value: rx.Var, icon: str) -> rx.Component:
    """Advanced metric card with more sophisticated styling."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon(icon, size=24, color="#7c3aed"),
                rx.spacer(),
            ),
            rx.text(label, size="2", color="#7c3aed", weight="medium"),
            rx.text(value, size="4", weight="bold", color="#ffffff"),
            align="center",
            spacing="1",
        ),
        style={
            "background": "linear-gradient(135deg, rgba(45, 27, 61, 0.9), rgba(124, 58, 237, 0.1))",
            "border": "1px solid #7c3aed",
            "min_height": "120px",
            "transition": "all 0.3s ease",
        }
    )


def greeks_metric_card(label: str, value: rx.Var, icon: str) -> rx.Component:
    """Greeks-specific metric card."""
    return rx.card(
        rx.vstack(
            rx.icon(icon, size=32, color="#a855f7"),
            rx.text(label, size="3", color="#a855f7", weight="medium"),
            rx.text(value, size="6", weight="bold"),
            align="center",
        ),
        style={
            "background": "rgba(45, 27, 61, 0.8)",
            "border": "1px solid #4c1d95",
            "min_height": "140px",
        }
    )
