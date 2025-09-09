"""Portfolio Management Page"""

import reflex as rx
from ..portfolio_state import PortfolioState

def portfolio_page() -> rx.Component:
    """Portfolio management page"""
    return rx.vstack(
        # Header
        rx.hstack(
            rx.heading("Portfolio Management", size="8", color="#a855f7"),
            rx.spacer(),
            rx.button(
                "Refresh",
                on_click=PortfolioState.fetch_portfolio_data,
                loading=PortfolioState.loading,
                color_scheme="purple",
                size="3",
            ),
            width="100%",
            padding="2rem",
        ),
        
        # Portfolio Metrics
        rx.grid(
            portfolio_metric_card("Total Value", PortfolioState.total_value_display, "purple"),
            portfolio_metric_card("Total P&L", PortfolioState.pnl_display, "green"),
            portfolio_metric_card("Return", PortfolioState.pnl_percent_display, "blue"),
            portfolio_metric_card("Sharpe Ratio", PortfolioState.sharpe_display, "orange"),
            columns="4",
            spacing="4",
            width="100%",
            padding="0 2rem",
        ),
        
        # View Selector
        rx.hstack(
            rx.button(
                "Positions",
                on_click=PortfolioState.set_view("positions"),
                color_scheme=rx.cond(
                    PortfolioState.selected_view == "positions", "purple", "gray"
                ),
                variant=rx.cond(
                    PortfolioState.selected_view == "positions", "solid", "outline"
                ),
            ),
            rx.button(
                "Allocation",
                on_click=PortfolioState.set_view("allocation"),
                color_scheme=rx.cond(
                    PortfolioState.selected_view == "allocation", "purple", "gray"
                ),
                variant=rx.cond(
                    PortfolioState.selected_view == "allocation", "solid", "outline"
                ),
            ),
            rx.button(
                "Performance",
                on_click=PortfolioState.set_view("performance"),
                color_scheme=rx.cond(
                    PortfolioState.selected_view == "performance", "purple", "gray"
                ),
                variant=rx.cond(
                    PortfolioState.selected_view == "performance", "solid", "outline"
                ),
            ),
            spacing="2",
            padding="1rem 2rem",
        ),
        
        # Content Area
        rx.cond(
            PortfolioState.selected_view == "positions",
            positions_view(),
            rx.cond(
                PortfolioState.selected_view == "allocation",
                allocation_view(),
                performance_view()
            )
        ),
        
        width="100%",
        style={
            "background": "linear-gradient(135deg, #1a0033 0%, #220044 50%, #1a0033 100%)",
            "min_height": "100vh",
        },
    )

def portfolio_metric_card(label: str, value: rx.Var, color: str) -> rx.Component:
    """Portfolio metric card"""
    return rx.card(
        rx.vstack(
            rx.text(label, size="2", color=f"var(--{color}-9)"),
            rx.text(value, size="6", weight="bold"),
            align="center",
        ),
        style={
            "background": "rgba(45, 27, 61, 0.8)",
            "border": f"1px solid var(--{color}-9)",
            "min_height": "100px",
        }
    )

def positions_view() -> rx.Component:
    """Positions table view"""
    return rx.card(
        rx.vstack(
            rx.heading("Current Positions", size="5"),
            rx.data_table(
                data=PortfolioState.positions,
                columns=[
                    {"name": "Symbol", "id": "symbol"},
                    {"name": "Type", "id": "type"},
                    {"name": "Quantity", "id": "quantity"},
                    {"name": "Entry", "id": "entry_price"},
                    {"name": "Current", "id": "current_price"},
                    {"name": "Value", "id": "value"},
                    {"name": "P&L", "id": "pnl"},
                    {"name": "P&L %", "id": "pnl_percent"},
                    {"name": "Allocation %", "id": "allocation"},
                ],
            ),
        ),
        width="95%",
        margin="0 2rem",
        style={
            "background": "rgba(45, 27, 61, 0.8)",
            "border": "1px solid #4c1d95",
        }
    )

def allocation_view() -> rx.Component:
    """Allocation charts view"""
    return rx.grid(
        rx.card(
            rx.vstack(
                rx.heading("Crypto Allocation", size="5"),
                rx.recharts.pie_chart(
                    rx.recharts.pie(
                        data=PortfolioState.crypto_allocation,
                        data_key="value",
                        name_key="name",
                        fill="#8884d8",
                        label=True,
                    ),
                    rx.recharts.tooltip(),
                    height=300,
                ),
            ),
            style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
        ),
        rx.card(
            rx.vstack(
                rx.heading("Sector Allocation", size="5"),
                rx.recharts.bar_chart(
                    rx.recharts.bar(
                        data_key="value",
                        fill="#a855f7",
                    ),
                    rx.recharts.x_axis(data_key="name"),
                    rx.recharts.y_axis(),
                    rx.recharts.tooltip(),
                    data=PortfolioState.sector_allocation,
                    height=300,
                ),
            ),
            style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
        ),
        columns="2",
        spacing="4",
        width="95%",
        margin="0 2rem",
    )

def performance_view() -> rx.Component:
    """Performance metrics view"""
    return rx.card(
        rx.vstack(
            rx.heading("Performance Metrics", size="5"),
            rx.grid(
                perf_metric("Daily Return", f"{PortfolioState.daily_return:.2f}%"),
                perf_metric("Sharpe Ratio", f"{PortfolioState.sharpe_ratio:.2f}"),
                perf_metric("Max Drawdown", f"{PortfolioState.max_drawdown:.2f}%"),
                perf_metric("Win Rate", f"{PortfolioState.win_rate:.1f}%"),
                columns="2",
                spacing="4",
            ),
        ),
        width="95%",
        margin="0 2rem",
        style={
            "background": "rgba(45, 27, 61, 0.8)",
            "border": "1px solid #4c1d95",
        }
    )

def perf_metric(label: str, value: str) -> rx.Component:
    """Performance metric display"""
    return rx.box(
        rx.hstack(
            rx.text(label, size="3", color="#9ca3af"),
            rx.spacer(),
            rx.text(value, size="4", weight="bold"),
            width="100%",
        ),
        padding="1rem",
        style={
            "background": "rgba(31, 27, 36, 0.5)",
            "border_radius": "0.5rem",
        }
    )
