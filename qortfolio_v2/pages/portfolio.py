"""Portfolio Management Page"""

import reflex as rx
from ..portfolio_state import PortfolioState
from ..components.navigation import page_layout

def portfolio_page() -> rx.Component:
    """Portfolio management page with sidebar and charts"""
    content = rx.vstack(
        # Header
        rx.hstack(
            rx.heading("Portfolio Management", size="8", color="#a855f7"),
            rx.spacer(),
            rx.button(
                "Refresh Data",
                on_click=PortfolioState.update_current_prices,  # This now does both refresh and price update
                loading=PortfolioState.loading,
                color_scheme="purple",
                size="3",
            ),
            width="100%",
            padding="2rem",
        ),
        
        # Portfolio Metrics
        rx.grid(
            portfolio_metric_card("Total Portfolio Value", PortfolioState.total_value_display, "purple", "dollar-sign", "Current market value of all positions"),
            portfolio_metric_card("Total Profit & Loss", PortfolioState.pnl_display, "green", "trending-up", "Unrealized gains/losses from entry prices"),
            portfolio_metric_card("Portfolio Return %", PortfolioState.pnl_percent_display, "blue", "percent", "Percentage return since inception"),
            portfolio_metric_card("Sharpe Ratio", PortfolioState.sharpe_display, "orange", "target", "Risk-adjusted return metric"),
            columns="4",
            spacing="4",
            width="100%",
            padding="0 2rem",
        ),
        
        # Portfolio Selector / Create / Delete / Benchmark
        rx.hstack(
            rx.text("Portfolio:", size="3"),
            rx.select(PortfolioState.portfolio_options, value=PortfolioState.selected_portfolio, on_change=PortfolioState.set_portfolio),
            rx.input(placeholder="New portfolio", value=PortfolioState.new_portfolio_name, on_change=PortfolioState.set_new_portfolio_name, width="200px"),
            rx.button("Create", on_click=PortfolioState.create_portfolio, size="2", color_scheme="green"),
            rx.button("Delete", on_click=PortfolioState.delete_current_portfolio, size="2", color_scheme="red"),
            rx.spacer(),
            rx.text("Benchmark:", size="3"),
            rx.select(PortfolioState.benchmark_options, value=PortfolioState.selected_benchmark, on_change=PortfolioState.set_benchmark),
            spacing="3",
            padding="0 2rem",
            width="100%",
        ),

        # View Selector & Filters
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
            rx.button(
                "Optimization",
                on_click=PortfolioState.set_view("optimization"),
                color_scheme=rx.cond(
                    PortfolioState.selected_view == "optimization", "purple", "gray"
                ),
                variant=rx.cond(
                    PortfolioState.selected_view == "optimization", "solid", "outline"
                ),
            ),
            rx.spacer(),
            rx.text("Asset Filter:", size="3"),
            rx.select(["all", "spot", "contracts"], value=PortfolioState.selected_asset_type, on_change=PortfolioState.set_asset_type),
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
                rx.cond(
                    PortfolioState.selected_view == "optimization",
                    optimization_view(),
                    performance_view()
                )
            )
        ),
        
        width="100%",
    )
    return page_layout(content, "Portfolio Management")

def portfolio_metric_card(label: str, value: rx.Var, color: str, icon: str = "bar-chart", description: str = "") -> rx.Component:
    """Portfolio metric card with icon and description"""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon(icon, size=20, color=f"var(--{color}-9)"),
                rx.spacer(),
                width="100%",
            ),
            rx.text(label, size="2", color=f"var(--{color}-9)", weight="medium"),
            rx.text(value, size="6", weight="bold"),
            rx.text(description, size="1", color="#9ca3af", text_align="center") if description else rx.box(),
            align="center",
            spacing="2",
        ),
        style={
            "background": "rgba(45, 27, 61, 0.8)",
            "border": f"1px solid var(--{color}-9)",
            "min_height": "120px",
            "padding": "1rem",
        }
    )

def positions_view() -> rx.Component:
    """Positions table view"""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.heading("Current Positions", size="5"),
                rx.spacer(),
                rx.button(
                    "Update Prices",
                    on_click=PortfolioState.update_current_prices,
                    color_scheme="blue",
                    size="2",
                    loading=PortfolioState.loading,
                ),
                justify="between",
                width="100%",
            ),
            # Asset Selection Section
            rx.hstack(
                rx.text("Asset Type:", size="3"),
                rx.select(
                    ["Spot", "Options", "Futures"],
                    value=PortfolioState.selected_position_type,
                    on_change=PortfolioState.set_position_type,
                    placeholder="Select Type",
                    width="120px"
                ),
                rx.text("Sector:", size="3"),
                rx.select(
                    ["All", "Infrastructure", "DeFi", "AI", "Gaming", "Stablecoin"],
                    value=PortfolioState.selected_sector_filter,
                    on_change=PortfolioState.set_sector_filter,
                    placeholder="Filter by Sector",
                    width="130px"
                ),
                rx.text("Asset:", size="3"),
                rx.cond(
                    PortfolioState.selected_position_type == "Options",
                    # For options, only show BTC and ETH
                    rx.select(
                        ["BTC", "ETH"],
                        value=PortfolioState.trade_symbol,
                        on_change=PortfolioState.set_trade_symbol,
                        placeholder="Choose Asset",
                        width="160px"
                    ),
                    # For other types, show all filtered assets
                    rx.select(
                        PortfolioState.filtered_assets,
                        value=PortfolioState.trade_symbol,
                        on_change=PortfolioState.set_trade_symbol,
                        placeholder="Choose Asset",
                        width="160px"
                    ),
                ),
                spacing="2",
                width="100%",
            ),

            # Input fields - dynamic based on position type
            rx.cond(
                PortfolioState.selected_position_type == "Options",
                # Options input fields
                rx.vstack(
                    rx.hstack(
                        rx.vstack(
                            rx.text("Quantity:", size="2"),
                            rx.input(placeholder="1", type="number", value=PortfolioState.trade_quantity, on_change=PortfolioState.set_trade_quantity, width="100px"),
                            spacing="1",
                        ),
                        rx.vstack(
                            rx.text("Strike:", size="2"),
                            rx.input(placeholder="50000", type="number", value=PortfolioState.options_strike, on_change=PortfolioState.set_options_strike, width="120px"),
                            spacing="1",
                        ),
                        rx.vstack(
                            rx.text("Expiry:", size="2"),
                            rx.input(placeholder="2024-03-29", type="date", value=PortfolioState.options_expiry, on_change=PortfolioState.set_options_expiry, width="140px"),
                            spacing="1",
                        ),
                        rx.vstack(
                            rx.text("Type:", size="2"),
                            rx.select(["call", "put"], value=PortfolioState.options_type, on_change=PortfolioState.set_options_type, width="80px"),
                            spacing="1",
                        ),
                        spacing="2",
                    ),
                    rx.hstack(
                        rx.vstack(
                            rx.text(f"Premium ({PortfolioState.trade_symbol}):", size="2"),
                            rx.input(placeholder="0.001", type="number", value=PortfolioState.options_premium, on_change=PortfolioState.set_options_premium, width="120px"),
                            spacing="1",
                        ),
                        rx.vstack(
                            rx.text("Side:", size="2"),
                            rx.select(["buy", "sell"], value=PortfolioState.trade_side, on_change=PortfolioState.set_trade_side, width="80px"),
                            spacing="1",
                        ),
                        rx.vstack(
                            rx.text("", size="2"),  # Empty label for button alignment
                            rx.button("Add Option", on_click=PortfolioState.add_options_position, color_scheme="purple"),
                            spacing="1",
                        ),
                        spacing="2",
                        align="end",
                    ),
                    spacing="2",
                ),
                # Spot/other input fields
                rx.hstack(
                    rx.vstack(
                        rx.text("Quantity:", size="2"),
                        rx.input(placeholder="0.00", type="number", value=PortfolioState.trade_quantity, on_change=PortfolioState.set_trade_quantity, width="120px"),
                        spacing="1",
                    ),
                    rx.vstack(
                        rx.text("Price (optional):", size="2"),
                        rx.input(placeholder="0.00", type="number", value=PortfolioState.trade_price, on_change=PortfolioState.set_trade_price, width="160px"),
                        spacing="1",
                    ),
                    rx.vstack(
                        rx.text("Side:", size="2"),
                        rx.select(["buy", "sell"], value=PortfolioState.trade_side, on_change=PortfolioState.set_trade_side, width="120px"),
                        spacing="1",
                    ),
                    rx.vstack(
                        rx.text("", size="2"),  # Empty label for button alignment
                        rx.button("Add Position", on_click=PortfolioState.add_spot_position, color_scheme="purple"),
                        spacing="1",
                    ),
                    spacing="3",
                    align="end",
                ),
            ),
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
    return rx.vstack(
        # Summary Cards
        rx.grid(
            portfolio_metric_card("Total Portfolio Value", PortfolioState.total_value_display, "purple", "wallet", "Current total portfolio value"),
            portfolio_metric_card("Total Assets", PortfolioState.positions.length(), "blue", "layers", "Number of different assets"),
            portfolio_metric_card("Largest Position", rx.cond(PortfolioState.crypto_allocation, PortfolioState.crypto_allocation[0]["name"], "None"), "green", "trending-up", "Asset with highest allocation"),
            columns="3",
            spacing="4",
            width="100%",
            margin="0 2rem",
        ),

        # Charts
        rx.grid(
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.heading("Asset Allocation", size="5"),
                        rx.button(
                            "Refresh",
                            on_click=PortfolioState.refresh,
                            size="1",
                            color_scheme="purple",
                        ),
                        justify="between",
                        width="100%",
                    ),
                    rx.cond(
                        PortfolioState.crypto_allocation,
                        rx.recharts.pie_chart(
                            rx.recharts.pie(
                                data=PortfolioState.crypto_allocation,
                                data_key="value",
                                name_key="name",
                                fill="#8884d8",
                                label=True,
                            ),
                            rx.recharts.tooltip(),
                            height=280,
                        ),
                        rx.vstack(
                            rx.text("No allocation data available", size="4", color="#9ca3af"),
                            rx.text("Add positions to see allocation breakdown", size="2", color="#6b7280"),
                            height="280px",
                            justify="center",
                            align="center",
                        )
                    ),
                ),
                style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            ),
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.heading("Sector Allocation", size="5"),
                        rx.text("Active sectors", size="2", color="#9ca3af"),
                        justify="between",
                        width="100%",
                    ),
                    rx.cond(
                        PortfolioState.sector_allocation,
                        rx.recharts.bar_chart(
                            rx.recharts.bar(
                                data_key="value",
                                fill="#a855f7",
                            ),
                            rx.recharts.x_axis(data_key="name"),
                            rx.recharts.y_axis(),
                            rx.recharts.tooltip(),
                            data=PortfolioState.sector_allocation,
                            height=280,
                        ),
                        rx.vstack(
                            rx.text("No sector data available", size="4", color="#9ca3af"),
                            rx.text("Positions will be categorized by crypto sectors", size="2", color="#6b7280"),
                            height="280px",
                            justify="center",
                            align="center",
                        )
                    ),
                ),
                style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            ),
            columns="2",
            spacing="4",
            width="95%",
            margin="1rem 2rem",
        ),

        # Detailed Tables
        rx.card(
            rx.vstack(
                rx.heading("Detailed Allocation", size="5"),
                rx.tabs.root(
                    rx.tabs.list(
                        rx.tabs.trigger("By Asset", value="assets"),
                        rx.tabs.trigger("By Sector", value="sectors"),
                    ),
                    rx.tabs.content(
                        rx.cond(
                            PortfolioState.crypto_allocation,
                            rx.data_table(
                                data=PortfolioState.crypto_allocation,
                                columns=[
                                    {"name": "Asset", "id": "name"},
                                    {"name": "Allocation %", "id": "value"},
                                ],
                            ),
                            rx.text("No asset allocation data", color="#9ca3af")
                        ),
                        value="assets",
                    ),
                    rx.tabs.content(
                        rx.cond(
                            PortfolioState.sector_allocation,
                            rx.data_table(
                                data=PortfolioState.sector_allocation,
                                columns=[
                                    {"name": "Sector", "id": "name"},
                                    {"name": "Allocation %", "id": "value"},
                                ],
                            ),
                            rx.text("No sector allocation data", color="#9ca3af")
                        ),
                        value="sectors",
                    ),
                    default_value="assets",
                ),
            ),
            width="95%",
            margin="1rem 2rem",
            style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
        ),
    )

def performance_view() -> rx.Component:
    """Performance metrics view"""
    return rx.vstack(
        # Performance Overview
        rx.card(
            rx.vstack(
                rx.hstack(
                    rx.heading("Performance Overview", size="5"),
                    rx.spacer(),
                    rx.vstack(
                        rx.text("Portfolio:", size="2", color="#9ca3af"),
                        rx.select(
                            PortfolioState.portfolio_options,
                            value=PortfolioState.selected_portfolio,
                            on_change=PortfolioState.set_portfolio,
                            placeholder="Select Portfolio",
                            width="150px",
                            size="1"
                        ),
                        spacing="1",
                    ),
                    rx.button(
                        "Refresh",
                        on_click=PortfolioState.refresh,
                        size="1",
                        color_scheme="purple",
                    ),
                    justify="between",
                    width="100%",
                ),
                rx.grid(
                    perf_metric("Daily Return", PortfolioState.daily_return.to(str) + "%", "Average daily portfolio return"),
                    perf_metric("Sharpe Ratio", PortfolioState.sharpe_display, "Risk-adjusted return measure"),
                    perf_metric("Max Drawdown", PortfolioState.max_drawdown.to(str) + "%", "Largest peak-to-trough decline"),
                    perf_metric("Win Rate", PortfolioState.win_rate.to(str) + "%", "Percentage of profitable periods"),
                    columns="2",
                    spacing="4",
                ),
                # Debug info
                rx.cond(
                    PortfolioState.sharpe_ratio == 0.0,
                    rx.text("Note: Performance metrics require portfolio history data", size="2", color="#f59e0b"),
                    rx.text("Performance calculated from portfolio history", size="2", color="#10b981"),
                ),
            ),
            width="95%",
            margin="0 2rem",
            style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
        ),

        # Portfolio vs Benchmark Chart
        rx.card(
            rx.vstack(
                rx.hstack(
                    rx.heading("Portfolio vs Benchmark", size="5"),
                    rx.spacer(),
                    rx.vstack(
                        rx.text("Benchmark:", size="2", color="#9ca3af"),
                        rx.select(
                            PortfolioState.benchmark_options,
                            value=PortfolioState.selected_benchmark,
                            on_change=PortfolioState.set_benchmark,
                            placeholder="Select Benchmark",
                            width="120px",
                            size="1"
                        ),
                        spacing="1",
                    ),
                    rx.vstack(
                        rx.text("Period:", size="2", color="#9ca3af"),
                        rx.select(
                            ["7d", "30d", "90d", "1y", "all"],
                            value=PortfolioState.selected_period,
                            on_change=PortfolioState.set_period,
                            placeholder="Period",
                            width="80px",
                            size="1"
                        ),
                        spacing="1",
                    ),
                    rx.button(
                        "Add Benchmark",
                        on_click=PortfolioState.add_custom_benchmark,
                        color_scheme="blue",
                        size="1"
                    ),
                    width="100%",
                ),
                rx.cond(
                    PortfolioState.portfolio_vs_btc,
                    rx.recharts.line_chart(
                        rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
                        rx.recharts.line(data_key="portfolio_pct", stroke="#a855f7", name="Portfolio (%)", dot=False),
                        rx.recharts.line(data_key="benchmark_pct", stroke="#22d3ee", name="Benchmark (%)", dot=False),
                        rx.recharts.x_axis(data_key="date"),
                        rx.recharts.y_axis(),
                        rx.recharts.tooltip(),
                        rx.recharts.legend(),
                        data=PortfolioState.portfolio_vs_btc,
                        height=300,
                    ),
                    rx.vstack(
                        rx.text("No performance data available", size="4", color="#9ca3af"),
                        rx.text("Portfolio needs historical price data to calculate performance", size="2", color="#6b7280"),
                        rx.text("Add positions and wait for data collection or use sample data", size="2", color="#6b7280"),
                        height="300px",
                        justify="center",
                        align="center",
                    )
                ),
            ),
            width="95%",
            margin="1rem 2rem",
            style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
        ),

        # Portfolio Statistics
        rx.card(
            rx.vstack(
                rx.heading("Portfolio Statistics", size="5"),
                rx.grid(
                    rx.vstack(
                        rx.text("Total Positions", size="2", color="#9ca3af"),
                        rx.text(PortfolioState.positions.length(), size="5", weight="bold"),
                        spacing="1",
                    ),
                    rx.vstack(
                        rx.text("Total P&L", size="2", color="#9ca3af"),
                        rx.text("$" + PortfolioState.total_pnl.to(str), size="5", weight="bold", color=rx.cond(PortfolioState.total_pnl >= 0, "#22c55e", "#ef4444")),
                        spacing="1",
                    ),
                    rx.vstack(
                        rx.text("Portfolio Value", size="2", color="#9ca3af"),
                        rx.text("$" + PortfolioState.total_value.to(str), size="5", weight="bold"),
                        spacing="1",
                    ),
                    rx.vstack(
                        rx.text("Return %", size="2", color="#9ca3af"),
                        rx.text(PortfolioState.total_pnl_percent.to(str) + "%", size="5", weight="bold", color=rx.cond(PortfolioState.total_pnl_percent >= 0, "#22c55e", "#ef4444")),
                        spacing="1",
                    ),
                    columns="4",
                    spacing="4",
                ),
            ),
            width="95%",
            margin="1rem 2rem 2rem 2rem",
            style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
        ),
    )

def optimization_view() -> rx.Component:
    """Advanced portfolio optimization view with HRP, HERC, and Mean-Variance"""
    return rx.vstack(
        # Optimization Controls
        rx.card(
            rx.vstack(
                rx.hstack(
                    rx.heading("Portfolio Optimization", size="5"),
                    rx.spacer(),
                    rx.text("Status:", size="3", color="#9ca3af"),
                    rx.text(PortfolioState.optimization_status, size="3", weight="bold", color="#a855f7"),
                    width="100%",
                ),

                # Portfolio Definition Section
                rx.vstack(
                    rx.text("Portfolio Definition", size="4", weight="bold"),
                    rx.hstack(
                        rx.vstack(
                            rx.text("Current Portfolio:", size="3"),
                            rx.select(
                                PortfolioState.portfolio_options,
                                value=PortfolioState.selected_portfolio,
                                on_change=PortfolioState.set_portfolio,
                                placeholder="Select Portfolio",
                                width="160px"
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            rx.text("New Portfolio Name:", size="3"),
                            rx.input(
                                value=PortfolioState.new_portfolio_name,
                                on_change=PortfolioState.set_new_portfolio_name,
                                placeholder="Enter name...",
                                width="140px"
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            rx.text("Actions:", size="3"),
                            rx.hstack(
                                rx.button(
                                    "Create",
                                    on_click=PortfolioState.create_portfolio,
                                    color_scheme="green",
                                    size="2",
                                    disabled=PortfolioState.new_portfolio_name == ""
                                ),
                                rx.button(
                                    "Delete",
                                    on_click=PortfolioState.delete_current_portfolio,
                                    color_scheme="red",
                                    size="2",
                                    disabled=PortfolioState.selected_portfolio == "default"
                                ),
                                spacing="2"
                            ),
                            spacing="1",
                        ),
                        spacing="3",
                        align="end",
                    ),
                    rx.text("Portfolio status: " + PortfolioState.selected_portfolio + " portfolio selected",
                           size="2", color="#9ca3af"),
                    spacing="2",
                ),

                # Asset Selection Section
                rx.vstack(
                    rx.text("Asset Selection", size="4", weight="bold"),
                    rx.hstack(
                        rx.vstack(
                            rx.text("Asset Type:", size="3"),
                            rx.select(
                                ["Spot"],
                                value="Spot",
                                on_change=PortfolioState.set_position_type,
                                placeholder="Select Type",
                                width="120px",
                                disabled=True
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            rx.text("Sector:", size="3"),
                            rx.select(
                                ["All", "Infrastructure", "DeFi", "AI", "Gaming", "Stablecoin"],
                                value=PortfolioState.selected_sector_filter,
                                on_change=PortfolioState.set_sector_filter,
                                placeholder="Filter by Sector",
                                width="130px"
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            rx.text("Available Assets:", size="3"),
                            rx.select(
                                PortfolioState.filtered_assets,
                                placeholder="Choose Asset",
                                width="160px",
                                on_change=PortfolioState.add_asset_to_optimization,
                            ),
                            spacing="1",
                        ),
                        spacing="3",
                        align="end",
                    ),
                    # Manual Asset Addition
                    rx.hstack(
                        rx.vstack(
                            rx.text("Add Custom Asset:", size="3"),
                            rx.input(
                                value=PortfolioState.custom_asset_input,
                                on_change=PortfolioState.set_custom_asset_input,
                                placeholder="Enter asset symbol (e.g. BTC-USD)",
                                width="180px"
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            rx.text("", size="3"),  # Empty for alignment
                            rx.button(
                                "Add Asset",
                                on_click=PortfolioState.add_custom_asset_to_optimization,
                                color_scheme="green",
                                size="2",
                                disabled=PortfolioState.custom_asset_input == ""
                            ),
                            spacing="1",
                        ),
                        spacing="3",
                        align="end",
                    ),
                    # Selected Assets Display and Management
                    rx.vstack(
                        rx.hstack(
                            rx.text("Selected Assets:", size="3", weight="bold"),
                            rx.button(
                                "Clear All",
                                on_click=PortfolioState.clear_optimization_assets,
                                color_scheme="red",
                                size="1",
                            ),
                            spacing="2",
                            align="center",
                        ),
                        rx.cond(
                            PortfolioState.optimization_assets.length() > 0,
                            rx.hstack(
                                rx.foreach(
                                    PortfolioState.optimization_assets,
                                    lambda asset: rx.badge(
                                        rx.hstack(
                                            rx.text(asset, size="2"),
                                            rx.button(
                                                "×",
                                                on_click=PortfolioState.remove_optimization_asset(asset),
                                                color_scheme="red",
                                                size="1",
                                                variant="ghost",
                                            ),
                                            spacing="1",
                                            align="center",
                                        ),
                                        color_scheme="green",
                                        variant="soft",
                                    ),
                                ),
                                wrap="wrap",
                                spacing="1",
                            ),
                            rx.text("No assets selected for optimization", size="2", color="gray"),
                        ),
                        spacing="2",
                        width="100%",
                    ),
                    spacing="2",
                    width="100%",
                ),

                rx.grid(
                    rx.vstack(
                        rx.text("Optimization Method", size="3", weight="bold"),
                        rx.select(
                            PortfolioState.optimization_methods,
                            value=PortfolioState.optimization_method,
                            on_change=PortfolioState.set_optimization_method,
                        ),
                        spacing="2",
                    ),
                    rx.vstack(
                        rx.text("Risk-Free Rate", size="3", weight="bold"),
                        rx.input(
                            value=PortfolioState.risk_free_rate,
                            on_change=lambda v: PortfolioState.set_optimization_parameter("risk_free_rate", v),
                            type="number",
                            placeholder="0.05",
                        ),
                        spacing="2",
                    ),
                    rx.vstack(
                        rx.text("Lookback Days", size="3", weight="bold"),
                        rx.input(
                            value=PortfolioState.lookback_days,
                            on_change=lambda v: PortfolioState.set_optimization_parameter("lookback_days", v),
                            type="number",
                            placeholder="365",
                        ),
                        spacing="2",
                    ),
                    rx.vstack(
                        rx.hstack(
                            rx.button(
                                "Optimize Portfolio",
                                on_click=PortfolioState.run_portfolio_optimization,
                                loading=PortfolioState.optimization_loading,
                                color_scheme="purple",
                                size="3",
                            ),
                            rx.button(
                                "Compare Methods",
                                on_click=PortfolioState.run_multi_method_comparison,
                                loading=PortfolioState.optimization_loading,
                                color_scheme="green",
                                size="3",
                            ),
                            spacing="3",
                        ),
                        spacing="2",
                    ),
                    columns="4",
                    spacing="4",
                    width="100%",
                ),
                spacing="4",
            ),
            width="95%",
            margin="0 2rem",
            style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
        ),

        # Constraints Configuration
        rx.card(
            rx.vstack(
                rx.heading("Optimization Constraints", size="5"),

                # Global Asset Constraints
                rx.hstack(
                    rx.vstack(
                        rx.text("Global Min Weight:", size="3"),
                        rx.input(
                            value=PortfolioState.min_asset_weight,
                            on_change=lambda v: PortfolioState.set_constraint("min_asset_weight", v),
                            type="number",
                            placeholder="0.01",
                            width="100px",
                        ),
                        spacing="1",
                    ),
                    rx.vstack(
                        rx.text("Global Max Weight:", size="3"),
                        rx.input(
                            value=PortfolioState.max_asset_weight,
                            on_change=lambda v: PortfolioState.set_constraint("max_asset_weight", v),
                            type="number",
                            placeholder="0.40",
                            width="100px",
                        ),
                        spacing="1",
                    ),
                    spacing="4",
                ),

                # Individual Asset Constraints
                rx.vstack(
                    rx.text("Individual Asset Constraints", size="4", weight="bold"),
                    rx.hstack(
                        rx.vstack(
                            rx.text("Asset:", size="3"),
                            rx.select(
                                PortfolioState.optimization_assets,
                                value=PortfolioState.constraint_asset,
                                on_change=PortfolioState.set_constraint_asset,
                                placeholder="Select Asset",
                                width="150px",
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            rx.text("Min Weight:", size="3"),
                            rx.input(
                                value=PortfolioState.constraint_min_weight,
                                on_change=PortfolioState.set_constraint_min_weight,
                                type="number",
                                placeholder="0.00",
                                width="100px",
                                step="0.01"
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            rx.text("Max Weight:", size="3"),
                            rx.input(
                                value=PortfolioState.constraint_max_weight,
                                on_change=PortfolioState.set_constraint_max_weight,
                                type="number",
                                placeholder="0.20",
                                width="100px",
                                step="0.01"
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            rx.text("", size="3"),  # Empty label for alignment
                            rx.button(
                                "Add Constraint",
                                on_click=PortfolioState.add_asset_constraint,
                                color_scheme="blue",
                                size="2"
                            ),
                            spacing="1",
                        ),
                        spacing="3",
                        align="end",
                    ),
                    spacing="2",
                ),

                # Sector Constraints
                rx.vstack(
                    rx.text("Sector Constraints", size="4", weight="bold"),
                    rx.hstack(
                        rx.vstack(
                            rx.text("Sector:", size="3"),
                            rx.select(
                                ["Infrastructure", "DeFi", "AI", "Gaming", "Stablecoin", "Layer2", "Privacy", "Storage", "Oracle", "Exchange", "Payments", "Meme"],
                                value=PortfolioState.constraint_sector,
                                on_change=PortfolioState.set_constraint_sector,
                                placeholder="Select Sector",
                                width="150px",
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            rx.text("Min Weight:", size="3"),
                            rx.input(
                                value=PortfolioState.constraint_sector_min,
                                on_change=PortfolioState.set_constraint_sector_min,
                                type="number",
                                placeholder="0.00",
                                width="100px",
                                step="0.01"
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            rx.text("Max Weight:", size="3"),
                            rx.input(
                                value=PortfolioState.constraint_sector_max,
                                on_change=PortfolioState.set_constraint_sector_max,
                                type="number",
                                placeholder="0.25",
                                width="100px",
                                step="0.01"
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            rx.text("", size="3"),  # Empty label for alignment
                            rx.button(
                                "Add Constraint",
                                on_click=PortfolioState.add_sector_constraint,
                                color_scheme="green",
                                size="2"
                            ),
                            spacing="1",
                        ),
                        spacing="3",
                        align="end",
                    ),
                    spacing="2",
                ),

                # Current Constraints Display
                rx.vstack(
                    rx.hstack(
                        rx.text("Active Constraints", size="4", weight="bold"),
                        rx.button(
                            "Clear All",
                            on_click=PortfolioState.clear_all_constraints,
                            color_scheme="red",
                            size="1"
                        ),
                        spacing="2",
                        align="center",
                    ),
                    rx.cond(
                        PortfolioState.active_constraints.length() > 0,
                        rx.vstack(
                            rx.foreach(
                                PortfolioState.active_constraints,
                                lambda constraint, index: rx.hstack(
                                    rx.text(constraint["display"], size="2"),
                                    rx.button(
                                        "×",
                                        on_click=lambda: PortfolioState.remove_constraint(index),
                                        color_scheme="red",
                                        size="1",
                                        width="30px"
                                    ),
                                    justify="between",
                                    width="100%",
                                    padding="0.5rem",
                                    style={"background": "rgba(255,255,255,0.1)", "border-radius": "5px"}
                                )
                            ),
                            spacing="1",
                            width="100%"
                        ),
                        rx.text("No active constraints", size="2", color="#9ca3af")
                    ),
                    spacing="2",
                ),

                # Optimization Execution
                rx.vstack(
                    rx.text("Run Optimization", size="4", weight="bold"),
                    rx.hstack(
                        rx.vstack(
                            rx.text("Method:", size="3"),
                            rx.select(
                                PortfolioState.optimization_methods,
                                value=PortfolioState.optimization_method,
                                on_change=PortfolioState.set_optimization_method,
                                placeholder="Select Method",
                                width="140px"
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            rx.text("Risk Free Rate:", size="3"),
                            rx.input(
                                value=PortfolioState.risk_free_rate,
                                on_change=lambda v: PortfolioState.set_optimization_parameter("risk_free_rate", v),
                                type="number",
                                placeholder="0.05",
                                width="100px"
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            rx.text("", size="3"),  # Empty label for alignment
                            rx.button(
                                rx.cond(
                                    PortfolioState.optimization_loading,
                                    "Optimizing...",
                                    "Optimize Portfolio"
                                ),
                                on_click=PortfolioState.run_portfolio_optimization,
                                color_scheme="purple",
                                size="3",
                                disabled=PortfolioState.optimization_loading,
                                width="140px"
                            ),
                            spacing="1",
                        ),
                        spacing="3",
                        align="end",
                    ),
                    spacing="2",
                ),

                spacing="4",
            ),
            width="95%",
            margin="1rem 2rem",
            style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
        ),

        # Optimization Results
        rx.grid(
            # Current Allocation vs Suggested
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.heading("Optimization Results", size="5"),
                        rx.spacer(),
                        rx.cond(
                            PortfolioState.current_optimization_result,
                            rx.text("Optimization method completed",
                                   size="3", color="#a855f7"),
                            rx.text("No optimization results", size="3", color="#9ca3af")
                        ),
                        width="100%",
                    ),
                    rx.cond(
                        PortfolioState.suggested_allocation,
                        rx.vstack(
                            # Optimization metrics
                            rx.grid(
                                rx.vstack(
                                    rx.text("Expected Return", size="2", color="#9ca3af"),
                                    rx.text(PortfolioState.current_optimization_result.get("expected_return", "0.0%"),
                                           size="4", weight="bold", color="#22c55e"),
                                    spacing="1",
                                ),
                                rx.vstack(
                                    rx.text("Volatility", size="2", color="#9ca3af"),
                                    rx.text(PortfolioState.current_optimization_result.get("volatility", "0.0%"),
                                           size="4", weight="bold", color="#f59e0b"),
                                    spacing="1",
                                ),
                                rx.vstack(
                                    rx.text("Sharpe Ratio", size="2", color="#9ca3af"),
                                    rx.text(PortfolioState.current_optimization_result.get("sharpe_ratio", "0.0"),
                                           size="4", weight="bold", color="#3b82f6"),
                                    spacing="1",
                                ),
                                columns="3",
                                spacing="4",
                                width="100%",
                            ),
                            # Allocation table
                            rx.text("Suggested Allocation", size="4", weight="bold", color="#22c55e", margin_top="1rem"),
                            rx.data_table(
                                data=PortfolioState.suggested_allocation,
                                columns=[
                                    {"Header": "Asset", "accessor": "asset"},
                                    {"Header": "Weight %", "accessor": "weight"}
                                ],
                                pagination=False,
                            ),
                            spacing="3",
                        ),
                        rx.vstack(
                            rx.text("No optimization results yet", size="4", color="#9ca3af"),
                            rx.text("Click 'Optimize Portfolio' button above to generate allocation suggestions",
                                   size="2", color="#6b7280"),
                            spacing="2",
                        ),
                    ),
                ),
                style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            ),

            # Optimization Comparison
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.heading("Method Comparison", size="5"),
                        rx.spacer(),
                        rx.button(
                            rx.cond(
                                PortfolioState.optimization_loading,
                                "Comparing...",
                                "Compare Methods"
                            ),
                            on_click=PortfolioState.run_multi_method_comparison,
                            color_scheme="blue",
                            size="2",
                            disabled=PortfolioState.optimization_loading,
                        ),
                        width="100%",
                    ),
                    rx.cond(
                        PortfolioState.comparison_results.length() > 0,
                        rx.vstack(
                            rx.text("Optimization Methods Performance", size="3", color="#9ca3af"),
                            rx.data_table(
                                data=PortfolioState.comparison_results,
                                columns=[
                                    {"Header": "Method", "accessor": "method"},
                                    {"Header": "Expected Return", "accessor": "expected_return"},
                                    {"Header": "Volatility", "accessor": "volatility"},
                                    {"Header": "Sharpe Ratio", "accessor": "sharpe_ratio"},
                                    {"Header": "Max Drawdown", "accessor": "max_drawdown"},
                                    {"Header": "Sortino Ratio", "accessor": "sortino_ratio"}
                                ],
                                pagination=False,
                            ),
                            spacing="2",
                        ),
                        rx.vstack(
                            rx.text("No comparison results yet", size="4", color="#9ca3af"),
                            rx.text("Click 'Compare Methods' to run all optimization methods and see which performs best",
                                   size="2", color="#6b7280"),
                            spacing="2",
                        ),
                    ),
                ),
                style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            ),

            columns="2",
            spacing="4",
            width="95%",
            margin="1rem 2rem",
        ),

        # Efficient Frontier Chart
        rx.cond(
            PortfolioState.efficient_frontier_data,
            rx.card(
                rx.vstack(
                    rx.heading("Efficient Frontier", size="5"),
                    rx.recharts.scatter_chart(
                        rx.recharts.scatter(
                            data=PortfolioState.efficient_frontier_data,
                            data_key="return",
                            fill="#a855f7",
                            name="Frontier",
                        ),
                        rx.recharts.scatter(
                            data=[PortfolioState.current_portfolio_point],
                            data_key="return",
                            fill="#22c55e",
                            name="Current Portfolio",
                        ),
                        rx.recharts.x_axis(data_key="risk", name="Risk (Volatility)"),
                        rx.recharts.y_axis(data_key="return", name="Return"),
                        rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
                        rx.recharts.tooltip(),
                        height=400,
                        width="100%",
                    ),
                ),
                width="95%",
                margin="1rem 2rem 2rem 2rem",
                style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            ),
            rx.box(),  # Empty box when no frontier data
        ),

        spacing="4",
        width="100%",
    )

def perf_metric(label: str, value: str, description: str = "") -> rx.Component:
    """Performance metric display with description"""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(label, size="3", color="#9ca3af", weight="medium"),
                rx.spacer(),
                rx.text(value, size="4", weight="bold"),
                width="100%",
            ),
            rx.text(description, size="1", color="#6b7280") if description else rx.box(),
            spacing="1",
        ),
        padding="1rem",
        style={
            "background": "rgba(31, 27, 36, 0.5)",
            "border_radius": "0.5rem",
            "border": "1px solid rgba(107, 114, 128, 0.2)",
        }
    )
