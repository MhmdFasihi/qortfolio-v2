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
                "Refresh",
                on_click=PortfolioState.refresh,
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
        
        # Portfolio Selector / Create / Delete / Benchmark
        rx.hstack(
            rx.text("Portfolio:", size="3"),
            rx.select(PortfolioState.portfolio_options, value=PortfolioState.selected_portfolio, on_change=PortfolioState.set_portfolio),
            rx.input(placeholder="New portfolio", value=PortfolioState.new_portfolio_name, on_change=lambda v: setattr(PortfolioState, 'new_portfolio_name', v), width="200px"),
            rx.button("Create", on_click=PortfolioState.create_portfolio, size="2", color_scheme="green"),
            rx.button("Delete", on_click=lambda: PortfolioState.delete_portfolio(PortfolioState.selected_portfolio), size="2", color_scheme="red"),
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
            rx.text("Assets:", size="3"),
            rx.select(["all", "spot", "options"], value=PortfolioState.selected_asset_type, on_change=PortfolioState.set_asset_type),
            rx.text("Timeframe:", size="3"),
            rx.select(["7d", "30d", "90d", "1y"], value=PortfolioState.selected_period, on_change=PortfolioState.set_period),
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
            rx.hstack(
                rx.input(placeholder="Symbol (e.g., BTC)", value=PortfolioState.trade_symbol, on_change=PortfolioState.set_trade_symbol, width="160px"),
                rx.input(placeholder="Quantity", type="number", value=PortfolioState.trade_quantity, on_change=PortfolioState.set_trade_quantity, width="120px"),
                rx.input(placeholder="Price (optional)", type="number", value=PortfolioState.trade_price, on_change=PortfolioState.set_trade_price, width="160px"),
                rx.select(["buy", "sell"], value=PortfolioState.trade_side, on_change=PortfolioState.set_trade_side, width="120px"),
                rx.button("Submit", on_click=PortfolioState.add_spot_position, color_scheme="purple"),
                spacing="2",
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
        rx.card(
            rx.vstack(
                rx.heading("Sector P&L Contribution", size="5"),
                rx.recharts.bar_chart(
                    rx.recharts.bar(
                        data_key="pnl",
                        fill="#22c55e",
                    ),
                    rx.recharts.x_axis(data_key="sector"),
                    rx.recharts.y_axis(),
                    rx.recharts.tooltip(),
                    data=PortfolioState.sector_risk_contribution,
                    height=300,
                ),
            ),
            style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
        ),
        columns="3",
        spacing="4",
        width="95%",
        margin="0 2rem",
    )

def performance_view() -> rx.Component:
    """Performance metrics view"""
    return rx.vstack(
        rx.card(
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
            style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
        ),
        rx.card(
            rx.vstack(
                rx.heading("Portfolio vs Benchmark", size="5"),
                rx.recharts.line_chart(
                    rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
                    rx.recharts.line(data_key="portfolio_pct", stroke="#a855f7", name="Portfolio (%)", dot=False),
                    rx.recharts.line(data_key="benchmark_pct", stroke="#22d3ee", name="Benchmark (%)", dot=False),
                    rx.recharts.x_axis(data_key="date"),
                    rx.recharts.y_axis(),
                    rx.recharts.tooltip(),
                    rx.recharts.legend(),
                    data=PortfolioState.portfolio_vs_btc,
                    height=320,
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

        # Optimization Results
        rx.grid(
            # Current Allocation vs Suggested
            rx.card(
                rx.vstack(
                    rx.heading("Current vs Suggested Allocation", size="5"),
                    rx.cond(
                        PortfolioState.suggested_allocation,
                        rx.vstack(
                            rx.text("Suggested Allocation", size="4", weight="bold", color="#22c55e"),
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
                        rx.text("Run optimization to see suggested allocation", color="#9ca3af"),
                    ),
                ),
                style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            ),

            # Optimization Comparison
            rx.card(
                rx.vstack(
                    rx.heading("Method Comparison", size="5"),
                    rx.cond(
                        PortfolioState.optimization_comparison,
                        rx.data_table(
                            data=PortfolioState.optimization_comparison,
                            columns=[
                                {"Header": "Method", "accessor": "method"},
                                {"Header": "Expected Return", "accessor": "expected_return"},
                                {"Header": "Volatility", "accessor": "volatility"},
                                {"Header": "Sharpe Ratio", "accessor": "sharpe_ratio"},
                                {"Header": "Time (s)", "accessor": "optimization_time"}
                            ],
                            pagination=False,
                        ),
                        rx.text("Run method comparison to see results", color="#9ca3af"),
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
