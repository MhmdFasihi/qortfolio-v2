"""Risk Analytics Page with Portfolio Risk Management and Performance Analysis"""

import reflex as rx
from ..state import RiskState
from ..components.navigation import page_layout

def risk_analytics_page() -> rx.Component:
    """Risk analytics page with portfolio risk management and performance analysis."""
    content = rx.vstack(
        # Header
        rx.hstack(
            rx.heading("Risk Analytics", size="8", color="#a855f7"),
            rx.spacer(),
            rx.button(
                "Calculate Risk Metrics",
                on_click=RiskState.calculate_portfolio_risk,
                loading=RiskState.loading,
                color_scheme="purple",
                size="3",
            ),
            width="100%",
            padding="2rem",
        ),

        # Portfolio Selection / Auto-refresh
        rx.hstack(
            rx.text("Portfolio:", size="4", weight="bold"),
            rx.select(RiskState.portfolio_options, value=RiskState.selected_portfolio, on_change=RiskState.set_portfolio),
            rx.text("Benchmark:", size="4", weight="bold", margin_left="2rem"),
            rx.select(["BTC", "ETH", "S&P500"], value=RiskState.selected_benchmark, on_change=RiskState.set_benchmark),
            rx.text("Auto:", size="3", margin_left="2rem"),
            rx.switch(is_checked=RiskState.auto_refresh, on_change=RiskState.toggle_auto_refresh),
            rx.spacer(),
            rx.text("Status:", color="#9ca3af"),
            rx.text(RiskState.calculation_status, color="#a855f7"),
            spacing="4",
            padding="1rem 2rem",
        ),

        # Risk Metrics Cards
        rx.grid(
            risk_metric_card("Portfolio Value", RiskState.portfolio_value_display, "dollar-sign"),
            risk_metric_card("VaR (95%)", RiskState.var_95_display, "trending-down"),
            risk_metric_card("CVaR (95%)", RiskState.cvar_95_display, "alert-triangle"),
            risk_metric_card("Max Drawdown", RiskState.max_drawdown_display, "arrow-down"),
            risk_metric_card("Sharpe Ratio", RiskState.sharpe_ratio_display, "target"),
            risk_metric_card("Sortino Ratio", RiskState.sortino_ratio_display, "trending-up"),
            risk_metric_card("Beta", RiskState.beta_display, "activity"),
            risk_metric_card("R-Squared", RiskState.r_squared_display, "percent"),
            columns="4",
            spacing="4",
            width="100%",
            padding="0 2rem",
        ),

        # Analytics Tabs
        rx.tabs(
            rx.tabs.list(
                rx.tabs.trigger("Risk Dashboard", value="dashboard"),
                rx.tabs.trigger("Performance Analysis", value="performance"),
                rx.tabs.trigger("Sector Analysis", value="sectors"),
            ),
            rx.tabs.content(
                # Risk Dashboard Tab
                risk_dashboard_content(),
                value="dashboard"
            ),
            rx.tabs.content(
                # Performance Analysis Tab
                performance_analysis_content(),
                value="performance"
            ),
            rx.tabs.content(
                # Sector Analysis Tab
                sector_analysis_content(),
                value="sectors"
            ),
            default_value="dashboard",
            width="100%",
            padding="0 2rem",
        ),

        width="100%",
    )

    return page_layout(content, "Risk Analytics")


def risk_dashboard_content() -> rx.Component:
    """Risk dashboard tab content."""
    return rx.vstack(
        rx.grid(
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.icon("bar-chart", color="#ef4444"),
                        rx.heading("Risk Distribution", size="5"),
                        width="100%",
                    ),
                    rx.cond(
                        RiskState.loading,
                        rx.center(rx.spinner(color="purple", size="3")),
                        rx.cond(
                            RiskState.risk_distribution_data,
                            rx.recharts.pie_chart(
                                rx.recharts.pie(
                                    data_key="value",
                                    name_key="risk_type",
                                    cx="50%",
                                    cy="50%",
                                    outer_radius=100,
                                    fill="#a855f7"
                                ),
                                rx.recharts.tooltip(),
                                data=RiskState.risk_distribution_data,
                                height=300,
                            ),
                            rx.text("No risk distribution data", color="#9ca3af"),
                        ),
                    ),
                ),
                style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            ),
            rx.card(
                rx.vstack(
                    rx.heading("Value at Risk History", size="5"),
                    rx.cond(
                        RiskState.var_history_data,
                        rx.recharts.line_chart(
                            rx.recharts.line(data_key="var_95", stroke="#ef4444", name="VaR 95%"),
                            rx.recharts.line(data_key="var_99", stroke="#dc2626", name="VaR 99%"),
                            rx.recharts.x_axis(data_key="date"),
                            rx.recharts.y_axis(),
                            rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
                            rx.recharts.tooltip(),
                            rx.recharts.legend(),
                            data=RiskState.var_history_data,
                            height=300,
                        ),
                        rx.text("No VaR history data", color="#9ca3af"),
                    ),
                ),
                style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            ),
            columns="2",
            spacing="4",
            width="100%",
        ),

        rx.card(
            rx.vstack(
                rx.heading("Portfolio Correlation Matrix", size="5"),
                rx.cond(
                    RiskState.correlation_matrix_data,
                    rx.data_table(
                        data=RiskState.correlation_matrix_data,
                        columns=RiskState.correlation_columns,
                        pagination=False,
                    ),
                    rx.text("No correlation data", color="#9ca3af"),
                ),
            ),
            style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            width="100%",
        ),
        spacing="4",
        width="100%",
    )


def performance_analysis_content() -> rx.Component:
    """Performance analysis tab content."""
    return rx.vstack(
        rx.grid(
            performance_metric_card("Total Return", RiskState.total_return_display, "trending-up"),
            performance_metric_card("Annual Return", RiskState.annual_return_display, "calendar"),
            performance_metric_card("Volatility", RiskState.volatility_display, "activity"),
            performance_metric_card("Win Rate", RiskState.win_rate_display, "percent"),
            columns="4",
            spacing="4",
            width="100%",
        ),

        rx.grid(
            rx.card(
                rx.vstack(
                    rx.heading("Performance vs Benchmark", size="5"),
                    rx.cond(
                        RiskState.performance_comparison_data,
                        rx.recharts.line_chart(
                            rx.recharts.line(data_key="portfolio", stroke="#a855f7", name="Portfolio"),
                            rx.recharts.line(data_key="benchmark", stroke="#6b7280", name="Benchmark"),
                            rx.recharts.x_axis(data_key="date"),
                            rx.recharts.y_axis(),
                            rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
                            rx.recharts.tooltip(),
                            rx.recharts.legend(),
                            data=RiskState.performance_comparison_data,
                            height=300,
                        ),
                        rx.text("No performance comparison data", color="#9ca3af"),
                    ),
                ),
                style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            ),
            rx.card(
                rx.vstack(
                    rx.heading("Drawdown Analysis", size="5"),
                    rx.cond(
                        RiskState.drawdown_data,
                        rx.recharts.area_chart(
                            rx.recharts.area(data_key="drawdown", fill="#ef4444", stroke="#ef4444"),
                            rx.recharts.x_axis(data_key="date"),
                            rx.recharts.y_axis(),
                            rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
                            rx.recharts.tooltip(),
                            data=RiskState.drawdown_data,
                            height=300,
                        ),
                        rx.text("No drawdown data", color="#9ca3af"),
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



def sector_analysis_content() -> rx.Component:
    """Sector analysis tab content."""
    return rx.vstack(
        rx.grid(
            rx.card(
                rx.vstack(
                    rx.heading("Sector Allocation", size="5"),
                    rx.cond(
                        RiskState.sector_allocation_data,
                        rx.recharts.bar_chart(
                            rx.recharts.bar(data_key="allocation", fill="#a855f7"),
                            rx.recharts.x_axis(data_key="sector"),
                            rx.recharts.y_axis(),
                            rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
                            rx.recharts.tooltip(),
                            data=RiskState.sector_allocation_data,
                            height=300,
                        ),
                        rx.text("No sector allocation data", color="#9ca3af"),
                    ),
                ),
                style={"background": "rgba(45, 27, 61, 0.8)", "border": "1px solid #4c1d95"},
            ),
            rx.card(
                rx.vstack(
                    rx.heading("Sector Risk Contribution", size="5"),
                    rx.cond(
                        RiskState.sector_risk_data,
                        rx.recharts.pie_chart(
                            rx.recharts.pie(
                                data_key="risk_contribution",
                                name_key="sector",
                                cx="50%",
                                cy="50%",
                                outer_radius=100,
                                fill="#ef4444"
                            ),
                            rx.recharts.tooltip(),
                            data=RiskState.sector_risk_data,
                            height=300,
                        ),
                        rx.text("No sector risk data", color="#9ca3af"),
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


def risk_metric_card(label: str, value: rx.Var, icon: str = "bar-chart") -> rx.Component:
    """Risk metric display card with icon."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon(icon, size=24, color="#ef4444"),
                rx.spacer(),
            ),
            rx.text(label, size="2", color="#ef4444"),
            rx.text(value, size="5", weight="bold"),
            align="center",
            spacing="1",
        ),
        style={
            "background": "rgba(45, 27, 61, 0.8)",
            "border": "1px solid #ef4444",
            "min_height": "120px",
        }
    )


def performance_metric_card(label: str, value: rx.Var, icon: str) -> rx.Component:
    """Performance metric card."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon(icon, size=24, color="#22c55e"),
                rx.spacer(),
            ),
            rx.text(label, size="2", color="#22c55e", weight="medium"),
            rx.text(value, size="4", weight="bold", color="#ffffff"),
            align="center",
            spacing="1",
        ),
        style={
            "background": "linear-gradient(135deg, rgba(45, 27, 61, 0.9), rgba(34, 197, 94, 0.1))",
            "border": "1px solid #22c55e",
            "min_height": "120px",
        }
    )