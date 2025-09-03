# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.

"""
Options Analytics Dashboard Page for Reflex
File: src/dashboard/pages/options_analytics.py
Displays real-time options data with Greeks
"""

import reflex as rx
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio

# Import our analytics processor
from src.analytics.options_processor import RealTimeOptionsProcessor, process_options
from src.models.options.options_chain import OptionsChainProcessor
from src.core.database.operations import DatabaseOperations


class OptionsAnalyticsState(rx.State):
    """State management for options analytics dashboard."""
    
    # Data state
    current_currency: str = "BTC"
    spot_price: float = 0.0
    options_data: List[Dict] = []
    chain_metrics: Dict[str, Any] = {}
    portfolio_greeks: Dict[str, Any] = {}
    risk_metrics: Dict[str, Any] = {}
    
    # UI state
    is_loading: bool = False
    last_update: str = ""
    selected_expiry: str = "All"
    selected_type: str = "All"
    
    # Charts data
    iv_surface_data: Dict = {}
    gamma_exposure_data: Dict = {}
    greeks_distribution: Dict = {}
    
    # Auto-refresh
    auto_refresh: bool = False
    refresh_interval: int = 60  # seconds
    
    @rx.background
    async def load_options_data(self):
        """Load options data with Greeks in background."""
        async with self:
            self.is_loading = True
        
        try:
            # Process options with Greeks
            processor = RealTimeOptionsProcessor()
            analytics = await processor.process_live_options(self.current_currency)
            
            if analytics:
                async with self:
                    self.spot_price = analytics.spot_price
                    self.chain_metrics = analytics.chain_metrics
                    self.portfolio_greeks = analytics.portfolio_greeks
                    self.risk_metrics = analytics.risk_metrics
                    self.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Load detailed options data
                    await self._load_detailed_options()
                    
                    # Update charts
                    await self._update_charts()
            
        except Exception as e:
            print(f"Error loading options data: {e}")
        
        finally:
            async with self:
                self.is_loading = False
    
    async def _load_detailed_options(self):
        """Load detailed options data from database."""
        try:
            db_ops = DatabaseOperations()
            
            # Get recent options with Greeks
            query = {
                'underlying': self.current_currency,
                'has_greeks': True,
                'processed_at': {'$gte': datetime.now() - timedelta(hours=1)}
            }
            
            options = list(db_ops.db.options_data.find(query).limit(500))
            
            # Convert to list of dicts for display
            self.options_data = [
                {
                    'instrument': opt.get('instrument_name', ''),
                    'strike': opt.get('strike', 0),
                    'expiry': str(opt.get('expiry', ''))[:10],
                    'type': opt.get('option_type', ''),
                    'mark_price': opt.get('mark_price', 0),
                    'iv': opt.get('implied_volatility', 0) * 100,
                    'delta': opt.get('delta', 0),
                    'gamma': opt.get('gamma', 0),
                    'theta': opt.get('theta', 0),
                    'vega': opt.get('vega', 0),
                    'volume': opt.get('volume', 0),
                    'oi': opt.get('open_interest', 0)
                }
                for opt in options
            ]
            
        except Exception as e:
            print(f"Error loading detailed options: {e}")
            self.options_data = []
    
    async def _update_charts(self):
        """Update chart data."""
        if not self.options_data:
            return
        
        df = pd.DataFrame(self.options_data)
        
        # IV Surface data
        self._prepare_iv_surface(df)
        
        # Gamma exposure profile
        self._prepare_gamma_exposure(df)
        
        # Greeks distribution
        self._prepare_greeks_distribution(df)
    
    def _prepare_iv_surface(self, df: pd.DataFrame):
        """Prepare IV surface data for 3D plot."""
        try:
            # Pivot data for surface
            surface_data = df.pivot_table(
                index='strike',
                columns='expiry',
                values='iv',
                aggfunc='mean'
            ).fillna(method='ffill').fillna(method='bfill')
            
            self.iv_surface_data = {
                'x': surface_data.columns.tolist(),
                'y': surface_data.index.tolist(),
                'z': surface_data.values.tolist()
            }
        except:
            self.iv_surface_data = {}
    
    def _prepare_gamma_exposure(self, df: pd.DataFrame):
        """Prepare gamma exposure data."""
        try:
            # Group by strike
            gamma_by_strike = df.groupby('strike')['gamma'].sum()
            
            self.gamma_exposure_data = {
                'strikes': gamma_by_strike.index.tolist(),
                'gamma': gamma_by_strike.values.tolist()
            }
        except:
            self.gamma_exposure_data = {}
    
    def _prepare_greeks_distribution(self, df: pd.DataFrame):
        """Prepare Greeks distribution data."""
        try:
            self.greeks_distribution = {
                'delta': {
                    'calls': df[df['type'] == 'call']['delta'].tolist(),
                    'puts': df[df['type'] == 'put']['delta'].tolist()
                },
                'gamma': {
                    'values': df['gamma'].tolist(),
                    'strikes': df['strike'].tolist()
                },
                'theta': {
                    'values': df['theta'].tolist(),
                    'strikes': df['strike'].tolist()
                }
            }
        except:
            self.greeks_distribution = {}
    
    def switch_currency(self, currency: str):
        """Switch between BTC and ETH."""
        self.current_currency = currency
        return self.load_options_data()
    
    def toggle_auto_refresh(self):
        """Toggle auto-refresh."""
        self.auto_refresh = not self.auto_refresh
        if self.auto_refresh:
            return self.start_auto_refresh()
    
    @rx.background
    async def start_auto_refresh(self):
        """Start auto-refresh loop."""
        while self.auto_refresh:
            await self.load_options_data()
            await asyncio.sleep(self.refresh_interval)
    
    def filter_by_expiry(self, expiry: str):
        """Filter options by expiry."""
        self.selected_expiry = expiry
    
    def filter_by_type(self, opt_type: str):
        """Filter options by type."""
        self.selected_type = opt_type


def create_metrics_card(title: str, value: str, subtitle: str = "", color: str = "blue") -> rx.Component:
    """Create a metric display card."""
    return rx.card(
        rx.vstack(
            rx.text(title, size="2", weight="medium", color="gray"),
            rx.text(value, size="6", weight="bold"),
            rx.cond(
                subtitle != "",
                rx.text(subtitle, size="1", color="gray"),
                rx.text("")
            ),
            spacing="1",
            align="start"
        ),
        padding="4",
        style={
            "background": f"linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(168, 85, 247, 0.05))",
            "border": "1px solid rgba(168, 85, 247, 0.2)"
        }
    )


def create_iv_surface_chart(state: OptionsAnalyticsState) -> rx.Component:
    """Create 3D IV surface visualization."""
    if not state.iv_surface_data:
        return rx.text("No IV surface data available", color="gray")
    
    fig = go.Figure(data=[
        go.Surface(
            x=state.iv_surface_data.get('x', []),
            y=state.iv_surface_data.get('y', []),
            z=state.iv_surface_data.get('z', []),
            colorscale='Viridis',
            name='IV Surface'
        )
    ])
    
    fig.update_layout(
        title="Implied Volatility Surface",
        scene=dict(
            xaxis_title="Expiry",
            yaxis_title="Strike",
            zaxis_title="IV (%)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        template="plotly_dark",
        height=400
    )
    
    return rx.plotly(data=fig)


def create_gamma_exposure_chart(state: OptionsAnalyticsState) -> rx.Component:
    """Create gamma exposure chart."""
    if not state.gamma_exposure_data:
        return rx.text("No gamma exposure data available", color="gray")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=state.gamma_exposure_data.get('strikes', []),
        y=state.gamma_exposure_data.get('gamma', []),
        name='Gamma Exposure',
        marker_color='rgba(168, 85, 247, 0.8)'
    ))
    
    fig.update_layout(
        title="Gamma Exposure by Strike",
        xaxis_title="Strike Price",
        yaxis_title="Gamma",
        template="plotly_dark",
        height=300
    )
    
    return rx.plotly(data=fig)


def create_options_table(state: OptionsAnalyticsState) -> rx.Component:
    """Create options data table with Greeks."""
    
    # Filter data based on selections
    filtered_data = state.options_data
    
    if state.selected_expiry != "All":
        filtered_data = [d for d in filtered_data if d['expiry'] == state.selected_expiry]
    
    if state.selected_type != "All":
        filtered_data = [d for d in filtered_data if d['type'] == state.selected_type]
    
    # Create table
    return rx.box(
        rx.table.root(
            rx.table.header(
                rx.table.row(
                    rx.table.column_header_cell("Instrument"),
                    rx.table.column_header_cell("Strike"),
                    rx.table.column_header_cell("Type"),
                    rx.table.column_header_cell("Price"),
                    rx.table.column_header_cell("IV %"),
                    rx.table.column_header_cell("Delta"),
                    rx.table.column_header_cell("Gamma"),
                    rx.table.column_header_cell("Theta"),
                    rx.table.column_header_cell("Vega"),
                    rx.table.column_header_cell("Volume"),
                )
            ),
            rx.table.body(
                rx.foreach(
                    filtered_data[:50],  # Show first 50
                    lambda option: rx.table.row(
                        rx.table.cell(option['instrument']),
                        rx.table.cell(f"${option['strike']:,.0f}"),
                        rx.table.cell(
                            rx.badge(
                                option['type'].upper(),
                                color="green" if option['type'] == 'call' else "red"
                            )
                        ),
                        rx.table.cell(f"{option['mark_price']:.4f}"),
                        rx.table.cell(f"{option['iv']:.1f}"),
                        rx.table.cell(f"{option['delta']:.4f}"),
                        rx.table.cell(f"{option['gamma']:.6f}"),
                        rx.table.cell(f"{option['theta']:.4f}"),
                        rx.table.cell(f"{option['vega']:.4f}"),
                        rx.table.cell(f"{option['volume']:.0f}"),
                    )
                )
            )
        ),
        style={
            "max_height": "400px",
            "overflow_y": "auto"
        }
    )


def options_analytics_page() -> rx.Component:
    """Create the options analytics dashboard page."""
    return rx.box(
        # Header
        rx.hstack(
            rx.heading("Options Analytics", size="8"),
            rx.spacer(),
            rx.hstack(
                rx.button(
                    "BTC",
                    on_click=lambda: OptionsAnalyticsState.switch_currency("BTC"),
                    variant="solid" if OptionsAnalyticsState.current_currency == "BTC" else "outline",
                    color_scheme="purple"
                ),
                rx.button(
                    "ETH",
                    on_click=lambda: OptionsAnalyticsState.switch_currency("ETH"),
                    variant="solid" if OptionsAnalyticsState.current_currency == "ETH" else "outline",
                    color_scheme="purple"
                ),
                rx.button(
                    rx.cond(
                        OptionsAnalyticsState.is_loading,
                        rx.spinner(size="3"),
                        rx.icon("refresh-cw", size=16)
                    ),
                    "Refresh",
                    on_click=OptionsAnalyticsState.load_options_data,
                    loading=OptionsAnalyticsState.is_loading,
                    color_scheme="purple"
                ),
                rx.switch(
                    checked=OptionsAnalyticsState.auto_refresh,
                    on_change=lambda: OptionsAnalyticsState.toggle_auto_refresh()
                ),
                rx.text("Auto", size="2"),
                spacing="3"
            ),
            width="100%",
            padding="4",
            align="center"
        ),
        
        # Metrics Cards
        rx.grid(
            create_metrics_card(
                "Spot Price",
                f"${OptionsAnalyticsState.spot_price:,.2f}",
                OptionsAnalyticsState.current_currency
            ),
            create_metrics_card(
                "Average IV",
                f"{OptionsAnalyticsState.chain_metrics.get('average_iv', 0)*100:.1f}%",
                "Chain average"
            ),
            create_metrics_card(
                "Put/Call Ratio",
                f"{OptionsAnalyticsState.chain_metrics.get('put_call_ratio', 0):.3f}",
                "Volume ratio"
            ),
            create_metrics_card(
                "Max Pain",
                f"${OptionsAnalyticsState.chain_metrics.get('max_pain_strike', 0):,.0f}",
                "Strike price"
            ),
            columns="4",
            spacing="4",
            padding="4",
            width="100%"
        ),
        
        # Portfolio Greeks
        rx.cond(
            OptionsAnalyticsState.portfolio_greeks != {},
            rx.card(
                rx.vstack(
                    rx.heading("Portfolio Greeks", size="5"),
                    rx.grid(
                        rx.box(
                            rx.text("Total Delta", size="2", color="gray"),
                            rx.text(f"{OptionsAnalyticsState.portfolio_greeks.get('total_delta', 0):.4f}", size="4", weight="bold")
                        ),
                        rx.box(
                            rx.text("Total Gamma", size="2", color="gray"),
                            rx.text(f"{OptionsAnalyticsState.portfolio_greeks.get('total_gamma', 0):.6f}", size="4", weight="bold")
                        ),
                        rx.box(
                            rx.text("Total Theta", size="2", color="gray"),
                            rx.text(f"{OptionsAnalyticsState.portfolio_greeks.get('total_theta', 0):.4f}", size="4", weight="bold")
                        ),
                        rx.box(
                            rx.text("Total Vega", size="2", color="gray"),
                            rx.text(f"{OptionsAnalyticsState.portfolio_greeks.get('total_vega', 0):.4f}", size="4", weight="bold")
                        ),
                        columns="4",
                        spacing="4",
                        width="100%"
                    ),
                    spacing="3",
                    width="100%"
                ),
                padding="4",
                margin="4"
            ),
            rx.text("")
        ),
        
        # Charts Grid
        rx.grid(
            rx.card(
                create_iv_surface_chart(OptionsAnalyticsState),
                padding="4"
            ),
            rx.card(
                create_gamma_exposure_chart(OptionsAnalyticsState),
                padding="4"
            ),
            columns="2",
            spacing="4",
            padding="4",
            width="100%"
        ),
        
        # Options Table
        rx.card(
            rx.vstack(
                rx.hstack(
                    rx.heading("Options Chain with Greeks", size="5"),
                    rx.spacer(),
                    rx.select(
                        ["All", "call", "put"],
                        placeholder="Filter by type",
                        on_change=OptionsAnalyticsState.filter_by_type,
                        size="2"
                    ),
                    spacing="3",
                    width="100%"
                ),
                create_options_table(OptionsAnalyticsState),
                spacing="4",
                width="100%"
            ),
            padding="4",
            margin="4"
        ),
        
        # Footer
        rx.hstack(
            rx.text(f"Last updated: {OptionsAnalyticsState.last_update}", size="1", color="gray"),
            rx.spacer(),
            rx.text(f"Options loaded: {len(OptionsAnalyticsState.options_data)}", size="1", color="gray"),
            width="100%",
            padding="4"
        ),
        
        # Background style
        style={
            "background": "linear-gradient(135deg, #0f0f1e, #1a1a2e)",
            "min_height": "100vh"
        },
        
        # Load data on mount
        on_mount=OptionsAnalyticsState.load_options_data
    )


# Add to main app
app = rx.App(
    theme=rx.theme(
        appearance="dark",
        has_background=True,
        radius="medium",
        accent_color="purple"
    )
)

app.add_page(
    options_analytics_page,
    route="/options",
    title="Options Analytics - Qortfolio V2"
)