"""Enhanced State Management with MongoDB + Deribit fallback"""

import reflex as rx
import plotly.graph_objects as go
from typing import Dict, List
from datetime import datetime
import asyncio


class OptionsState(rx.State):
    """Options page state"""

    # UI State
    selected_currency: str = "BTC"
    loading: bool = False
    db_status: str = "Checking..."

    # Data
    options_data: List[Dict] = []
    # Keep an unfiltered/raw copy for analytics like IV surface
    raw_options_data: List[Dict] = []
    calls_data: List[Dict] = []
    puts_data: List[Dict] = []
    total_contracts: int = 0
    avg_iv: float = 0.0
    max_oi: int = 0
    total_volume: int = 0
    last_update: str = "Never"
    spot_price: float = 0.0

    @rx.var
    def spot_display(self) -> str:
        try:
            return f"{float(self.spot_price):,.2f}"
        except Exception:
            return "-"

    # Expiry filter
    expiry_options: List[str] = []
    selected_expiry: str = ""
    # Auto refresh
    auto_refresh: bool = False
    refresh_seconds: int = 60

    # Week 3 Analytics Extensions
    calls_count: int = 0
    puts_count: int = 0

    # Chain Analytics
    call_put_ratio: float = 1.0
    max_pain_strike: float = 0.0
    gamma_exposure: float = 0.0
    iv_rank: float = 0.5

    # Volatility Surface
    surface_loading: bool = False
    volatility_surface_data: List[Dict] = []
    volatility_surface_spec: Dict = {}
    atm_term_structure_data: List[Dict] = []

    # Flow Analysis
    flow_direction: str = "NEUTRAL"
    flow_confidence: float = 0.0
    unusual_activity: List[Dict] = []

    # Greeks Analysis
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_theta: float = 0.0
    total_vega: float = 0.0
    gamma_profile_data: List[Dict] = []

    @rx.var
    def avg_iv_pct(self) -> str:
        """Formatted average IV as percentage"""
        return f"{self.avg_iv:.2%}"

    @rx.var
    def call_put_ratio_display(self) -> str:
        """Formatted call/put ratio"""
        try:
            return f"{self.call_put_ratio:.2f}"
        except Exception:
            return "1.00"

    @rx.var
    def max_pain_display(self) -> str:
        """Formatted max pain strike"""
        try:
            return f"${self.max_pain_strike:,.0f}"
        except Exception:
            return "$0"

    @rx.var
    def gamma_exposure_display(self) -> str:
        """Formatted gamma exposure"""
        try:
            if abs(self.gamma_exposure) > 1e6:
                return f"${self.gamma_exposure/1e6:.1f}M"
            elif abs(self.gamma_exposure) > 1e3:
                return f"${self.gamma_exposure/1e3:.1f}K"
            else:
                return f"${self.gamma_exposure:.0f}"
        except Exception:
            return "$0"

    @rx.var
    def iv_rank_display(self) -> str:
        """Formatted IV rank as percentage"""
        try:
            return f"{self.iv_rank:.1%}"
        except Exception:
            return "50.0%"

    @rx.var
    def total_delta_display(self) -> str:
        """Formatted total delta"""
        try:
            return f"{self.total_delta:.2f}"
        except Exception:
            return "0.00"

    @rx.var
    def total_gamma_display(self) -> str:
        """Formatted total gamma"""
        try:
            return f"{self.total_gamma:.4f}"
        except Exception:
            return "0.0000"

    @rx.var
    def total_theta_display(self) -> str:
        """Formatted total theta"""
        try:
            return f"{self.total_theta:.2f}"
        except Exception:
            return "0.00"

    @rx.var
    def total_vega_display(self) -> str:
        """Formatted total vega"""
        try:
            return f"{self.total_vega:.2f}"
        except Exception:
            return "0.00"

    @rx.var
    def volatility_surface_figure(self) -> go.Figure:
        """Create a 3D volatility surface figure from data."""
        if not self.volatility_surface_data:
            return go.Figure()
        
        # Extract data for 3D surface plot
        strikes = []
        expiries = []
        ivs = []
        
        for point in self.volatility_surface_data:
            if 'strike' in point and 'expiry' in point and 'iv' in point:
                strikes.append(point['strike'])
                expiries.append(point['expiry'])
                ivs.append(point['iv'])
        
        if not strikes or not expiries or not ivs:
            return go.Figure()
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(
            z=ivs,
            x=strikes,
            y=expiries,
            colorscale='Viridis',
            name='Volatility Surface'
        )])
        
        fig.update_layout(
            title='Volatility Surface',
            scene=dict(
                xaxis_title='Strike Price',
                yaxis_title='Expiry',
                zaxis_title='Implied Volatility'
            ),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        return fig

    @rx.var
    def volatility_surface_plot(self) -> go.Figure:
        """Build a Plotly Figure from stored surface spec (data/layout)."""
        try:
            spec = self.volatility_surface_spec or {}
            if not spec:
                return go.Figure()
            # Let Plotly build the figure from dict spec
            return go.Figure(spec)
        except Exception:
            return go.Figure()

    async def set_currency(self, currency: str):
        """Set selected currency and fetch data"""
        self.selected_currency = currency
        await self.fetch_options_data()

    async def set_expiry(self, expiry: str):
        """Set selected expiry and refresh data"""
        self.selected_expiry = expiry
        await self.fetch_options_data()

    def set_refresh_seconds(self, secs: str):
        """Update refresh interval from UI select (string -> int)."""
        try:
            self.refresh_seconds = int(secs)
        except Exception:
            self.refresh_seconds = 60

    def toggle_auto_refresh(self, value: bool):
        """Handle switch toggle for auto refresh."""
        if bool(value):
            return OptionsState.start_auto_refresh()
        else:
            return OptionsState.stop_auto_refresh()

    async def start_auto_refresh(self):
        """Start periodic refresh loop."""
        if self.auto_refresh:
            return
        self.auto_refresh = True
        # Run a background loop
        try:
            while self.auto_refresh:
                await self.fetch_options_data()
                # Avoid zero/negative
                delay = self.refresh_seconds if self.refresh_seconds and self.refresh_seconds > 0 else 60
                await asyncio.sleep(delay)
        except Exception:
            # Stop on error to avoid runaway logs
            self.auto_refresh = False

    def stop_auto_refresh(self):
        """Stop periodic refresh loop."""
        self.auto_refresh = False

    async def build_volatility_surface(self):
        """Build volatility surface from current options data."""
        self.surface_loading = True
        try:
            # Import analytics modules
            from src.analytics.volatility.surface_builder import VolatilitySurfaceBuilder
            from src.core.database.operations import db_ops

            # Build surface
            builder = VolatilitySurfaceBuilder(db_ops)
            # Use raw (unfiltered) options data for surface construction across expiries
            data_for_surface = self.raw_options_data if self.raw_options_data else self.options_data
            if not data_for_surface:
                # Graceful no-op if there is no data
                self.volatility_surface_data = []
                self.volatility_surface_spec = {}
                self.atm_term_structure_data = []
                return

            surface = await builder.build_and_store_surface(
                self.selected_currency,
                data_for_surface
            )

            # Build serializable surface config (no Plotly Figure in state)
            if surface and surface.surface_data:
                surface_grid = surface.surface_data
                x = surface_grid.get('strikes_range', [])
                y = surface_grid.get('time_range', [])
                z = surface_grid.get('iv_grid', [])
                trace = {'type': 'surface', 'x': x, 'y': y, 'z': z, 'colorscale': 'Viridis'}
                layout = {
                    'title': 'Implied Volatility Surface',
                    'scene': {
                        'xaxis': {'title': 'Strike'},
                        'yaxis': {'title': 'Days to Expiry'},
                        'zaxis': {'title': 'Implied Volatility'},
                    },
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'plot_bgcolor': 'rgba(0,0,0,0)'
                }
                self.volatility_surface_data = [trace]
                self.volatility_surface_spec = {'data': [trace], 'layout': layout}

                # Create term structure data
                atm_data = []
                for expiry, iv in surface.atm_term_structure.items():
                    atm_data.append({
                        'expiry': expiry,
                        'iv': iv
                    })
                self.atm_term_structure_data = atm_data

        except Exception as e:
            print(f"Surface building failed: {e}")
            self.volatility_surface_data = []
            self.volatility_surface_spec = {}
            self.atm_term_structure_data = []
        finally:
            self.surface_loading = False

    async def analyze_options_chain(self):
        """Perform comprehensive options chain analysis."""
        try:
            if not self.options_data:
                return

            # Import analytics modules
            from src.analytics.options.chain_analyzer import OptionsChainAnalyzer
            from src.core.database.operations import db_ops

            # Analyze chain
            analyzer = OptionsChainAnalyzer(db_ops)
            metrics = analyzer.analyze_options_chain(self.options_data, self.spot_price)

            # Update state with analytics
            self.call_put_ratio = metrics.call_put_ratio
            self.max_pain_strike = metrics.max_pain_strike
            self.gamma_exposure = metrics.gamma_exposure
            self.iv_rank = metrics.iv_rank
            self.flow_direction = metrics.flow_direction.value.upper()
            self.unusual_activity = metrics.unusual_activity

            # Calculate flow confidence (simplified)
            if metrics.call_put_ratio > 1.5:
                self.flow_confidence = min(0.8, (metrics.call_put_ratio - 1) / 2)
            elif metrics.call_put_ratio < 0.5:
                self.flow_confidence = min(0.8, (1 - metrics.call_put_ratio) / 2)
            else:
                self.flow_confidence = 0.1

        except Exception as e:
            print(f"Chain analysis failed: {e}")

    async def calculate_portfolio_greeks(self):
        """Calculate portfolio Greeks from options data."""
        try:
            if not self.options_data:
                return

            # Import Greeks calculator
            from src.models.options.greeks_calculator import GreeksCalculator

            # Convert options data to positions format
            positions = []
            for opt in self.options_data:
                try:
                    # Simulate position quantities (in real app, this would come from portfolio)
                    quantity = 1  # Default position size

                    # Calculate time to maturity
                    from datetime import datetime
                    expiry_str = opt.get('expiry', '')
                    if expiry_str:
                        expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d')
                        time_to_maturity = (expiry_date - datetime.now()).days / 365.25
                    else:
                        time_to_maturity = 30 / 365.25

                    position = {
                        'quantity': quantity,
                        'spot_price': self.spot_price,
                        'strike_price': float(opt.get('strike', 0)),
                        'time_to_maturity': max(time_to_maturity, 1/365),
                        'volatility': float(opt.get('iv', '0%').replace('%', '')) / 100,
                        'option_type': opt.get('option_type', 'call').lower(),
                        'underlying': self.selected_currency,
                        'is_coin_based': True
                    }
                    positions.append(position)
                except Exception:
                    continue

            if positions:
                # Calculate portfolio Greeks
                calc = GreeksCalculator()
                portfolio_greeks = calc.calculate_portfolio_greeks(positions)

                # Update state
                self.total_delta = portfolio_greeks.total_delta
                self.total_gamma = portfolio_greeks.total_gamma
                self.total_theta = portfolio_greeks.total_theta
                self.total_vega = portfolio_greeks.total_vega

                # Calculate gamma profile
                gamma_profile = calc.calculate_gamma_exposure_profile(positions)
                self.gamma_profile_data = gamma_profile.to_dict('records')

        except Exception as e:
            print(f"Greeks calculation failed: {e}")

    async def fetch_options_data(self):
        """Fetch options data with expiry filtering and robust fallbacks."""
        self.loading = True

        def to_date_str(v) -> str:
            try:
                return str(v)[:10]
            except Exception:
                return ""

        try:
            from src.core.database.connection import db_connection
            from src.core.database.operations import db_ops

            # Try to connect asynchronously
            try:
                await db_connection.connect_async()
                self.db_status = "Connected"
            except Exception:
                self.db_status = "Disconnected"

            raw: List[Dict] = []
            if self.db_status == "Connected":
                # Pull more rows; we'll filter by expiry client-side
                raw = await db_ops.get_latest_options(
                    underlying=self.selected_currency,
                    limit=1000,
                )

            # Fallback to live Deribit data if DB empty
            if not raw:
                try:
                    from src.data.collectors.deribit_collector import DeribitCollector
                    collector = DeribitCollector(testnet=False)
                    try:
                        df = await collector.get_options_chain(
                            currency=self.selected_currency,
                            strikes_around_atm=6,
                        )
                        raw = df.to_dict("records") if df is not None else []

                        # Attempt to persist the fetched data to MongoDB
                        if raw:
                            try:
                                await collector.store_data("options_data", raw)
                                # Mark DB as connected if storage succeeded
                                self.db_status = "Connected"
                            except Exception:
                                # Ignore storage failure; still render data
                                pass
                    finally:
                        # Ensure we close aiohttp session / connectors
                        try:
                            await collector.close()
                        except Exception:
                            pass
                except Exception:
                    raw = []

            # Preserve raw (unfiltered) data for analytics like IV surface
            self.raw_options_data = raw.copy() if raw else []

            # Expiry choices
            expiries = sorted({to_date_str(item.get("expiry")) for item in raw if item.get("expiry")})
            self.expiry_options = expiries
            if not self.selected_expiry or self.selected_expiry not in expiries:
                self.selected_expiry = expiries[0] if expiries else ""

            # Filter by selected expiry
            filtered = [
                item for item in raw
                if (self.selected_expiry == "" or to_date_str(item.get("expiry")) == self.selected_expiry)
            ]

            # Determine a spot price if present
            try:
                # Prefer any underlying_price present in data
                spots = [float(d.get("underlying_price", 0) or 0) for d in filtered if isinstance(d, dict)]
                spots = [s for s in spots if s > 0]
                if spots:
                    self.spot_price = sum(spots) / len(spots)
                else:
                    # Fallback: try Deribit index price
                    try:
                        from src.data.collectors.deribit_collector import DeribitCollector
                        _c = DeribitCollector(testnet=False)
                        self.spot_price = float(await _c.get_index_price(self.selected_currency))
                        try:
                            await _c.close()
                        except Exception:
                            pass
                    except Exception:
                        self.spot_price = 0.0
            except Exception:
                self.spot_price = 0.0

            # Normalize fields and split calls/puts
            def to_decimal_iv(x) -> float:
                try:
                    iv = x if x is not None else 0.0
                    iv = float(iv)
                    return iv / 100.0 if iv > 1 else iv
                except Exception:
                    return 0.0

            def norm_row(opt: Dict) -> Dict:
                bid = float(opt.get("bid", 0) or 0)
                ask = float(opt.get("ask", 0) or 0)
                iv = opt.get("mark_iv")
                if iv is None:
                    iv = opt.get("implied_volatility")
                ivd = to_decimal_iv(iv or 0)
                return {
                    "strike": float(opt.get("strike", 0) or 0),
                    "option_type": str(opt.get("option_type", "")).upper(),
                    "expiry": to_date_str(opt.get("expiry")),
                    "bid": f"{bid:.4f}",
                    "ask": f"{ask:.4f}",
                    "iv": f"{ivd:.2%}",
                    "volume": int(opt.get("volume", 0) or 0),
                    "open_interest": int(opt.get("open_interest", 0) or 0),
                }

            normalized = [norm_row(opt) for opt in filtered]
            # Order by strike ascending
            normalized.sort(key=lambda r: r.get("strike", 0))
            self.calls_data = [r for r in normalized if r["option_type"] == "CALL"]
            self.puts_data = [r for r in normalized if r["option_type"] == "PUT"]
            self.options_data = normalized

            # Update counts
            self.calls_count = len(self.calls_data)
            self.puts_count = len(self.puts_data)

            # Metrics based on filtered set
            self.total_contracts = len(normalized)
            ivs = []
            for item in filtered:
                if isinstance(item, dict):
                    val = item.get("mark_iv")
                    if val is None:
                        val = item.get("implied_volatility")
                    if val is not None:
                        ivs.append(to_decimal_iv(val))
            self.avg_iv = (sum(ivs) / len(ivs)) if ivs else 0.0
            ois = [int(item.get("open_interest", 0) or 0) for item in filtered if isinstance(item, dict)]
            self.max_oi = max(ois) if ois else 0
            volumes = [int(item.get("volume", 0) or 0) for item in filtered if isinstance(item, dict)]
            self.total_volume = sum(volumes) if volumes else 0

            # Perform advanced analytics
            await self.analyze_options_chain()
            await self.calculate_portfolio_greeks()

        except Exception as e:
            print(f"Error fetching data: {e}")
            self.db_status = f"Error: {str(e)[:30]}"
            self.options_data = []
            self.calls_data = []
            self.puts_data = []
            self.total_contracts = 0
            self.avg_iv = 0.0
            self.max_oi = 0
            self.total_volume = 0
        finally:
            self.loading = False
            self.last_update = datetime.now().strftime("%H:%M:%S")


class RiskState(rx.State):
    """Risk analytics page state with portfolio risk management"""

    # UI State
    selected_portfolio: str = "default"
    selected_benchmark: str = "BTC"
    loading: bool = False
    optimization_loading: bool = False
    auto_refresh: bool = False
    calculation_status: str = "Ready"

    # Portfolio Options
    portfolio_options: List[str] = ["default", "conservative", "aggressive"]

    # Risk Metrics
    portfolio_value: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    beta: float = 0.0
    r_squared: float = 0.0

    # Performance Metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0

    # Chart Data
    risk_distribution_data: List[Dict] = []
    var_history_data: List[Dict] = []
    correlation_matrix_data: List[Dict] = []
    correlation_columns: List[Dict] = []
    performance_comparison_data: List[Dict] = []
    drawdown_data: List[Dict] = []
    current_allocation_data: List[Dict] = []
    suggested_allocation_data: List[Dict] = []
    efficient_frontier_data: List[Dict] = []
    sector_allocation_data: List[Dict] = []
    sector_risk_data: List[Dict] = []

    @rx.var
    def portfolio_value_display(self) -> str:
        """Formatted portfolio value"""
        try:
            if self.portfolio_value > 1e6:
                return f"${self.portfolio_value/1e6:.1f}M"
            elif self.portfolio_value > 1e3:
                return f"${self.portfolio_value/1e3:.1f}K"
            else:
                return f"${self.portfolio_value:.2f}"
        except Exception:
            return "$0.00"

    @rx.var
    def var_95_display(self) -> str:
        """Formatted VaR 95%"""
        try:
            return f"-{abs(self.var_95):.2%}"
        except Exception:
            return "0.00%"

    @rx.var
    def cvar_95_display(self) -> str:
        """Formatted CVaR 95%"""
        try:
            return f"-{abs(self.cvar_95):.2%}"
        except Exception:
            return "0.00%"

    @rx.var
    def max_drawdown_display(self) -> str:
        """Formatted max drawdown"""
        try:
            return f"-{abs(self.max_drawdown):.2%}"
        except Exception:
            return "0.00%"

    @rx.var
    def sharpe_ratio_display(self) -> str:
        """Formatted Sharpe ratio"""
        try:
            return f"{self.sharpe_ratio:.2f}"
        except Exception:
            return "0.00"

    @rx.var
    def sortino_ratio_display(self) -> str:
        """Formatted Sortino ratio"""
        try:
            return f"{self.sortino_ratio:.2f}"
        except Exception:
            return "0.00"

    @rx.var
    def beta_display(self) -> str:
        """Formatted beta"""
        try:
            return f"{self.beta:.2f}"
        except Exception:
            return "0.00"

    @rx.var
    def r_squared_display(self) -> str:
        """Formatted R-squared"""
        try:
            return f"{self.r_squared:.2%}"
        except Exception:
            return "0.00%"

    @rx.var
    def total_return_display(self) -> str:
        """Formatted total return"""
        try:
            return f"{self.total_return:.2%}"
        except Exception:
            return "0.00%"

    @rx.var
    def annual_return_display(self) -> str:
        """Formatted annual return"""
        try:
            return f"{self.annual_return:.2%}"
        except Exception:
            return "0.00%"

    @rx.var
    def volatility_display(self) -> str:
        """Formatted volatility"""
        try:
            return f"{self.volatility:.2%}"
        except Exception:
            return "0.00%"

    @rx.var
    def win_rate_display(self) -> str:
        """Formatted win rate"""
        try:
            return f"{self.win_rate:.1%}"
        except Exception:
            return "0.0%"

    async def set_portfolio(self, portfolio: str):
        """Set selected portfolio"""
        self.selected_portfolio = portfolio
        await self.calculate_portfolio_risk()

    async def set_benchmark(self, benchmark: str):
        """Set selected benchmark"""
        self.selected_benchmark = benchmark
        await self.calculate_portfolio_risk()

    async def toggle_auto_refresh(self, enabled: bool):
        """Toggle auto refresh"""
        self.auto_refresh = enabled

    async def calculate_portfolio_risk(self):
        """Calculate comprehensive portfolio risk metrics"""
        if self.loading:
            return

        self.loading = True
        self.calculation_status = "Calculating..."

        try:
            # Initialize database operations and analytics modules
            import sys
            import os
            # Add src to path for imports
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

            from src.core.database.operations import DatabaseOperations
            from src.analytics.risk.portfolio_risk import PortfolioRiskAnalyzer
            from src.analytics.performance.quantstats_analyzer import QuantStatsAnalyzer
            from src.core.database.connection import get_database_async

            # Get database connection
            db = await get_database_async()
            db_ops = DatabaseOperations()

            risk_analyzer = PortfolioRiskAnalyzer(db)
            performance_analyzer = QuantStatsAnalyzer(db)

            # Check if portfolio exists, create sample if not
            portfolio_data = await db_ops.get_portfolio_data(self.selected_portfolio)
            if not portfolio_data:
                # Create sample portfolio data for demonstration
                sample_portfolio = {
                    'portfolio_id': self.selected_portfolio,
                    'user_id': 'demo_user',
                    'assets': ['BTC', 'ETH', 'SOL', 'AVAX'],
                    'weights': {'BTC': 0.4, 'ETH': 0.3, 'SOL': 0.2, 'AVAX': 0.1},
                    'total_value': 100000.0,
                    'cash_position': 5000.0,
                    'currency': 'USD'
                }
                await db_ops.store_portfolio_data(sample_portfolio)
                self.portfolio_value = sample_portfolio['total_value']

            # Calculate risk metrics
            risk_metrics = await risk_analyzer.calculate_portfolio_metrics(
                portfolio_id=self.selected_portfolio
            )

            if risk_metrics:
                # Update risk metrics with correct key mapping
                self.var_95 = risk_metrics.get('var_5', 0.0)
                self.var_99 = risk_metrics.get('var_1', 0.0)
                self.cvar_95 = risk_metrics.get('cvar_5', 0.0)
                self.max_drawdown = risk_metrics.get('max_drawdown', 0.0)
                self.sharpe_ratio = risk_metrics.get('sharpe_ratio', 0.0)
                self.sortino_ratio = risk_metrics.get('sortino_ratio', 0.0)
                self.beta = risk_metrics.get('beta', 1.0) or 1.0
                self.r_squared = risk_metrics.get('r_squared', 0.0) or 0.0

                # Update chart data
                await self._update_risk_charts()

            # Calculate performance metrics
            performance_report = await performance_analyzer.generate_performance_report(
                portfolio_id=self.selected_portfolio,
                benchmark_symbol=self.selected_benchmark
            )

            if performance_report:
                self.total_return = performance_report.get('total_return', 0.0)
                self.annual_return = performance_report.get('annual_return', 0.0)
                self.volatility = performance_report.get('annual_volatility', 0.0)
                self.win_rate = performance_report.get('win_rate', 0.0)

                # Update performance charts
                await self._update_performance_charts()

            # Generate comprehensive analytics if portfolio has data
            if portfolio_data or risk_metrics:
                # Calculate performance attribution
                await self._update_attribution_analysis()

                # Update sector analysis
                await self._update_sector_analysis()

            self.calculation_status = "Complete"

        except Exception as e:
            print(f"Error calculating portfolio risk: {e}")
            self.calculation_status = f"Error: {str(e)[:20]}"
        finally:
            self.loading = False

    async def _update_attribution_analysis(self):
        """Update performance attribution data"""
        try:
            import sys
            import os
            # Ensure src is on path for absolute imports
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
            from src.analytics.performance.quantstats_analyzer import QuantStatsAnalyzer
            from src.core.database.connection import get_database_async

            db = await get_database_async()
            performance_analyzer = QuantStatsAnalyzer(db)

            # Generate sector attribution
            sector_attribution = await performance_analyzer.generate_performance_attribution(
                self.selected_portfolio, "sector", 252
            )

            # Update state with attribution data (simplified for demo)
            if sector_attribution:
                # Convert attribution data to chart format
                attribution_chart_data = []
                for sector, data in sector_attribution.items():
                    if isinstance(data, dict):
                        attribution_chart_data.append({
                            'sector': sector,
                            'contribution': data.get('contribution_pct', 0),
                            'return': data.get('return', 0) * 100,
                            'weight': data.get('weight', 0) * 100
                        })

                # Update chart data (this would populate attribution charts in UI)
                # For now, we'll just update sector data
                self.sector_allocation_data = attribution_chart_data[:4]  # Top 4 sectors

        except Exception as e:
            print(f"Error updating attribution analysis: {e}")

    async def _update_sector_analysis(self):
        """Update sector-based analysis"""
        try:
            import sys
            import os
            # Add src to path for imports
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

            from src.core.config.crypto_sectors import get_portfolio_sector_allocation
            from src.core.database.operations import DatabaseOperations

            db_ops = DatabaseOperations()
            portfolio_data = await db_ops.get_portfolio_data(self.selected_portfolio)

            if portfolio_data and portfolio_data.get('weights'):
                # Calculate sector allocation
                sector_allocation = get_portfolio_sector_allocation(portfolio_data['weights'])

                # Convert to chart data
                self.sector_allocation_data = [
                    {'sector': sector, 'allocation': allocation * 100}
                    for sector, allocation in sector_allocation.items()
                    if allocation > 0.01  # Only sectors with >1% allocation
                ]

                # Calculate sector risk (simplified)
                sector_risk_data = []
                for sector, allocation in sector_allocation.items():
                    if allocation > 0.01:
                        # Simple risk calculation based on allocation and sector type
                        risk_multiplier = 1.5 if sector in ['DeFi', 'Gaming'] else 1.0
                        risk_contribution = allocation * risk_multiplier * 50  # Normalized to 0-50
                        sector_risk_data.append({
                            'sector': sector,
                            'risk_contribution': risk_contribution
                        })

                self.sector_risk_data = sector_risk_data

        except Exception as e:
            print(f"Error updating sector analysis: {e}")

    async def optimize_portfolio(self):
        """Optimize portfolio allocation using riskfolio-lib"""
        if self.optimization_loading:
            return

        self.optimization_loading = True

        try:
            import sys
            import os
            # Add src to path for imports
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

            from src.core.database.connection import get_database_async
            from src.analytics.risk.portfolio_risk import PortfolioRiskAnalyzer

            db = await get_database_async()
            risk_analyzer = PortfolioRiskAnalyzer(db)

            # Get current portfolio data and run optimization
            # This would implement HRP/HERC optimization from the roadmap
            # For now, set sample data
            self.suggested_allocation_data = [
                {"asset": "BTC", "weight": 0.4},
                {"asset": "ETH", "weight": 0.3},
                {"asset": "SOL", "weight": 0.2},
                {"asset": "AVAX", "weight": 0.1},
            ]

            # Generate efficient frontier data
            await self._generate_efficient_frontier()

        except Exception as e:
            print(f"Error optimizing portfolio: {e}")
        finally:
            self.optimization_loading = False

    async def _update_risk_charts(self):
        """Update risk-related chart data"""
        # Sample data - in production, this would come from database
        self.risk_distribution_data = [
            {"risk_type": "Market Risk", "value": 60},
            {"risk_type": "Liquidity Risk", "value": 25},
            {"risk_type": "Concentration Risk", "value": 15},
        ]

        self.var_history_data = [
            {"date": "2025-01-01", "var_95": -0.05, "var_99": -0.08},
            {"date": "2025-01-02", "var_95": -0.04, "var_99": -0.07},
            {"date": "2025-01-03", "var_95": -0.06, "var_99": -0.09},
        ]

        # Generate correlation matrix sample data
        assets = ["BTC", "ETH", "SOL", "AVAX"]
        self.correlation_matrix_data = []
        self.correlation_columns = [{"name": "Asset", "id": "asset"}]

        for asset in assets:
            self.correlation_columns.append({"name": asset, "id": asset.lower()})

        for i, asset1 in enumerate(assets):
            row = {"asset": asset1}
            for j, asset2 in enumerate(assets):
                corr = 1.0 if i == j else 0.7 - abs(i-j) * 0.2
                row[asset2.lower()] = f"{corr:.2f}"
            self.correlation_matrix_data.append(row)

    async def _update_performance_charts(self):
        """Update performance-related chart data from REAL returns."""
        try:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
            from src.core.database.connection import get_database_async
            from src.analytics.performance.quantstats_analyzer import QuantStatsAnalyzer
            import pandas as pd
            import numpy as np

            db = await get_database_async()
            analyzer = QuantStatsAnalyzer(db)

            # Get returns series (use internal methods for consistency)
            port_returns = await analyzer._get_portfolio_returns(self.selected_portfolio, 252)
            bench_returns = await analyzer._get_benchmark_returns(self.selected_benchmark, 252)

            if port_returns is None or len(port_returns) == 0:
                self.performance_comparison_data = []
                self.drawdown_data = []
                return

            # Align indexes
            df = pd.DataFrame({'portfolio': port_returns})
            if bench_returns is not None and len(bench_returns) > 0:
                df['benchmark'] = bench_returns
            else:
                df['benchmark'] = np.nan
            df = df.dropna(subset=['portfolio'])

            # Compute cumulative NAV
            df['portfolio_nav'] = (1.0 + df['portfolio']).cumprod()
            if df['benchmark'].notna().any():
                df['benchmark_nav'] = (1.0 + df['benchmark'].fillna(0)).cumprod()
            else:
                df['benchmark_nav'] = np.nan

            # Build performance comparison data
            self.performance_comparison_data = [
                {
                    'date': idx.strftime('%Y-%m-%d'),
                    'portfolio': float(row['portfolio_nav']),
                    'benchmark': (float(row['benchmark_nav']) if not np.isnan(row['benchmark_nav']) else None)
                }
                for idx, row in df[['portfolio_nav', 'benchmark_nav']].iterrows()
            ]

            # Drawdown from portfolio NAV
            nav = df['portfolio_nav']
            running_max = nav.cummax()
            drawdown = (nav / running_max) - 1.0
            self.drawdown_data = [
                {'date': idx.strftime('%Y-%m-%d'), 'drawdown': float(val)}
                for idx, val in drawdown.items()
            ]

        except Exception as e:
            print(f"Error updating performance charts: {e}")

    async def _generate_efficient_frontier(self):
        """Generate efficient frontier data"""
        # Sample efficient frontier points
        self.efficient_frontier_data = [
            {"risk": 0.10, "return": 0.05},
            {"risk": 0.15, "return": 0.08},
            {"risk": 0.20, "return": 0.12},
            {"risk": 0.25, "return": 0.15},
            {"risk": 0.30, "return": 0.17},
        ]

        # Sample sector data
        self.sector_allocation_data = [
            {"sector": "DeFi", "allocation": 40},
            {"sector": "Infrastructure", "allocation": 35},
            {"sector": "AI", "allocation": 15},
            {"sector": "Gaming", "allocation": 10},
        ]

        self.sector_risk_data = [
            {"sector": "DeFi", "risk_contribution": 45},
            {"sector": "Infrastructure", "risk_contribution": 30},
            {"sector": "AI", "risk_contribution": 15},
            {"sector": "Gaming", "risk_contribution": 10},
        ]


class State(rx.State):
    """Main app state"""
    pass
