"""Enhanced State Management with MongoDB + Deribit fallback"""

import reflex as rx
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

    def set_currency(self, currency: str):
        """Set selected currency and fetch data"""
        self.selected_currency = currency
        # Return an event spec to avoid Reflex wrapping a bound method/partial.
        return self.fetch_options_data()

    def set_expiry(self, expiry: str):
        """Set selected expiry and refresh data"""
        self.selected_expiry = expiry
        return self.fetch_options_data()

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
            surface = await builder.build_and_store_surface(
                self.selected_currency,
                self.options_data
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


class State(rx.State):
    """Main app state"""
    pass
