"""Portfolio Management State"""

import reflex as rx
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
import logging

# Import backend service (NO CALCULATIONS IN FRONTEND)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.services.portfolio_optimization_service import get_optimization_service
    OPTIMIZATION_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import optimization service: {e}")
    OPTIMIZATION_SERVICE_AVAILABLE = False

logger = logging.getLogger(__name__)


class PortfolioState(rx.State):
    """Portfolio management state (Mongo-backed with fallbacks)."""

    # Portfolio data
    positions: List[Dict] = []
    total_value: float = 0.0
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0

    # Allocations
    crypto_allocation: List[Dict] = []
    sector_allocation: List[Dict] = []
    sector_risk_contribution: List[Dict] = []

    # Performance time series
    portfolio_vs_btc: List[Dict] = []  # {date, portfolio_pct, btc_pct}

    # Performance metrics
    daily_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0

    # UI state
    loading: bool = False
    selected_view: str = "positions"  # positions, allocation, performance
    selected_asset_type: str = "all"   # all, spot, options
    selected_period: str = "30d"       # 7d, 30d, 90d, 1y

    # Portfolios
    portfolio_options: List[str] = ["default"]
    selected_portfolio: str = "default"
    new_portfolio_name: str = ""

    # Benchmarks
    benchmark_options: List[str] = ["BTC-USD", "ETH-USD", "SPY", "GC=F"]
    selected_benchmark: str = "BTC-USD"

    # Trade inputs
    trade_symbol: str = "BTC"
    trade_quantity: float = 0.0
    trade_price: float = 0.0
    trade_side: str = "buy"  # buy or sell

    @rx.var
    def total_value_display(self) -> str:
        return f"${self.total_value:,.2f}"

    @rx.var
    def pnl_display(self) -> str:
        sign = "+" if self.total_pnl >= 0 else ""
        return f"{sign}${abs(self.total_pnl):,.2f}"

    @rx.var
    def pnl_percent_display(self) -> str:
        sign = "+" if self.total_pnl_percent >= 0 else ""
        return f"{sign}{abs(self.total_pnl_percent):.2f}%"

    @rx.var
    def sharpe_display(self) -> str:
        return f"{self.sharpe_ratio:.2f}"

    def set_view(self, view: str):
        self.selected_view = view

    def set_asset_type(self, asset_type: str):
        self.selected_asset_type = asset_type
        return PortfolioState.fetch_portfolio_data()

    def set_period(self, period: str):
        self.selected_period = period
        return PortfolioState.fetch_portfolio_data()

    async def refresh(self):
        await self.load_portfolios()
        await self.fetch_portfolio_data()

    def set_portfolio(self, portfolio_id: str):
        self.selected_portfolio = portfolio_id
        return PortfolioState.fetch_portfolio_data()

    def set_benchmark(self, bench: str):
        self.selected_benchmark = bench
        return PortfolioState.fetch_portfolio_data()

    def set_trade_symbol(self, sym: str):
        self.trade_symbol = sym

    def set_trade_quantity(self, qty):
        try:
            self.trade_quantity = float(qty)
        except Exception:
            self.trade_quantity = 0.0

    def set_trade_price(self, px):
        try:
            self.trade_price = float(px)
        except Exception:
            self.trade_price = 0.0

    def set_trade_side(self, side: str):
        self.trade_side = side

    async def load_portfolios(self):
        """Load list of available portfolios from DB."""
        try:
            from src.core.database.operations import DatabaseOperations
            ops = DatabaseOperations()
            portfolios = await ops.get_portfolio_list()
            ids = [p.get('portfolio_id') for p in portfolios if p.get('portfolio_id')]
            self.portfolio_options = ids or ["default"]
            if self.selected_portfolio not in self.portfolio_options:
                self.selected_portfolio = self.portfolio_options[0]
        except Exception:
            self.portfolio_options = ["default"]

    async def create_portfolio(self):
        """Create a new named portfolio and persist to DB."""
        name = (self.new_portfolio_name or '').strip() or 'portfolio'
        try:
            from src.core.database.operations import DatabaseOperations
            ops = DatabaseOperations()
            doc = {
                'portfolio_id': name,
                'user_id': 'local',
                'assets': [],
                'weights': {},
                'total_value': 0.0,
                'cash_position': 0.0,
                'currency': 'USD'
            }
            await ops.store_portfolio_data(doc)
            self.selected_portfolio = name
            await self.load_portfolios()
        except Exception as e:
            print(f"Create portfolio failed: {e}")

    async def delete_portfolio(self, portfolio_id: str):
        """Delete portfolio and its positions."""
        try:
            from src.core.database.connection import db_connection
            adb = await db_connection.get_database_async()
            await adb.portfolio_data.delete_many({'portfolio_id': portfolio_id})
            await adb.portfolio_positions.delete_many({'portfolio_id': portfolio_id})
            await self.load_portfolios()
            self.selected_portfolio = self.portfolio_options[0]
            await self.fetch_portfolio_data()
        except Exception as e:
            print(f"Delete portfolio failed: {e}")

    async def add_spot_position(self):
        """Add or update a spot position using yfinance price if price not provided."""
        try:
            sym = (self.trade_symbol or '').upper()
            qty = float(self.trade_quantity or 0)
            px = float(self.trade_price or 0)
            if qty == 0:
                return
            # Fetch current price if not provided
            if px <= 0:
                from src.data.providers.yfinance_provider import YFinanceProvider
                yf = YFinanceProvider()
                # Map to Yahoo symbol if pure crypto ticker given
                price_data = yf.get_current_price(sym) or yf.get_current_price(f"{sym}-USD")
                if price_data:
                    px = float(price_data.get('price_usd') or 0)

            from src.core.database.connection import db_connection
            adb = await db_connection.get_database_async()
            side = (self.trade_side or 'buy').lower()
            signed_qty = qty if side == 'buy' else -qty
            await adb.portfolio_positions.insert_one({
                'portfolio_id': self.selected_portfolio,
                'symbol': sym,
                'type': 'Spot',
                'quantity': signed_qty,
                'entry_price': px,
                'current_price': px,
                'timestamp': datetime.utcnow()
            })
            await self.fetch_portfolio_data()
        except Exception as e:
            print(f"Add spot position failed: {e}")

    async def fetch_portfolio_data(self):
        """Fetch portfolio data from MongoDB and compute metrics/series."""
        self.loading = True
        try:
            from src.core.database.connection import db_connection
            db = await db_connection.get_database_async()

            # Fetch positions for selected portfolio
            cursor = db.portfolio_positions.find({'portfolio_id': self.selected_portfolio})
            positions = await cursor.to_list(length=None)
            if not positions:
                # Fallback sample if no positions saved
                positions = [
                    {"symbol": "BTC", "type": "Spot", "quantity": 0.5, "entry_price": 42000, "current_price": 45000},
                    {"symbol": "ETH", "type": "Spot", "quantity": 5, "entry_price": 2800, "current_price": 3000},
                ]

            # Filter by asset type
            atype = self.selected_asset_type.lower()
            if atype != "all":
                positions = [p for p in positions if str(p.get("type", "")).lower().startswith(atype)]

            # Compute position values and totals
            vals = []
            total_value = 0.0
            total_cost = 0.0
            # Aggregate by symbol for simple net position
            by_sym: Dict[str, Dict] = {}
            for p in positions:
                sym = p.get("symbol", "")
                if sym not in by_sym:
                    by_sym[sym] = {"symbol": sym, "type": p.get("type", "Spot"), "quantity": 0.0, "cost": 0.0, "entry_price": 0.0, "current_price": float(p.get("current_price", 0) or 0)}
                qty = float(p.get("quantity", 0) or 0)
                px = float(p.get("entry_price", 0) or 0)
                by_sym[sym]["quantity"] += qty
                by_sym[sym]["cost"] += qty * px
                by_sym[sym]["current_price"] = float(p.get("current_price", by_sym[sym]["current_price"]) or by_sym[sym]["current_price"])

            for p in by_sym.values():
                qty = float(p.get("quantity", 0) or 0)
                entry = float(p.get("cost", 0) / qty) if qty != 0 else float(p.get("entry_price", 0) or 0)
                current = float(p.get("current_price", entry) or entry)
                value = qty * current if p.get("type", "").lower().startswith("spot") else float(p.get("value", 0) or 0)
                pnl = qty * (current - entry) if p.get("type", "").lower().startswith("spot") else float(p.get("pnl", 0) or 0)
                vals.append({
                    "symbol": p.get("symbol", ""),
                    "type": p.get("type", "Unknown"),
                    "quantity": qty,
                    "entry_price": entry,
                    "current_price": current,
                    "value": value,
                    "pnl": pnl,
                    "pnl_percent": (pnl / (qty*entry) * 100) if qty*entry > 0 else 0.0,
                })
                total_value += value
                total_cost += qty * entry

            # Allocation
            crypto_alloc = {}
            for v in vals:
                sym = v["symbol"]
                crypto_alloc[sym] = crypto_alloc.get(sym, 0.0) + v["value"]
            # Include USDT cash from portfolio_data if any
            try:
                from src.core.database.operations import DatabaseOperations
                ops = DatabaseOperations()
                pdata = await ops.get_portfolio_data(self.selected_portfolio)
                cash = float(pdata.get('cash_position', 0) or 0) if pdata else 0.0
                if cash > 0:
                    crypto_alloc['USDT'] = crypto_alloc.get('USDT', 0.0) + cash
                    total_value = total_value + cash
            except Exception:
                pass
            self.crypto_allocation = [
                {"name": k, "value": round((v/total_value)*100.0, 2)} for k, v in crypto_alloc.items() if total_value > 0
            ]

            # Sector allocation and contributions
            try:
                from src.core.config.crypto_sectors import get_asset_sector
                sector_val: Dict[str, float] = {}
                sector_pnl: Dict[str, float] = {}
                for r in vals:
                    sec = get_asset_sector(r['symbol'])
                    sector_val[sec] = sector_val.get(sec, 0.0) + float(r['value'] or 0)
                    sector_pnl[sec] = sector_pnl.get(sec, 0.0) + float(r['pnl'] or 0)
                self.sector_allocation = [
                    {"name": s, "value": round((val/total_value)*100.0, 2)} for s, val in sector_val.items() if total_value > 0
                ]
                # Optional: could store sector risk/pnl contribution for enhanced charts
                self.sector_risk_contribution = [
                    {"sector": s, "pnl": round(sector_pnl.get(s, 0.0), 2)} for s in sector_val.keys()
                ]
            except Exception:
                self.sector_allocation = []

            # Save positions and totals
            self.positions = vals
            self.total_value = total_value
            self.total_pnl = sum(v["pnl"] for v in vals)
            self.total_pnl_percent = (self.total_pnl / total_cost * 100.0) if total_cost > 0 else 0.0

            # Build performance series (spot-only approximation)
            days = {"7d": 7, "30d": 30, "90d": 90, "1y": 365}.get(self.selected_period, 30)
            # symbols to include for series: spot assets only
            spot_symbols = [v["symbol"] for v in vals if v["type"].lower().startswith("spot")]
            start = datetime.utcnow() - timedelta(days=days)
            series_map: Dict[str, List[Dict]] = {}
            for sym in spot_symbols:
                cur = db.price_data.find({"symbol": sym, "timestamp": {"$gte": start}}).sort("timestamp", 1)
                rows = await cur.to_list(length=None)
                if not rows:
                    continue
                series_map[sym] = [{"date": r["timestamp"], "close": float(r.get("close", 0) or 0)} for r in rows if r.get("close")]

            # Align dates (daily) and compute weighted index
            # Weight by latest allocation
            weights: Dict[str, float] = {}
            for sym in spot_symbols:
                val = crypto_alloc.get(sym, 0.0)
                weights[sym] = (val/total_value) if total_value > 0 else 0.0

            # Build date set
            date_set = set()
            for s in series_map.values():
                for r in s:
                    date_set.add(r["date"].strftime("%Y-%m-%d"))
            dates = sorted(date_set)
            # Helper: get price by date per sym
            def get_price(sym: str, d: str) -> float:
                for r in series_map.get(sym, []):
                    if r["date"].strftime("%Y-%m-%d") == d:
                        return r["close"]
                return 0.0

            # Base index at 100
            if dates:
                base_prices = {sym: get_price(sym, dates[0]) for sym in spot_symbols}
                portfolio_series = []
                nav_values = []
                for d in dates:
                    idx = 0.0
                    for sym in spot_symbols:
                        p0 = base_prices.get(sym, 0.0)
                        pt = get_price(sym, d)
                        if p0 > 0 and pt > 0:
                            idx += weights.get(sym, 0.0) * (pt / p0)
                    portfolio_series.append({"date": d, "portfolio_pct": round((idx-1.0)*100.0, 2)})
                    nav_values.append(idx)
            else:
                portfolio_series = []
                nav_values = []

            # Benchmark series (from selected_benchmark)
            bench_symbol = self.selected_benchmark
            bench_cur = db.price_data.find({"symbol": bench_symbol, "timestamp": {"$gte": start}}).sort("timestamp", 1)
            bench_rows = await bench_cur.to_list(length=None)
            # Backfill via yfinance if empty
            if len(bench_rows) < 2:
                try:
                    from src.data.providers.yfinance_provider import YFinanceProvider
                    yfp = YFinanceProvider()
                    hist = yfp.get_historical_data(bench_symbol, period=f"{days}d", interval="1d")
                    if not hist.empty:
                        # Store to DB
                        docs = []
                        for _, row in hist.iterrows():
                            ts = row['timestamp'] if 'timestamp' in row else _
                            docs.append({'symbol': bench_symbol, 'timestamp': ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts, 'close': float(row.get('close', row.get('price_usd', 0)) or 0)})
                        if docs:
                            await db.price_data.insert_many(docs)
                        bench_rows = await bench_cur.to_list(length=None)
                except Exception:
                    pass
            bench_series = []
            if bench_rows:
                base = float(bench_rows[0].get("close", 0) or 0)
                for r in bench_rows:
                    if base > 0 and r.get("close"):
                        bench_series.append({"date": r["timestamp"].strftime("%Y-%m-%d"), "bench_pct": round(((float(r["close"]) / base) - 1.0)*100.0, 2)})

            # Merge portfolio and benchmark
            bench_map = {r["date"]: r["bench_pct"] for r in bench_series}
            merged = []
            for row in portfolio_series:
                d = row["date"]
                merged.append({"date": d, "portfolio_pct": row["portfolio_pct"], "benchmark_pct": bench_map.get(d)})
            self.portfolio_vs_btc = merged

            # Current portfolio point for efficient frontier overlay
            try:
                if len(nav_values) > 2:
                    import numpy as _np
                    rets = _np.diff(_np.array(nav_values)) / _np.array(nav_values[:-1])
                    vol = float(_np.std(rets, ddof=1))
                    total_ret = float(nav_values[-1] - 1.0)
                    # Store as a single-point frontier overlay by appending marker in efficient_frontier_data_meta
                    self.current_portfolio_point = {"risk": vol, "return": total_ret}
                else:
                    self.current_portfolio_point = {"risk": 0.0, "return": 0.0}
            except Exception:
                self.current_portfolio_point = {"risk": 0.0, "return": 0.0}

        except Exception as e:
            print(f"Error fetching portfolio data: {e}")
            # Keep previous or sample values if desired
        finally:
            self.loading = False

    # ===============================
    # ADVANCED PORTFOLIO OPTIMIZATION
    # ===============================

    # Optimization state variables
    optimization_loading: bool = False
    optimization_method: str = "HRP"  # HRP, HERC, Sharpe, MinRisk
    optimization_results: List[Dict] = []
    current_optimization_result: Optional[Dict] = None
    suggested_allocation: List[Dict] = []
    optimization_comparison: List[Dict] = []
    efficient_frontier_data: List[Dict] = []
    current_portfolio_point: Dict = {}

    # Optimization configuration
    risk_free_rate: float = 0.05
    risk_aversion: float = 2.0
    lookback_days: int = 365
    min_weight: float = 0.01
    max_weight: float = 0.40

    @rx.var
    def optimization_status(self) -> str:
        """Current optimization status display"""
        if self.optimization_loading:
            return "Optimizing..."
        if self.current_optimization_result:
            method = self.current_optimization_result.get('method_used', 'Unknown')
            sharpe = self.current_optimization_result.get('sharpe_ratio', 0)
            return f"{method} - Sharpe: {sharpe:.3f}"
        return "Not optimized"

    @rx.var
    def optimization_methods(self) -> List[str]:
        """Available optimization methods"""
        return ["HRP", "HERC", "Sharpe", "MinRisk", "Utility", "MaxRet"]

    async def run_portfolio_optimization(self):
        """Run portfolio optimization using backend service (NO CALCULATIONS HERE)"""
        if not OPTIMIZATION_SERVICE_AVAILABLE:
            print("Optimization service not available")
            return

        self.optimization_loading = True
        yield

        try:
            # Get optimization service
            service = await get_optimization_service()

            # Get current portfolio assets
            assets = [pos["symbol"] + "-USD" if not pos["symbol"].endswith("-USD")
                     else pos["symbol"] for pos in self.positions if pos.get("type", "").lower().startswith("spot")]

            if not assets:
                # Get assets from service
                assets = await service.get_portfolio_assets("default_portfolio")

            # Call backend service (ALL CALCULATIONS HAPPEN IN BACKEND)
            result = await service.run_single_optimization(
                portfolio_id="default_portfolio",
                assets=assets,
                method=self.optimization_method,
                risk_free_rate=self.risk_free_rate,
                risk_aversion=self.risk_aversion,
                lookback_days=self.lookback_days,
                min_weight=self.min_weight,
                max_weight=self.max_weight,
                sector_constraints={
                    'Infrastructure': {'min': 0.10, 'max': 0.50},
                    'DeFi': {'max': 0.30},
                    'Meme': {'max': 0.15}
                }
            )

            # Update UI state with results (NO CALCULATIONS, JUST UI STATE)
            if result['success']:
                self.current_optimization_result = {
                    'method_used': result['method_used'],
                    'objective_used': result['objective_used'],
                    'expected_return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'optimization_time': result['optimization_time'],
                    'timestamp': result['timestamp'],
                    'sector_allocation': result['sector_allocation']
                }

                self.suggested_allocation = result['suggested_allocation']
                self.optimization_results.append(self.current_optimization_result)

                logger.info(f"Portfolio optimization completed: {result['method_used']}")
            else:
                logger.error(f"Optimization failed: {result.get('error_message', 'Unknown error')}")

        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            print(f"Optimization error: {e}")

        finally:
            self.optimization_loading = False
            yield

    async def run_multi_method_comparison(self):
        """Run multiple optimization methods comparison using backend service (NO CALCULATIONS HERE)"""
        if not OPTIMIZATION_SERVICE_AVAILABLE:
            return

        self.optimization_loading = True
        yield

        try:
            # Get optimization service
            service = await get_optimization_service()

            # Get current portfolio assets
            assets = [pos["symbol"] + "-USD" if not pos["symbol"].endswith("-USD")
                     else pos["symbol"] for pos in self.positions if pos.get("type", "").lower().startswith("spot")]

            if not assets:
                assets = await service.get_portfolio_assets("comparison_portfolio")

            # Call backend service for comparison (ALL CALCULATIONS HAPPEN IN BACKEND)
            methods = ['HRP', 'HERC', 'Sharpe', 'MinRisk']
            comparison_data = await service.run_multi_method_comparison(
                portfolio_id="comparison_portfolio",
                assets=assets,
                methods=methods,
                risk_free_rate=self.risk_free_rate,
                lookback_days=self.lookback_days,
                sector_constraints={
                    'Infrastructure': {'min': 0.10, 'max': 0.50},
                    'DeFi': {'max': 0.30}
                }
            )

            # Update UI state with results (NO CALCULATIONS, JUST UI STATE)
            self.optimization_comparison = comparison_data

            # Generate efficient frontier using backend service
            try:
                frontier_data = await service.generate_efficient_frontier(
                    portfolio_id="comparison_portfolio",
                    assets=assets,
                    points=20
                )
                self.efficient_frontier_data = frontier_data
            except Exception as e:
                logger.warning(f"Could not generate efficient frontier: {e}")

            logger.info(f"Multi-method comparison completed with {len(comparison_data)} methods")

        except Exception as e:
            logger.error(f"Error in multi-method comparison: {e}")
            print(f"Comparison error: {e}")

        finally:
            self.optimization_loading = False
            yield

    def set_optimization_method(self, method: str):
        """Set the optimization method"""
        self.optimization_method = method

    def set_optimization_parameter(self, param: str, value):
        """Set optimization parameters (parse from UI event string safely)."""
        def _to_float(v, default: float) -> float:
            try:
                s = str(v).strip()
                if s == "" or s.lower() == "none":
                    return default
                return float(s)
            except Exception:
                return default

        if param == "risk_free_rate":
            self.risk_free_rate = _to_float(value, 0.05)
        elif param == "risk_aversion":
            self.risk_aversion = _to_float(value, 2.0)
        elif param == "lookback_days":
            self.lookback_days = int(_to_float(value, 365))
        elif param == "min_weight":
            self.min_weight = _to_float(value, 0.01)
        elif param == "max_weight":
            self.max_weight = _to_float(value, 0.40)

    # ===============================
    # EXISTING METHODS
    # ===============================

    def add_position(self):
        pass

    def close_position(self, symbol: str):
        self.positions = [p for p in self.positions if p["symbol"] != symbol]
        return PortfolioState.fetch_portfolio_data()
