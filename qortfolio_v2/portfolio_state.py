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

    async def fetch_portfolio_data(self):
        """Fetch portfolio data from MongoDB and compute metrics/series."""
        self.loading = True
        try:
            from src.core.database.connection import db_connection
            db = await db_connection.get_database_async()

            # Fetch positions
            cursor = db.portfolio_positions.find({})
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
            for p in positions:
                qty = float(p.get("quantity", 0) or 0)
                entry = float(p.get("entry_price", 0) or 0)
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
            self.crypto_allocation = [
                {"name": k, "value": round((v/total_value)*100.0, 2)} for k, v in crypto_alloc.items() if total_value > 0
            ]

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
                for d in dates:
                    idx = 0.0
                    for sym in spot_symbols:
                        p0 = base_prices.get(sym, 0.0)
                        pt = get_price(sym, d)
                        if p0 > 0 and pt > 0:
                            idx += weights.get(sym, 0.0) * (pt / p0)
                    portfolio_series.append({"date": d, "portfolio_pct": round((idx-1.0)*100.0, 2)})
            else:
                portfolio_series = []

            # BTC benchmark
            btc_cur = db.price_data.find({"symbol": "BTC", "timestamp": {"$gte": start}}).sort("timestamp", 1)
            btc_rows = await btc_cur.to_list(length=None)
            btc_series = []
            if btc_rows:
                base = float(btc_rows[0].get("close", 0) or 0)
                for r in btc_rows:
                    if base > 0 and r.get("close"):
                        btc_series.append({"date": r["timestamp"].strftime("%Y-%m-%d"), "btc_pct": round(((float(r["close"]) / base) - 1.0)*100.0, 2)})

            # Merge portfolio and BTC
            btc_map = {r["date"]: r["btc_pct"] for r in btc_series}
            merged = []
            for row in portfolio_series:
                d = row["date"]
                merged.append({"date": d, "portfolio_pct": row["portfolio_pct"], "btc_pct": btc_map.get(d)})
            self.portfolio_vs_btc = merged

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

    # Optimization configuration
    risk_free_rate: float = 0.05
    risk_aversion: float = 2.0
    lookback_days: int = 252
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
            self.lookback_days = int(_to_float(value, 252))
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
