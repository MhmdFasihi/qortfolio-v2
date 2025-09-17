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
    selected_asset_type: str = "all"   # all, spot, contracts
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

    # Options trade inputs
    options_strike: float = 0.0
    options_expiry: str = ""
    options_type: str = "call"  # call or put
    options_premium: float = 0.0

    # Asset selection
    selected_position_type: str = "Spot"  # Spot, Options, Futures
    selected_sector_filter: str = "All"  # All, Infrastructure, DeFi, AI, etc.
    custom_asset_input: str = ""
    optimization_assets: List[str] = []

    # Constraints management
    active_constraints: List[Dict] = []
    comparison_results: List[Dict] = []

    # Constraint form inputs
    constraint_asset: str = ""
    constraint_min_weight: float = 0.0
    constraint_max_weight: float = 0.2
    constraint_sector: str = ""
    constraint_sector_min: float = 0.0
    constraint_sector_max: float = 0.25

    def set_new_portfolio_name(self, name: str):
        """Set new portfolio name."""
        self.new_portfolio_name = name

    def set_constraint_asset(self, asset: str):
        """Set constraint asset."""
        self.constraint_asset = asset

    def set_constraint_min_weight(self, weight: str):
        """Set constraint min weight."""
        try:
            self.constraint_min_weight = float(weight) if weight else 0.0
        except ValueError:
            self.constraint_min_weight = 0.0

    def set_constraint_max_weight(self, weight: str):
        """Set constraint max weight."""
        try:
            self.constraint_max_weight = float(weight) if weight else 0.2
        except ValueError:
            self.constraint_max_weight = 0.2

    def set_constraint_sector(self, sector: str):
        """Set constraint sector."""
        self.constraint_sector = sector

    def set_constraint_sector_min(self, weight: str):
        """Set constraint sector min weight."""
        try:
            self.constraint_sector_min = float(weight) if weight else 0.0
        except ValueError:
            self.constraint_sector_min = 0.0

    def set_constraint_sector_max(self, weight: str):
        """Set constraint sector max weight."""
        try:
            self.constraint_sector_max = float(weight) if weight else 0.25
        except ValueError:
            self.constraint_sector_max = 0.25

    async def delete_current_portfolio(self):
        """Delete the currently selected portfolio."""
        await self.delete_portfolio(self.selected_portfolio)

    async def update_current_prices(self):
        """Update current prices for all positions and refresh portfolio data"""
        try:
            from src.data.providers.yfinance_provider import YFinanceProvider
            from src.core.database.connection import db_connection

            # Load latest portfolios first
            await self.load_portfolios()

            yf = YFinanceProvider()
            adb = await db_connection.get_database_async()

            # Get all unique symbols from positions
            positions = await adb.portfolio_positions.find(
                {"portfolio_id": self.selected_portfolio}
            ).to_list(length=None)

            symbols = list(set(pos["symbol"] for pos in positions))

            # Update prices for each symbol
            for symbol in symbols:
                price_data = yf.get_current_price(symbol) or yf.get_current_price(f"{symbol}-USD")
                if price_data:
                    current_price = float(price_data.get('price_usd') or 0)
                    if current_price > 0:
                        # Update all positions with this symbol
                        await adb.portfolio_positions.update_many(
                            {"portfolio_id": self.selected_portfolio, "symbol": symbol},
                            {"$set": {"current_price": current_price, "updated_at": datetime.utcnow()}}
                        )

            # Refresh portfolio data
            await self.fetch_portfolio_data()

        except Exception as e:
            print(f"Error updating current prices: {e}")

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

    def set_options_strike(self, strike):
        try:
            self.options_strike = float(strike)
        except Exception:
            self.options_strike = 0.0

    def set_options_expiry(self, expiry: str):
        self.options_expiry = expiry

    def set_options_type(self, opt_type: str):
        self.options_type = opt_type

    def set_options_premium(self, premium):
        try:
            self.options_premium = float(premium)
        except Exception:
            self.options_premium = 0.0

    def set_position_type(self, ptype: str):
        """Set the position type (Spot, Options, Futures)"""
        self.selected_position_type = ptype

    def set_sector_filter(self, sector: str):
        """Set the sector filter for asset selection"""
        self.selected_sector_filter = sector

    def add_custom_benchmark(self):
        """Add a custom benchmark to the list"""
        # For now, just add a few more default options
        additional_benchmarks = ["QQQ", "IWM", "TLT", "VTI", "TSLA", "NVDA"]
        for bench in additional_benchmarks:
            if bench not in self.benchmark_options:
                self.benchmark_options.append(bench)

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

    async def add_options_position(self):
        """Add or update an options position"""
        try:
            sym = (self.trade_symbol or '').upper()
            qty = float(self.trade_quantity or 0)
            strike = float(self.options_strike or 0)
            premium = float(self.options_premium or 0)

            if qty == 0 or strike == 0:
                return

            # Create options symbol (e.g., BTC-240329-50000-C)
            expiry_clean = self.options_expiry.replace('-', '')
            option_type_short = 'C' if self.options_type.lower() == 'call' else 'P'
            options_symbol = f"{sym}-{expiry_clean}-{int(strike)}-{option_type_short}"

            from src.core.database.connection import db_connection
            adb = await db_connection.get_database_async()
            side = (self.trade_side or 'buy').lower()
            signed_qty = qty if side == 'buy' else -qty

            # For options, premium is the entry price
            # Value calculation will be handled differently based on current option price
            await adb.portfolio_positions.insert_one({
                'portfolio_id': self.selected_portfolio,
                'symbol': options_symbol,
                'underlying': sym,
                'type': 'Option',
                'quantity': signed_qty,
                'entry_price': premium,  # Premium paid/received
                'current_price': premium,  # Will be updated with real option price
                'strike_price': strike,
                'expiry_date': self.options_expiry,
                'option_type': self.options_type,
                'timestamp': datetime.utcnow()
            })
            await self.fetch_portfolio_data()
        except Exception as e:
            print(f"Add options position failed: {e}")

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
                if atype == "contracts":
                    # Include options, futures, and derivatives
                    positions = [p for p in positions if str(p.get("type", "")).lower() in ["option", "future", "contract", "derivative"]]
                else:
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

                # Handle different asset types with proper value calculations
                asset_type = p.get("type", "").lower()
                if asset_type.startswith("spot"):
                    value = qty * current
                    pnl = qty * (current - entry)
                elif asset_type.startswith("option"):
                    # For options: value = quantity * premium * multiplier (usually 1 for crypto options)
                    # Premium is typically in the same currency as underlying (BTC for BTC options)
                    value = qty * current
                    pnl = qty * (current - entry)
                else:
                    # Other types (futures, etc.)
                    value = float(p.get("value", qty * current) or qty * current)
                    pnl = float(p.get("pnl", qty * (current - entry)) or qty * (current - entry))
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

                logger.debug(f"Calculating sector allocation for {len(vals)} positions")

                for r in vals:
                    symbol = r['symbol']
                    # Normalize symbol for sector lookup (remove -USD suffix)
                    clean_symbol = symbol.replace('-USD', '').replace('=F', '')
                    sec = get_asset_sector(clean_symbol)
                    value = float(r['value'] or 0)
                    pnl = float(r['pnl'] or 0)

                    logger.debug(f"Asset {symbol} (clean: {clean_symbol}) -> Sector {sec}, Value: {value}, PnL: {pnl}")

                    sector_val[sec] = sector_val.get(sec, 0.0) + value
                    sector_pnl[sec] = sector_pnl.get(sec, 0.0) + pnl

                logger.debug(f"Sector values: {sector_val}")

                if total_value > 0:
                    self.sector_allocation = [
                        {"name": s, "value": round((val/total_value)*100.0, 2)}
                        for s, val in sector_val.items()
                    ]
                    # Optional: could store sector risk/pnl contribution for enhanced charts
                    self.sector_risk_contribution = [
                        {"sector": s, "pnl": round(sector_pnl.get(s, 0.0), 2)}
                        for s in sector_val.keys()
                    ]

                    logger.debug(f"Final sector allocation: {self.sector_allocation}")
                else:
                    self.sector_allocation = []
                    self.sector_risk_contribution = []

            except Exception as e:
                logger.error(f"Error calculating sector allocation: {e}")
                self.sector_allocation = []
                self.sector_risk_contribution = []

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

            # Current portfolio point for efficient frontier overlay and compute performance metrics
            try:
                logger.debug(f"Computing performance metrics with {len(nav_values)} nav values")

                if len(nav_values) > 2:
                    import numpy as _np
                    nav_array = _np.array(nav_values)
                    rets = _np.diff(nav_array) / nav_array[:-1]

                    logger.debug(f"Calculated {len(rets)} returns, mean: {_np.mean(rets):.6f}, std: {_np.std(rets):.6f}")

                    # Remove any NaN or infinite values
                    rets = rets[_np.isfinite(rets)]

                    if len(rets) > 1:
                        vol = float(_np.std(rets, ddof=1))
                        mean_return = float(_np.mean(rets))
                        total_ret = float(nav_values[-1] - 1.0)

                        # Calculate performance metrics
                        self.daily_return = mean_return * 100.0  # Convert to percentage

                        # Sharpe ratio (assuming crypto trades 365 days/year)
                        if vol > 0 and not _np.isnan(vol):
                            risk_free_daily = self.risk_free_rate / 365.0
                            excess_return = mean_return - risk_free_daily
                            self.sharpe_ratio = float(excess_return / vol * _np.sqrt(365.0))  # Annualized

                            logger.debug(f"Sharpe calculation: mean_return={mean_return:.6f}, rf_daily={risk_free_daily:.6f}, vol={vol:.6f}, sharpe={self.sharpe_ratio:.3f}")
                        else:
                            self.sharpe_ratio = 0.0
                            logger.warning(f"Invalid volatility for Sharpe calculation: {vol}")

                        # Maximum drawdown
                        cumulative = _np.cumprod(1.0 + rets)
                        running_max = _np.maximum.accumulate(cumulative)
                        drawdowns = (cumulative - running_max) / running_max
                        self.max_drawdown = float(_np.min(drawdowns)) * 100.0  # Convert to percentage

                        # Win rate (percentage of positive returns)
                        positive_days = _np.sum(rets > 0)
                        self.win_rate = float(positive_days / len(rets)) * 100.0

                        # Store as a single-point frontier overlay
                        self.current_portfolio_point = {"risk": vol, "return": total_ret}

                        logger.debug(f"Performance metrics calculated - Sharpe: {self.sharpe_ratio:.3f}, Return: {self.daily_return:.2f}%, Drawdown: {self.max_drawdown:.2f}%")
                    else:
                        logger.warning("No valid returns available for performance calculation")
                        self._reset_performance_metrics()
                else:
                    logger.warning(f"Insufficient data for performance calculation: {len(nav_values)} nav values")
                    self._reset_performance_metrics()

            except Exception as e:
                logger.error(f"Error calculating performance metrics: {e}")
                import traceback
                traceback.print_exc()
                self._reset_performance_metrics()

        except Exception as e:
            print(f"Error fetching portfolio data: {e}")
            # Keep previous or sample values if desired
        finally:
            self.loading = False

    def _reset_performance_metrics(self):
        """Reset performance metrics to default values"""
        self.daily_return = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        self.current_portfolio_point = {"risk": 0.0, "return": 0.0}

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

    # Asset selection
    available_assets: List[str] = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD", "AAVE-USD", "USDT-USD", "GC=F", "SPY"]
    selected_assets: List[str] = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"]

    # Constraints
    min_asset_weight: float = 0.01
    max_asset_weight: float = 0.40
    defi_max_weight: float = 0.30
    stablecoin_max_weight: float = 0.20

    # Individual constraints
    individual_asset_constraints: List[Dict] = []  # [{"asset": "BTC-USD", "min": 0.05, "max": 0.30}]
    individual_sector_constraints: List[Dict] = []  # [{"sector": "DeFi", "min": 0.05, "max": 0.30}]

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

    @rx.var
    def selected_assets_display(self) -> str:
        """Display selected assets for optimization"""
        if not self.selected_assets:
            return "No assets selected"
        return f"{len(self.selected_assets)} assets: {', '.join(self.selected_assets[:3])}{'...' if len(self.selected_assets) > 3 else ''}"

    @rx.var
    def positions_with_usd_display(self) -> List[Dict]:
        """Enhanced positions with USD display values"""
        enhanced_positions = []
        for pos in self.positions:
            enhanced_pos = dict(pos)
            # For display purposes, show both BTC and USD values for crypto options
            if pos.get("type", "").lower().startswith("option") and "BTC" in pos.get("symbol", ""):
                btc_value = pos.get("value", 0)
                enhanced_pos["value_display"] = f"{btc_value:.4f} BTC"  # Will add USD conversion dynamically
            else:
                enhanced_pos["value_display"] = f"${pos.get('value', 0):,.2f}"
            enhanced_positions.append(enhanced_pos)
        return enhanced_positions

    async def run_portfolio_optimization(self):
        """Run portfolio optimization using backend service (NO CALCULATIONS HERE)"""
        if not OPTIMIZATION_SERVICE_AVAILABLE:
            print("Optimization service not available - creating mock results")
            async for _ in self._create_mock_optimization_results():
                yield
            return

        self.optimization_loading = True
        yield

        try:
            # Get optimization service
            service = await get_optimization_service()

            # Use optimization_assets instead of positions
            assets = []
            if self.optimization_assets:
                for asset in self.optimization_assets:
                    symbol = asset
                    # Ensure symbol has proper format
                    if not symbol.endswith("-USD") and not symbol in ["GC=F", "SPY"]:
                        symbol += "-USD"
                    assets.append(symbol)
            else:
                # Fallback to positions if no optimization_assets selected
                for pos in self.positions:
                    if pos.get("type", "").lower().startswith("spot"):
                        symbol = pos["symbol"]
                        # Ensure symbol has proper format
                        if not symbol.endswith("-USD") and not symbol in ["GC=F", "SPY"]:
                            symbol += "-USD"
                        assets.append(symbol)

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
                sector_constraints=self._build_sector_constraints(),
                asset_constraints=self._build_asset_constraints()
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
            print("Optimization service not available - creating mock comparison results")
            async for _ in self._create_mock_comparison_results():
                yield
            return

        self.optimization_loading = True
        yield

        try:
            # Get optimization service
            service = await get_optimization_service()

            # Use optimization_assets instead of positions
            assets = []
            if self.optimization_assets:
                for asset in self.optimization_assets:
                    symbol = asset
                    # Ensure symbol has proper format
                    if not symbol.endswith("-USD") and not symbol in ["GC=F", "SPY"]:
                        symbol += "-USD"
                    assets.append(symbol)
            else:
                # Fallback to positions if no optimization_assets selected
                for pos in self.positions:
                    if pos.get("type", "").lower().startswith("spot"):
                        symbol = pos["symbol"]
                        # Ensure symbol has proper format
                        if not symbol.endswith("-USD") and not symbol in ["GC=F", "SPY"]:
                            symbol += "-USD"
                        assets.append(symbol)

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

    def set_constraint(self, constraint: str, value):
        """Set optimization constraints"""
        def _to_float(v, default: float) -> float:
            try:
                s = str(v).strip()
                if s == "" or s.lower() == "none":
                    return default
                return float(s)
            except Exception:
                return default

        if constraint == "min_asset_weight":
            self.min_asset_weight = _to_float(value, 0.01)
        elif constraint == "max_asset_weight":
            self.max_asset_weight = _to_float(value, 0.40)
        elif constraint == "defi_max_weight":
            self.defi_max_weight = _to_float(value, 0.30)
        elif constraint == "stablecoin_max_weight":
            self.stablecoin_max_weight = _to_float(value, 0.20)

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

    @rx.var
    def filtered_assets(self) -> List[str]:
        """Get filtered asset symbols based on position type and sector filter"""
        try:
            from src.core.config.crypto_sectors import CRYPTO_SECTORS

            all_assets = []

            # Get all assets from all sectors
            for sector_name, sector_data in CRYPTO_SECTORS.items():
                for ticker in sector_data["tickers"]:
                    all_assets.append({"symbol": ticker, "sector": sector_name})

            # Filter by sector if not "All"
            if self.selected_sector_filter and self.selected_sector_filter != "All":
                all_assets = [asset for asset in all_assets if asset["sector"] == self.selected_sector_filter]

            # Filter by position type if needed (for future options/futures)
            # For optimization, we only support Spot assets, so we always show all spot assets
            # Skip position type filtering since optimization is hardcoded to Spot only

            # Return just the symbol strings
            symbols = [asset["symbol"] for asset in all_assets]
            return sorted(list(set(symbols)))  # Remove duplicates and sort

        except Exception as e:
            logger.error(f"Error getting filtered assets: {e}")
            return ["BTC", "ETH", "SOL"]

    def close_position(self, symbol: str):
        self.positions = [p for p in self.positions if p["symbol"] != symbol]
        return PortfolioState.fetch_portfolio_data()

    def _build_sector_constraints(self) -> Dict:
        """Build sector constraints from individual constraints"""
        constraints = {}

        # Add basic sector constraints
        constraints['DeFi'] = {'max': self.defi_max_weight}
        constraints['Stablecoin'] = {'max': self.stablecoin_max_weight}

        # Add individual sector constraints
        for constraint in self.individual_sector_constraints:
            sector = constraint['sector']
            if sector not in constraints:
                constraints[sector] = {}

            if 'min' in constraint:
                constraints[sector]['min'] = constraint['min']
            if 'max' in constraint:
                constraints[sector]['max'] = constraint['max']

        return constraints

    def _build_asset_constraints(self) -> Dict:
        """Build asset constraints from individual constraints"""
        constraints = {}

        for constraint in self.individual_asset_constraints:
            asset = constraint['asset']
            constraints[asset] = {}

            if 'min' in constraint:
                constraints[asset]['min'] = constraint['min']
            if 'max' in constraint:
                constraints[asset]['max'] = constraint['max']

        return constraints

    def set_custom_asset_input(self, value: str):
        """Set the custom asset input value"""
        self.custom_asset_input = value

    def add_custom_asset_to_optimization(self):
        """Add custom asset to optimization assets list"""
        asset = self.custom_asset_input.strip().upper()
        if asset and asset not in self.optimization_assets:
            # Ensure proper format for crypto assets
            if not asset.endswith("-USD") and not asset in ["GC=F", "SPY", "QQQ", "TLT"]:
                asset += "-USD"

            self.optimization_assets.append(asset)
            # Clear input after adding
            self.custom_asset_input = ""

    def add_asset_to_optimization(self, asset: str):
        """Add asset from dropdown to optimization assets list"""
        if asset and asset not in self.optimization_assets:
            self.optimization_assets.append(asset)

    def remove_optimization_asset(self, asset: str):
        """Remove asset from optimization assets list"""
        if asset in self.optimization_assets:
            self.optimization_assets.remove(asset)

    def clear_optimization_assets(self):
        """Clear all optimization assets"""
        self.optimization_assets = []

    def add_asset_constraint(self):
        """Add asset constraint to active constraints"""
        if self.constraint_asset and self.constraint_asset in self.optimization_assets:
            constraint = {
                "type": "asset",
                "asset": self.constraint_asset,
                "min_weight": self.constraint_min_weight,
                "max_weight": self.constraint_max_weight,
                "display": f"{self.constraint_asset}: {self.constraint_min_weight:.2f} - {self.constraint_max_weight:.2f}"
            }
            self.active_constraints.append(constraint)
            # Reset form
            self.constraint_asset = ""
            self.constraint_min_weight = 0.0
            self.constraint_max_weight = 0.2

    def add_sector_constraint(self):
        """Add sector constraint to active constraints"""
        if self.constraint_sector:
            constraint = {
                "type": "sector",
                "sector": self.constraint_sector,
                "min_weight": self.constraint_sector_min,
                "max_weight": self.constraint_sector_max,
                "display": f"{self.constraint_sector}: {self.constraint_sector_min:.2f} - {self.constraint_sector_max:.2f}"
            }
            self.active_constraints.append(constraint)
            # Reset form
            self.constraint_sector = ""
            self.constraint_sector_min = 0.0
            self.constraint_sector_max = 0.25

    def remove_constraint(self, index: int):
        """Remove constraint at given index"""
        if 0 <= index < len(self.active_constraints):
            self.active_constraints.pop(index)

    def clear_all_constraints(self):
        """Clear all active constraints"""
        self.active_constraints = []

    async def _create_mock_optimization_results(self):
        """Create mock optimization results when service is unavailable"""
        self.optimization_loading = True
        yield

        try:
            import random
            from datetime import datetime

            # Generate equal weights for selected assets
            assets = self.optimization_assets if self.optimization_assets else ["BTC-USD", "ETH-USD", "AVAX-USD"]
            num_assets = len(assets)
            equal_weight = 1.0 / num_assets if num_assets > 0 else 0

            # Create suggested allocation
            suggested_allocation = {}
            for asset in assets:
                # Add some random variation around equal weight
                weight = equal_weight + random.uniform(-0.05, 0.05)
                weight = max(0.05, min(0.95, weight))  # Keep within reasonable bounds
                suggested_allocation[asset] = weight

            # Normalize weights to sum to 1
            total_weight = sum(suggested_allocation.values())
            if total_weight > 0:
                for asset in suggested_allocation:
                    suggested_allocation[asset] /= total_weight

            # Create mock optimization result
            expected_return = random.uniform(0.15, 0.35)
            volatility = random.uniform(0.25, 0.45)
            sharpe_ratio = random.uniform(0.8, 1.5)

            self.current_optimization_result = {
                'method_used': self.optimization_method,
                'objective_used': 'max_sharpe',
                'expected_return': f"{expected_return:.2%}",
                'volatility': f"{volatility:.2%}",
                'sharpe_ratio': f"{sharpe_ratio:.2f}",
                'optimization_time': 0.5,
                'timestamp': datetime.now().isoformat(),
                'sector_allocation': {'Infrastructure': 0.6, 'DeFi': 0.4}
            }

            # Convert to list format for data table
            self.suggested_allocation = [
                {"asset": asset, "weight": f"{weight:.2%}"}
                for asset, weight in suggested_allocation.items()
            ]
            self.optimization_results.append(self.current_optimization_result)

            print(f"Mock optimization completed for assets: {assets}")

        except Exception as e:
            print(f"Error creating mock results: {e}")
        finally:
            self.optimization_loading = False
            yield

    async def _create_mock_comparison_results(self):
        """Create mock comparison results when service is unavailable"""
        self.optimization_loading = True
        yield

        try:
            import random
            from datetime import datetime

            # Get assets for comparison
            assets = self.optimization_assets if self.optimization_assets else ["BTC-USD", "ETH-USD", "AVAX-USD"]
            methods = ['HRP', 'HERC', 'Sharpe', 'MinRisk']

            # Create mock comparison data
            comparison_results = []
            for method in methods:
                # Generate realistic performance metrics for each method
                expected_return = random.uniform(0.12, 0.40)
                volatility = random.uniform(0.20, 0.50)
                sharpe_ratio = expected_return / volatility if volatility > 0 else 0

                comparison_results.append({
                    'method': method,
                    'expected_return': f"{expected_return:.2%}",
                    'volatility': f"{volatility:.2%}",
                    'sharpe_ratio': f"{sharpe_ratio:.2f}",
                    'max_drawdown': f"{random.uniform(0.15, 0.35):.2%}",
                    'sortino_ratio': f"{random.uniform(0.6, 1.8):.3f}"
                })

            # Store comparison results
            self.comparison_results = comparison_results
            print(f"Mock comparison completed for {len(methods)} methods with assets: {assets}")

        except Exception as e:
            print(f"Error creating mock comparison results: {e}")
        finally:
            self.optimization_loading = False
            yield

    def add_portfolio_assets_to_optimization(self):
        """Add all current portfolio assets to optimization"""
        for pos in self.positions:
            if pos.get("type", "").lower().startswith("spot"):
                symbol = pos["symbol"]
                # Ensure symbol has proper format
                if not symbol.endswith("-USD") and not symbol in ["GC=F", "SPY"]:
                    symbol += "-USD"
                if symbol not in self.optimization_assets:
                    self.optimization_assets.append(symbol)

    async def get_btc_usd_price(self) -> float:
        """Get current BTC/USD price for conversions"""
        try:
            from src.data.providers.yfinance_provider import YFinanceProvider
            yf = YFinanceProvider()
            price_data = yf.get_current_price('BTC-USD')
            if price_data:
                return float(price_data.get('price_usd', 0))
            return 0.0
        except Exception as e:
            logger.error(f"Error getting BTC/USD price: {e}")
            return 0.0

    def convert_btc_to_usd(self, btc_amount: float, btc_usd_rate: float) -> float:
        """Convert BTC amount to USD"""
        return btc_amount * btc_usd_rate if btc_usd_rate > 0 else 0.0
