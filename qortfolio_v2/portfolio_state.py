"""Portfolio Management State"""

import reflex as rx
from typing import Dict, List
from datetime import datetime, timedelta


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

    def add_position(self):
        pass

    def close_position(self, symbol: str):
        self.positions = [p for p in self.positions if p["symbol"] != symbol]
        return PortfolioState.fetch_portfolio_data()
