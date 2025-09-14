"""Enhanced State Management with MongoDB + Deribit fallback"""

import reflex as rx
from typing import Dict, List
from datetime import datetime


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

    @rx.var
    def avg_iv_pct(self) -> str:
        """Formatted average IV as percentage"""
        return f"{self.avg_iv:.2%}"

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
