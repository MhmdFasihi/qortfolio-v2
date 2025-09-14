# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Volatility Surface Builder for Cryptocurrency Options
Constructs and stores volatility surfaces with database persistence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from scipy.interpolate import griddata, RBFInterpolator
from scipy.optimize import minimize_scalar
import logging
import asyncio

# Import your models
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.options.black_scholes import BlackScholesModel, OptionParameters, OptionType
from src.core.utils.time_utils import TimeUtils

logger = logging.getLogger(__name__)


@dataclass
class VolatilityPoint:
    """Single point on volatility surface."""
    strike: float
    time_to_maturity: float  # in years
    implied_volatility: float
    moneyness: float  # S/K ratio
    log_moneyness: float  # ln(S/K)
    market_price: float
    option_type: str
    volume: int = 0
    open_interest: int = 0
    bid: Optional[float] = None
    ask: Optional[float] = None


@dataclass
class VolatilitySurface:
    """Complete volatility surface data."""
    currency: str
    spot_price: float
    timestamp: datetime
    points: List[VolatilityPoint]
    surface_data: Dict[str, Any]  # Interpolated surface grid
    atm_term_structure: Dict[str, float]  # ATM vol by expiry
    skew_data: Dict[str, Dict[str, float]]  # Vol skew by expiry
    quality_metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'currency': self.currency,
            'spot_price': self.spot_price,
            'timestamp': self.timestamp,
            'points': [asdict(point) for point in self.points],
            'surface_data': self.surface_data,
            'atm_term_structure': self.atm_term_structure,
            'skew_data': self.skew_data,
            'quality_metrics': self.quality_metrics
        }


class VolatilitySurfaceBuilder:
    """
    Advanced volatility surface builder for cryptocurrency options.

    Features:
    - Implied volatility extraction from market data
    - Surface interpolation and smoothing
    - Quality validation and filtering
    - Database persistence
    - Real-time updates
    """

    def __init__(self, db_connection=None, min_time_to_maturity: float = 1/365):
        """
        Initialize volatility surface builder.

        Args:
            db_connection: Database connection for persistence
            min_time_to_maturity: Minimum time to maturity (default 1 day)
        """
        self.db = db_connection
        self.bs_model = BlackScholesModel(min_time_to_maturity)
        self.min_time_to_maturity = min_time_to_maturity
        self.logger = logging.getLogger(__name__)

    async def build_and_store_surface(self, currency: str, options_data: List[Dict]) -> VolatilitySurface:
        """
        Build complete volatility surface from options data and store in database.

        Args:
            currency: Currency symbol (BTC, ETH)
            options_data: List of options market data

        Returns:
            Complete volatility surface
        """
        try:
            # Extract spot price
            spot_price = self._extract_spot_price(options_data)

            # Calculate implied volatilities
            iv_points = await self._calculate_implied_volatilities(currency, options_data, spot_price)

            # Filter and validate data
            clean_points = self._filter_and_validate(iv_points)

            if len(clean_points) < 5:
                raise ValueError(f"Insufficient clean data points ({len(clean_points)}) for surface construction")

            # Build surface
            surface = self._construct_surface(currency, spot_price, clean_points)

            # Store in database if available
            if self.db:
                await self._store_surface(surface)

            return surface

        except Exception as e:
            self.logger.error(f"Failed to build volatility surface for {currency}: {e}")
            raise

    def _extract_spot_price(self, options_data: List[Dict]) -> float:
        """Extract current spot price from options data.

        Supports multiple input shapes:
        - Records that include 'underlying_price'
        - Records with a top-level 'spot_price' or 'spot' field
        """
        if not options_data:
            raise ValueError("No options data provided")

        # Try common fields present in Deribit/DB records
        spot_prices = []
        for opt in options_data:
            v = (
                opt.get('underlying_price')
                or opt.get('spot_price')
                or opt.get('spot')
            )
            try:
                if v is not None:
                    fv = float(v)
                    if fv > 0:
                        spot_prices.append(fv)
            except Exception:
                continue

        if spot_prices:
            return float(np.mean(spot_prices))

        # As a last resort, attempt to infer from moneyness-like fields if present (rare)
        raise ValueError("No spot price found in options data")

    async def _calculate_implied_volatilities(self, currency: str, options_data: List[Dict],
                                           spot_price: float) -> List[VolatilityPoint]:
        """Calculate implied volatilities for all options.

        Accepts both DB-stored and live Deribit shapes. Tries sensible fallbacks for
        market price and expiry fields.
        """
        iv_points: List[VolatilityPoint] = []

        for option in options_data:
            try:
                # Extract option details
                strike_raw = option.get('strike', 0) or option.get('strike_price', 0)
                strike = float(strike_raw) if strike_raw is not None else 0.0

                # Determine market price: prefer mark_price, fallback to mid(bid,ask), then last_price
                mp = option.get('mark_price')
                if mp is None:
                    bid = option.get('bid_price', option.get('bid'))
                    ask = option.get('ask_price', option.get('ask'))
                    try:
                        b = float(bid) if bid is not None else None
                        a = float(ask) if ask is not None else None
                        if b is not None and a is not None and b > 0 and a > 0:
                            mp = (a + b) / 2
                        elif b is not None and b > 0:
                            mp = b
                        elif a is not None and a > 0:
                            mp = a
                    except Exception:
                        mp = None
                if mp is None:
                    mp = option.get('last_price')

                market_price = float(mp) if mp is not None else 0.0

                # Option type normalization
                ot = option.get('option_type') or option.get('type')
                if isinstance(ot, str):
                    option_type = ot.lower()
                else:
                    option_type = 'call'

                # Expiry handling: support 'expiration_timestamp' (ms) or 'expiry' (datetime/str)
                expiry_time: Optional[datetime] = None
                expiry_ts = option.get('expiration_timestamp') or option.get('expiry_timestamp')
                if isinstance(expiry_ts, (int, float)) and expiry_ts > 0:
                    # Deribit epoch in ms
                    expiry_time = datetime.fromtimestamp(float(expiry_ts) / 1000.0)
                else:
                    exp = option.get('expiry')
                    if isinstance(exp, datetime):
                        expiry_time = exp
                    elif isinstance(exp, str) and exp:
                        # Try a couple of common formats
                        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d%b%y"):
                            try:
                                expiry_time = datetime.strptime(exp[:19], fmt)
                                break
                            except Exception:
                                continue

                # Skip if missing critical data
                if not (strike > 0 and market_price > 0 and isinstance(expiry_time, datetime)):
                    continue

                # Calculate time to maturity
                current_time = datetime.utcnow()
                time_to_maturity = TimeUtils.calculate_time_to_maturity(current_time, expiry_time)

                # Skip if too close to expiry
                if time_to_maturity < self.min_time_to_maturity:
                    continue

                # Calculate moneyness
                moneyness = spot_price / strike
                log_moneyness = np.log(max(moneyness, 1e-9))

                # Calculate implied volatility
                iv = await self._calculate_single_iv(
                    spot_price, strike, time_to_maturity, market_price, option_type
                )

                if iv and 0.01 <= iv <= 5.0:  # Reasonable IV range
                    iv_point = VolatilityPoint(
                        strike=strike,
                        time_to_maturity=time_to_maturity,
                        implied_volatility=iv,
                        moneyness=moneyness,
                        log_moneyness=log_moneyness,
                        market_price=market_price,
                        option_type=option_type,
                        volume=int(option.get('volume', 0) or 0),
                        open_interest=int(option.get('open_interest', 0) or 0),
                        bid=(option.get('bid_price') if option.get('bid_price') is not None else option.get('bid')),
                        ask=(option.get('ask_price') if option.get('ask_price') is not None else option.get('ask')),
                    )
                    iv_points.append(iv_point)

            except Exception as e:
                self.logger.debug(f"Failed to process option {option.get('symbol', 'unknown')}: {e}")
                continue

        self.logger.info(f"Calculated IV for {len(iv_points)} options out of {len(options_data)}")
        return iv_points

    async def _calculate_single_iv(self, spot: float, strike: float, time_to_maturity: float,
                                 market_price: float, option_type: str) -> Optional[float]:
        """Calculate implied volatility for a single option."""
        try:
            params = OptionParameters(
                spot_price=spot,
                strike_price=strike,
                time_to_maturity=time_to_maturity,
                volatility=0.5,  # Initial guess
                risk_free_rate=0.05,
                option_type=OptionType.CALL if option_type == 'call' else OptionType.PUT,
                is_coin_based=True  # Crypto options are coin-based
            )

            # Use Black-Scholes wrapper for IV calculation
            from src.models.options.black_scholes import BlackScholes
            bs_wrapper = BlackScholes()

            iv = bs_wrapper.calculate_implied_volatility(market_price, params)
            return iv

        except Exception as e:
            self.logger.debug(f"IV calculation failed for strike {strike}: {e}")
            return None

    def _filter_and_validate(self, iv_points: List[VolatilityPoint]) -> List[VolatilityPoint]:
        """Filter and validate volatility points."""
        clean_points = []

        for point in iv_points:
            # Quality filters
            if (0.01 <= point.implied_volatility <= 5.0 and  # Reasonable IV range
                point.time_to_maturity >= self.min_time_to_maturity and
                0.1 <= point.moneyness <= 10.0):  # Reasonable moneyness range

                # Additional quality checks
                if point.volume > 0 or point.open_interest > 0:  # Some market activity
                    clean_points.append(point)

        # Remove outliers using IQR
        if len(clean_points) > 10:
            ivs = [p.implied_volatility for p in clean_points]
            q1, q3 = np.percentile(ivs, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            clean_points = [p for p in clean_points
                          if lower_bound <= p.implied_volatility <= upper_bound]

        return clean_points

    def _construct_surface(self, currency: str, spot_price: float,
                         iv_points: List[VolatilityPoint]) -> VolatilitySurface:
        """Construct volatility surface from clean IV points."""
        # Create surface grid
        log_moneyness_range = np.linspace(-0.5, 0.5, 20)  # -50% to +50% moneyness
        time_range = np.linspace(self.min_time_to_maturity, max(p.time_to_maturity for p in iv_points), 15)

        # Prepare data for interpolation
        points_array = np.array([[p.log_moneyness, p.time_to_maturity] for p in iv_points])
        values_array = np.array([p.implied_volatility for p in iv_points])

        # Create mesh grid
        log_moneyness_grid, time_grid = np.meshgrid(log_moneyness_range, time_range)

        # Interpolate using RBF for smooth surface
        try:
            rbf_interpolator = RBFInterpolator(points_array, values_array, kernel='thin_plate_spline')
            grid_points = np.column_stack([log_moneyness_grid.ravel(), time_grid.ravel()])
            iv_grid = rbf_interpolator(grid_points).reshape(log_moneyness_grid.shape)
        except:
            # Fallback to linear interpolation
            iv_grid = griddata(points_array, values_array,
                             (log_moneyness_grid, time_grid), method='linear')
            # Fill NaN values
            iv_grid = np.nan_to_num(iv_grid, nan=0.5)

        # Calculate ATM term structure
        atm_term_structure = self._calculate_atm_term_structure(iv_points)

        # Calculate skew by expiry
        skew_data = self._calculate_skew_data(iv_points)

        # Quality metrics
        quality_metrics = self._calculate_quality_metrics(iv_points, iv_grid)

        # Package surface data
        surface_data = {
            'log_moneyness_range': log_moneyness_range.tolist(),
            'time_range': time_range.tolist(),
            'iv_grid': iv_grid.tolist(),
            'moneyness_range': np.exp(log_moneyness_range).tolist(),
            'strikes_range': (spot_price * np.exp(log_moneyness_range)).tolist()
        }

        return VolatilitySurface(
            currency=currency,
            spot_price=spot_price,
            timestamp=datetime.now(),
            points=iv_points,
            surface_data=surface_data,
            atm_term_structure=atm_term_structure,
            skew_data=skew_data,
            quality_metrics=quality_metrics
        )

    def _calculate_atm_term_structure(self, iv_points: List[VolatilityPoint]) -> Dict[str, float]:
        """Calculate ATM volatility term structure."""
        # Group by expiry and find ATM vol
        expiry_groups = {}
        for point in iv_points:
            expiry_key = f"{int(point.time_to_maturity * 365)}d"
            if expiry_key not in expiry_groups:
                expiry_groups[expiry_key] = []
            expiry_groups[expiry_key].append(point)

        atm_term_structure = {}
        for expiry, points in expiry_groups.items():
            # Find point closest to ATM (moneyness = 1)
            atm_point = min(points, key=lambda p: abs(p.moneyness - 1.0))
            atm_term_structure[expiry] = atm_point.implied_volatility

        return atm_term_structure

    def _calculate_skew_data(self, iv_points: List[VolatilityPoint]) -> Dict[str, Dict[str, float]]:
        """Calculate volatility skew by expiry."""
        skew_data = {}

        # Group by expiry
        expiry_groups = {}
        for point in iv_points:
            expiry_key = f"{int(point.time_to_maturity * 365)}d"
            if expiry_key not in expiry_groups:
                expiry_groups[expiry_key] = []
            expiry_groups[expiry_key].append(point)

        for expiry, points in expiry_groups.items():
            if len(points) >= 3:
                # Calculate skew metrics
                sorted_points = sorted(points, key=lambda p: p.moneyness)

                # Find 90% and 110% moneyness points (or closest)
                otm_put = min(points, key=lambda p: abs(p.moneyness - 0.9))
                atm = min(points, key=lambda p: abs(p.moneyness - 1.0))
                otm_call = min(points, key=lambda p: abs(p.moneyness - 1.1))

                put_call_skew = otm_put.implied_volatility - otm_call.implied_volatility
                atm_vol = atm.implied_volatility

                skew_data[expiry] = {
                    'put_call_skew': put_call_skew,
                    'atm_vol': atm_vol,
                    'min_vol': min(p.implied_volatility for p in points),
                    'max_vol': max(p.implied_volatility for p in points)
                }

        return skew_data

    def _calculate_quality_metrics(self, iv_points: List[VolatilityPoint],
                                 iv_grid: np.ndarray) -> Dict[str, float]:
        """Calculate surface quality metrics."""
        ivs = [p.implied_volatility for p in iv_points]

        return {
            'data_points_count': len(iv_points),
            'iv_mean': np.mean(ivs),
            'iv_std': np.std(ivs),
            'iv_min': np.min(ivs),
            'iv_max': np.max(ivs),
            'moneyness_coverage': max(p.moneyness for p in iv_points) - min(p.moneyness for p in iv_points),
            'time_coverage': max(p.time_to_maturity for p in iv_points) - min(p.time_to_maturity for p in iv_points),
            'surface_smoothness': np.std(iv_grid) if iv_grid.size > 0 else 0,
            'total_volume': sum(p.volume for p in iv_points),
            'total_open_interest': sum(p.open_interest for p in iv_points)
        }

    async def _store_surface(self, surface: VolatilitySurface) -> None:
        """Store volatility surface in database."""
        try:
            if hasattr(self.db, 'volatility_surfaces'):
                surface_doc = surface.to_dict()
                surface_doc['_id'] = f"{surface.currency}_{surface.timestamp.strftime('%Y%m%d_%H%M%S')}"

                # Store current surface
                await self.db.volatility_surfaces.replace_one(
                    {'currency': surface.currency},
                    surface_doc,
                    upsert=True
                )

                # Store historical surface
                await self.db.volatility_surfaces_history.insert_one(surface_doc)

                self.logger.info(f"Stored volatility surface for {surface.currency}")

        except Exception as e:
            self.logger.error(f"Failed to store volatility surface: {e}")

    def interpolate_volatility(self, surface: VolatilitySurface, strike: float,
                             time_to_maturity: float) -> float:
        """Interpolate volatility at specific strike and time."""
        try:
            spot_price = surface.spot_price
            log_moneyness = np.log(spot_price / strike)

            # Get surface data
            log_moneyness_range = np.array(surface.surface_data['log_moneyness_range'])
            time_range = np.array(surface.surface_data['time_range'])
            iv_grid = np.array(surface.surface_data['iv_grid'])

            # Create interpolator
            log_moneyness_grid, time_grid = np.meshgrid(log_moneyness_range, time_range)
            points = np.column_stack([log_moneyness_grid.ravel(), time_grid.ravel()])
            values = iv_grid.ravel()

            # Interpolate
            interpolated_iv = griddata(points, values,
                                     [(log_moneyness, time_to_maturity)],
                                     method='linear')[0]

            # Bounds checking
            if np.isnan(interpolated_iv):
                # Use nearest neighbor as fallback
                interpolated_iv = griddata(points, values,
                                         [(log_moneyness, time_to_maturity)],
                                         method='nearest')[0]

            return max(0.01, min(5.0, interpolated_iv))  # Reasonable bounds

        except Exception as e:
            self.logger.warning(f"Volatility interpolation failed: {e}")
            return 0.5  # Default fallback


# Convenience functions
async def build_crypto_vol_surface(currency: str, options_data: List[Dict],
                                 db_connection=None) -> VolatilitySurface:
    """
    Convenience function to build volatility surface for crypto currency.

    Args:
        currency: Currency symbol (BTC, ETH)
        options_data: Options market data
        db_connection: Database connection

    Returns:
        Complete volatility surface
    """
    builder = VolatilitySurfaceBuilder(db_connection)
    return await builder.build_and_store_surface(currency, options_data)


if __name__ == "__main__":
    # Example usage
    print("Volatility Surface Builder for Crypto Options")
    print("=" * 50)

    # This would typically be called with real market data
    print("✅ Volatility Surface Builder ready for Qortfolio V2!")
    print("   - Implied volatility extraction")
    print("   - Surface interpolation and smoothing")
    print("   - Database persistence")
    print("   - Quality metrics and validation")
