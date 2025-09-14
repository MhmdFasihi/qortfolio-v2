# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Options Chain Analyzer
Advanced analytics for crypto options chains with coin-based pricing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging

# Import your models
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.options.black_scholes import BlackScholesModel, OptionParameters, OptionType
from src.models.options.greeks_calculator import GreeksCalculator, GreeksProfile
from src.core.utils.time_utils import TimeUtils

logger = logging.getLogger(__name__)


class OptionsFlowDirection(Enum):
    """Options flow direction."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class OptionsChainMetrics:
    """Comprehensive options chain metrics."""
    total_call_volume: int
    total_put_volume: int
    call_put_ratio: float
    total_call_oi: int
    total_put_oi: int
    call_put_oi_ratio: float
    max_pain_strike: float
    gamma_exposure: float
    vanna_exposure: float
    charm_exposure: float
    avg_iv_calls: float
    avg_iv_puts: float
    iv_rank: float  # Current IV vs historical range
    term_structure_slope: float
    skew_25d: float  # 25 delta skew
    flow_direction: OptionsFlowDirection
    unusual_activity: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['flow_direction'] = result['flow_direction'].value
        return result


@dataclass
class StrikeAnalysis:
    """Analysis for specific strike."""
    strike: float
    call_data: Optional[Dict] = None
    put_data: Optional[Dict] = None
    total_gamma: float = 0.0
    total_vanna: float = 0.0
    net_delta: float = 0.0
    support_resistance_score: float = 0.0
    pin_risk: float = 0.0


class OptionsChainAnalyzer:
    """
    Advanced options chain analyzer for cryptocurrency options.

    Features:
    - Comprehensive chain metrics calculation
    - Greeks exposure analysis
    - Flow analysis and unusual activity detection
    - Support/resistance level identification
    - Max pain calculation
    - Real-time analytics
    """

    def __init__(self, db_connection=None):
        """
        Initialize options chain analyzer.

        Args:
            db_connection: Database connection for historical data
        """
        self.db = db_connection
        self.bs_model = BlackScholesModel()
        self.greeks_calc = GreeksCalculator(self.bs_model)
        self.logger = logging.getLogger(__name__)

    def analyze_options_chain(self, options_data: List[Dict],
                            spot_price: float) -> OptionsChainMetrics:
        """
        Perform comprehensive options chain analysis.

        Args:
            options_data: List of options market data
            spot_price: Current spot price

        Returns:
            Complete options chain metrics
        """
        try:
            # Separate calls and puts
            calls = [opt for opt in options_data if opt.get('option_type', '').lower() == 'call']
            puts = [opt for opt in options_data if opt.get('option_type', '').lower() == 'put']

            # Basic volume and OI metrics
            total_call_volume = sum(opt.get('volume', 0) for opt in calls)
            total_put_volume = sum(opt.get('volume', 0) for opt in puts)
            total_call_oi = sum(opt.get('open_interest', 0) for opt in calls)
            total_put_oi = sum(opt.get('open_interest', 0) for opt in puts)

            # Calculate ratios
            call_put_ratio = total_call_volume / max(total_put_volume, 1)
            call_put_oi_ratio = total_call_oi / max(total_put_oi, 1)

            # Calculate max pain
            max_pain_strike = self._calculate_max_pain(options_data, spot_price)

            # Calculate Greeks exposures
            gamma_exposure = self._calculate_gamma_exposure(options_data, spot_price)
            vanna_exposure = self._calculate_vanna_exposure(options_data, spot_price)
            charm_exposure = self._calculate_charm_exposure(options_data, spot_price)

            # IV analysis
            call_ivs = [opt.get('mark_iv', 0) for opt in calls if opt.get('mark_iv', 0) > 0]
            put_ivs = [opt.get('mark_iv', 0) for opt in puts if opt.get('mark_iv', 0) > 0]

            avg_iv_calls = np.mean(call_ivs) if call_ivs else 0
            avg_iv_puts = np.mean(put_ivs) if put_ivs else 0

            # Calculate IV rank (simplified - would need historical data)
            iv_rank = 0.5  # Placeholder - requires historical IV data

            # Term structure analysis
            term_structure_slope = self._calculate_term_structure_slope(options_data)

            # Skew analysis
            skew_25d = self._calculate_skew_25d(options_data, spot_price)

            # Flow direction analysis
            flow_direction = self._analyze_flow_direction(options_data, spot_price)

            # Unusual activity detection
            unusual_activity = self._detect_unusual_activity(options_data, spot_price)

            return OptionsChainMetrics(
                total_call_volume=total_call_volume,
                total_put_volume=total_put_volume,
                call_put_ratio=call_put_ratio,
                total_call_oi=total_call_oi,
                total_put_oi=total_put_oi,
                call_put_oi_ratio=call_put_oi_ratio,
                max_pain_strike=max_pain_strike,
                gamma_exposure=gamma_exposure,
                vanna_exposure=vanna_exposure,
                charm_exposure=charm_exposure,
                avg_iv_calls=avg_iv_calls,
                avg_iv_puts=avg_iv_puts,
                iv_rank=iv_rank,
                term_structure_slope=term_structure_slope,
                skew_25d=skew_25d,
                flow_direction=flow_direction,
                unusual_activity=unusual_activity
            )

        except Exception as e:
            self.logger.error(f"Options chain analysis failed: {e}")
            raise

    def _calculate_max_pain(self, options_data: List[Dict], spot_price: float) -> float:
        """Calculate max pain strike."""
        try:
            # Group by strike
            strikes = {}
            for opt in options_data:
                strike = float(opt.get('strike', 0))
                if strike > 0:
                    if strike not in strikes:
                        strikes[strike] = {'calls': [], 'puts': []}

                    if opt.get('option_type', '').lower() == 'call':
                        strikes[strike]['calls'].append(opt)
                    else:
                        strikes[strike]['puts'].append(opt)

            # Calculate pain for each strike
            max_pain_strike = spot_price
            min_pain = float('inf')

            for test_strike in strikes.keys():
                total_pain = 0

                for strike, options in strikes.items():
                    # Call pain
                    for call_opt in options['calls']:
                        oi = call_opt.get('open_interest', 0)
                        if test_strike > strike:
                            total_pain += (test_strike - strike) * oi

                    # Put pain
                    for put_opt in options['puts']:
                        oi = put_opt.get('open_interest', 0)
                        if test_strike < strike:
                            total_pain += (strike - test_strike) * oi

                if total_pain < min_pain:
                    min_pain = total_pain
                    max_pain_strike = test_strike

            return max_pain_strike

        except Exception as e:
            self.logger.warning(f"Max pain calculation failed: {e}")
            return spot_price

    def _calculate_gamma_exposure(self, options_data: List[Dict], spot_price: float) -> float:
        """Calculate total gamma exposure."""
        total_gamma = 0

        for opt in options_data:
            try:
                strike = float(opt.get('strike', 0))
                mark_iv = opt.get('mark_iv', 0.5)
                time_to_maturity = self._get_time_to_maturity(opt)
                oi = opt.get('open_interest', 0)
                option_type = opt.get('option_type', 'call').lower()

                if all([strike > 0, mark_iv > 0, time_to_maturity > 0, oi > 0]):
                    # Calculate gamma
                    params = OptionParameters(
                        spot_price=spot_price,
                        strike_price=strike,
                        time_to_maturity=time_to_maturity,
                        volatility=mark_iv,
                        option_type=OptionType.CALL if option_type == 'call' else OptionType.PUT,
                        is_coin_based=True
                    )

                    pricing = self.bs_model.calculate_option_price(params)
                    total_gamma += pricing.gamma * oi

            except Exception:
                continue

        return total_gamma * spot_price * spot_price / 100  # Dollar gamma

    def _calculate_vanna_exposure(self, options_data: List[Dict], spot_price: float) -> float:
        """Calculate total vanna exposure (simplified)."""
        # Simplified vanna calculation
        total_vanna = 0

        for opt in options_data:
            try:
                strike = float(opt.get('strike', 0))
                mark_iv = opt.get('mark_iv', 0.5)
                time_to_maturity = self._get_time_to_maturity(opt)
                oi = opt.get('open_interest', 0)

                if all([strike > 0, mark_iv > 0, time_to_maturity > 0, oi > 0]):
                    # Simplified vanna approximation
                    vanna_approx = np.sqrt(time_to_maturity) * (1 - abs(np.log(spot_price / strike))) * mark_iv
                    total_vanna += vanna_approx * oi

            except Exception:
                continue

        return total_vanna

    def _calculate_charm_exposure(self, options_data: List[Dict], spot_price: float) -> float:
        """Calculate total charm exposure (simplified)."""
        # Simplified charm calculation
        total_charm = 0

        for opt in options_data:
            try:
                strike = float(opt.get('strike', 0))
                time_to_maturity = self._get_time_to_maturity(opt)
                oi = opt.get('open_interest', 0)

                if all([strike > 0, time_to_maturity > 0, oi > 0]):
                    # Simplified charm approximation
                    charm_approx = abs(spot_price - strike) / (time_to_maturity * spot_price)
                    total_charm += charm_approx * oi

            except Exception:
                continue

        return total_charm

    def _get_time_to_maturity(self, option: Dict) -> float:
        """Extract time to maturity from option data."""
        try:
            expiry_timestamp = option.get('expiration_timestamp', 0)
            if expiry_timestamp > 0:
                current_time = datetime.now()
                expiry_time = datetime.fromtimestamp(expiry_timestamp / 1000)
                return TimeUtils.calculate_time_to_maturity(current_time, expiry_time)
        except Exception:
            pass
        return 0.0

    def _calculate_term_structure_slope(self, options_data: List[Dict]) -> float:
        """Calculate term structure slope."""
        try:
            # Group ATM options by expiry
            atm_options = []
            for opt in options_data:
                spot_price = opt.get('underlying_price', 100)
                strike = opt.get('strike', 100)
                moneyness = spot_price / strike

                # Consider ATM if within 5% of spot
                if 0.95 <= moneyness <= 1.05:
                    time_to_maturity = self._get_time_to_maturity(opt)
                    mark_iv = opt.get('mark_iv', 0)
                    if time_to_maturity > 0 and mark_iv > 0:
                        atm_options.append((time_to_maturity, mark_iv))

            if len(atm_options) >= 2:
                # Sort by time to maturity
                atm_options.sort(key=lambda x: x[0])

                # Calculate slope between shortest and longest expiry
                short_term = atm_options[0]
                long_term = atm_options[-1]

                time_diff = long_term[0] - short_term[0]
                if time_diff > 0:
                    return (long_term[1] - short_term[1]) / time_diff

        except Exception as e:
            self.logger.debug(f"Term structure slope calculation failed: {e}")

        return 0.0

    def _calculate_skew_25d(self, options_data: List[Dict], spot_price: float) -> float:
        """Calculate 25 delta skew."""
        # This is a simplified implementation
        # In practice, you'd need to find 25 delta puts and calls
        try:
            calls = [opt for opt in options_data if opt.get('option_type', '').lower() == 'call']
            puts = [opt for opt in options_data if opt.get('option_type', '').lower() == 'put']

            if calls and puts:
                call_ivs = [opt.get('mark_iv', 0) for opt in calls if opt.get('mark_iv', 0) > 0]
                put_ivs = [opt.get('mark_iv', 0) for opt in puts if opt.get('mark_iv', 0) > 0]

                if call_ivs and put_ivs:
                    return np.mean(put_ivs) - np.mean(call_ivs)

        except Exception:
            pass

        return 0.0

    def _analyze_flow_direction(self, options_data: List[Dict], spot_price: float) -> OptionsFlowDirection:
        """Analyze options flow direction."""
        try:
            # Simple flow analysis based on call/put ratios and volumes
            total_call_volume = sum(opt.get('volume', 0) for opt in options_data
                                  if opt.get('option_type', '').lower() == 'call')
            total_put_volume = sum(opt.get('volume', 0) for opt in options_data
                                 if opt.get('option_type', '').lower() == 'put')

            if total_call_volume + total_put_volume == 0:
                return OptionsFlowDirection.NEUTRAL

            call_ratio = total_call_volume / (total_call_volume + total_put_volume)

            if call_ratio > 0.6:
                return OptionsFlowDirection.BULLISH
            elif call_ratio < 0.4:
                return OptionsFlowDirection.BEARISH
            else:
                return OptionsFlowDirection.NEUTRAL

        except Exception:
            return OptionsFlowDirection.NEUTRAL

    def _detect_unusual_activity(self, options_data: List[Dict], spot_price: float) -> List[Dict[str, Any]]:
        """Detect unusual options activity."""
        unusual_activity = []

        try:
            # Calculate volume and OI statistics
            volumes = [opt.get('volume', 0) for opt in options_data if opt.get('volume', 0) > 0]
            ois = [opt.get('open_interest', 0) for opt in options_data if opt.get('open_interest', 0) > 0]

            if not volumes or not ois:
                return unusual_activity

            volume_threshold = np.percentile(volumes, 90)  # Top 10% volume
            oi_threshold = np.percentile(ois, 90)  # Top 10% OI

            for opt in options_data:
                volume = opt.get('volume', 0)
                oi = opt.get('open_interest', 0)

                # Flag unusual activity
                if volume > volume_threshold or oi > oi_threshold:
                    unusual_activity.append({
                        'symbol': opt.get('symbol', ''),
                        'strike': opt.get('strike', 0),
                        'option_type': opt.get('option_type', ''),
                        'volume': volume,
                        'open_interest': oi,
                        'mark_iv': opt.get('mark_iv', 0),
                        'reason': 'high_volume' if volume > volume_threshold else 'high_oi'
                    })

        except Exception as e:
            self.logger.debug(f"Unusual activity detection failed: {e}")

        return unusual_activity[:10]  # Return top 10

    def analyze_strike_levels(self, options_data: List[Dict],
                            spot_price: float) -> List[StrikeAnalysis]:
        """Analyze individual strike levels for support/resistance."""
        strike_analysis = {}

        for opt in options_data:
            try:
                strike = float(opt.get('strike', 0))
                if strike <= 0:
                    continue

                if strike not in strike_analysis:
                    strike_analysis[strike] = StrikeAnalysis(strike=strike)

                # Add option data
                if opt.get('option_type', '').lower() == 'call':
                    strike_analysis[strike].call_data = opt
                else:
                    strike_analysis[strike].put_data = opt

                # Calculate Greeks contribution
                time_to_maturity = self._get_time_to_maturity(opt)
                mark_iv = opt.get('mark_iv', 0.5)
                oi = opt.get('open_interest', 0)

                if all([time_to_maturity > 0, mark_iv > 0, oi > 0]):
                    params = OptionParameters(
                        spot_price=spot_price,
                        strike_price=strike,
                        time_to_maturity=time_to_maturity,
                        volatility=mark_iv,
                        option_type=OptionType.CALL if opt.get('option_type', '').lower() == 'call' else OptionType.PUT,
                        is_coin_based=True
                    )

                    pricing = self.bs_model.calculate_option_price(params)

                    strike_analysis[strike].total_gamma += pricing.gamma * oi
                    strike_analysis[strike].net_delta += pricing.delta * oi

                    # Support/resistance score based on gamma and OI
                    strike_analysis[strike].support_resistance_score += abs(pricing.gamma) * oi

            except Exception:
                continue

        # Convert to list and sort by strike
        result = list(strike_analysis.values())
        result.sort(key=lambda x: x.strike)

        return result

    async def get_historical_analysis(self, currency: str, days: int = 30) -> Dict[str, Any]:
        """Get historical options chain analysis."""
        if not self.db:
            return {}

        try:
            # This would query historical chain data from database
            # Simplified implementation
            return {
                'avg_call_put_ratio': 1.2,
                'iv_percentile': 0.65,
                'historical_skew': 0.05,
                'flow_direction_frequency': {
                    'bullish': 0.4,
                    'bearish': 0.3,
                    'neutral': 0.3
                }
            }

        except Exception as e:
            self.logger.error(f"Historical analysis failed: {e}")
            return {}


# Convenience functions
def analyze_chain_metrics(options_data: List[Dict], spot_price: float) -> OptionsChainMetrics:
    """Quick options chain analysis."""
    analyzer = OptionsChainAnalyzer()
    return analyzer.analyze_options_chain(options_data, spot_price)


def find_key_strikes(options_data: List[Dict], spot_price: float,
                    top_n: int = 5) -> List[StrikeAnalysis]:
    """Find key support/resistance strikes."""
    analyzer = OptionsChainAnalyzer()
    strikes = analyzer.analyze_strike_levels(options_data, spot_price)

    # Sort by support/resistance score
    strikes.sort(key=lambda x: x.support_resistance_score, reverse=True)

    return strikes[:top_n]


if __name__ == "__main__":
    print("Options Chain Analyzer for Crypto Options")
    print("=" * 50)

    print("✅ Options Chain Analyzer ready for Qortfolio V2!")
    print("   - Comprehensive chain metrics")
    print("   - Greeks exposure analysis")
    print("   - Flow direction analysis")
    print("   - Unusual activity detection")
    print("   - Support/resistance identification")