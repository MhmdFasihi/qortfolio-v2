# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Volatility Surface Analysis
Location: src/analytics/volatility_surface.py

Advanced volatility surface construction and analysis for cryptocurrency options.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy import interpolate
from scipy.optimize import minimize_scalar
import plotly.graph_objects as go
import plotly.express as px
from typing import List


from core.utils.time_utils import calculate_time_to_maturity
from models.options.black_scholes import BlackScholesModel, OptionParameters, OptionType

logger = logging.getLogger(__name__)

@dataclass
class VolatilitySurfacePoint:
    """Single point on volatility surface."""
    strike: float
    time_to_expiry: float
    implied_vol: float
    moneyness: float
    option_type: str
    volume: float = 0.0
    open_interest: float = 0.0
    
@dataclass
class VolatilitySurfaceData:
    """Complete volatility surface data."""
    surface_points: List[VolatilitySurfacePoint]
    spot_price: float
    reference_date: datetime
    atm_volatility: float
    total_volume: float
    skew_metrics: Dict[str, float]

class VolatilitySurfaceAnalyzer:
    """
    Advanced volatility surface analysis and construction.
    
    Features:
    - 3D volatility surface visualization
    - Skew analysis across strikes and tenors
    - ATM volatility tracking
    - Volume-weighted analysis
    - Surface smoothing and interpolation
    """
    
    def __init__(self):
        self.bs_model = BlackScholesModel()
        self.logger = logging.getLogger(__name__)
        
    def build_volatility_surface(
        self,
        options_data: pd.DataFrame,
        spot_price: float,
        reference_date: Optional[datetime] = None
    ) -> VolatilitySurfaceData:
        """
        Build volatility surface from options market data.
        
        Args:
            options_data: DataFrame with options data
            spot_price: Current spot price
            reference_date: Reference date (default: now)
            
        Returns:
            VolatilitySurfaceData object
        """
        try:
            if reference_date is None:
                reference_date = datetime.now()
            
            surface_points = []
            total_volume = 0.0
            
            for _, row in options_data.iterrows():
                try:
                    # Calculate implied volatility
                    if 'implied_volatility' in row and pd.notna(row['implied_volatility']):
                        iv = float(row['implied_volatility'])
                    else:
                        # Calculate IV from market price
                        iv = self._calculate_implied_volatility(
                            market_price=row.get('mark_price', row.get('price', 0)),
                            spot_price=spot_price,
                            strike=row['strike'],
                            time_to_expiry=row.get('time_to_maturity', row.get('tte', 0)),
                            option_type=row.get('option_type', 'call'),
                            risk_free_rate=0.05
                        )
                    
                    if iv is None or iv <= 0:
                        continue
                    
                    # Calculate moneyness
                    moneyness = row['strike'] / spot_price
                    
                    # Create surface point
                    point = VolatilitySurfacePoint(
                        strike=float(row['strike']),
                        time_to_expiry=float(row.get('time_to_maturity', row.get('tte', 0))),
                        implied_vol=iv,
                        moneyness=moneyness,
                        option_type=str(row.get('option_type', 'call')),
                        volume=float(row.get('volume', 0)),
                        open_interest=float(row.get('open_interest', 0))
                    )
                    
                    surface_points.append(point)
                    total_volume += point.volume
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process option data point: {e}")
                    continue
            
            if not surface_points:
                self.logger.error("No valid surface points generated")
                return self._get_empty_surface_data(spot_price, reference_date)
            
            # Calculate ATM volatility
            atm_vol = self._calculate_atm_volatility(surface_points, spot_price)
            
            # Calculate skew metrics
            skew_metrics = self._calculate_skew_metrics(surface_points, spot_price)
            
            return VolatilitySurfaceData(
                surface_points=surface_points,
                spot_price=spot_price,
                reference_date=reference_date,
                atm_volatility=atm_vol,
                total_volume=total_volume,
                skew_metrics=skew_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Failed to build volatility surface: {e}")
            return self._get_empty_surface_data(spot_price, reference_date)
    
    def analyze_volatility_skew(
        self,
        surface_data: VolatilitySurfaceData,
        target_tenor_days: float = 30
    ) -> Dict[str, Any]:
        """
        Analyze volatility skew for a specific tenor.
        
        Args:
            surface_data: Volatility surface data
            target_tenor_days: Target tenor in days
            
        Returns:
            Skew analysis results
        """
        try:
            target_tte = target_tenor_days / 365.25
            tolerance = 7 / 365.25  # 7-day tolerance
            
            # Filter points near target tenor
            relevant_points = [
                p for p in surface_data.surface_points
                if abs(p.time_to_expiry - target_tte) <= tolerance
            ]
            
            if len(relevant_points) < 3:
                return {"error": f"Insufficient data for {target_tenor_days}D analysis"}
            
            # Sort by moneyness
            relevant_points.sort(key=lambda p: p.moneyness)
            
            # Extract data
            moneyness_values = [p.moneyness for p in relevant_points]
            iv_values = [p.implied_vol for p in relevant_points]
            strikes = [p.strike for p in relevant_points]
            
            # Calculate skew metrics
            atm_index = self._find_closest_atm_index(moneyness_values)
            atm_iv = iv_values[atm_index] if atm_index is not None else np.mean(iv_values)
            
            # Risk reversal (25-delta put vs 25-delta call)
            rr_25 = self._calculate_risk_reversal(relevant_points, 0.25)
            
            # Butterfly spread (25-delta strangle vs ATM)
            bf_25 = self._calculate_butterfly(relevant_points, 0.25)
            
            # Skew slope
            skew_slope = self._calculate_skew_slope(moneyness_values, iv_values)
            
            return {
                "tenor_days": target_tenor_days,
                "data_points": len(relevant_points),
                "atm_volatility": atm_iv,
                "risk_reversal_25d": rr_25,
                "butterfly_25d": bf_25,
                "skew_slope": skew_slope,
                "moneyness_range": [min(moneyness_values), max(moneyness_values)],
                "iv_range": [min(iv_values), max(iv_values)],
                "strikes": strikes,
                "moneyness": moneyness_values,
                "implied_vols": iv_values
            }
            
        except Exception as e:
            self.logger.error(f"Skew analysis failed: {e}")
            return {"error": str(e)}
    
    def create_3d_surface_plot(
        self,
        surface_data: VolatilitySurfaceData,
        title: str = "Implied Volatility Surface"
    ) -> go.Figure:
        """
        Create 3D volatility surface plot.
        
        Args:
            surface_data: Volatility surface data
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        try:
            # Extract data for plotting
            strikes = [p.strike for p in surface_data.surface_points]
            times = [p.time_to_expiry * 365.25 for p in surface_data.surface_points]  # Convert to days
            vols = [p.implied_vol * 100 for p in surface_data.surface_points]  # Convert to percentage
            
            # Create grid for interpolation
            strike_grid = np.linspace(min(strikes), max(strikes), 20)
            time_grid = np.linspace(max(1, min(times)), max(times), 15)
            
            # Interpolate to create smooth surface
            try:
                # Create interpolation function
                points = np.column_stack((strikes, times))
                vol_interp = interpolate.griddata(
                    points, vols, 
                    np.meshgrid(strike_grid, time_grid, indexing='ij'),
                    method='cubic',
                    fill_value=np.nan
                )
                
                # Create 3D surface
                fig = go.Figure(data=[
                    go.Surface(
                        x=strike_grid,
                        y=time_grid,
                        z=vol_interp,
                        colorscale='Viridis',
                        name='IV Surface'
                    )
                ])
                
                # Add scatter points for actual data
                fig.add_trace(go.Scatter3d(
                    x=strikes,
                    y=times,
                    z=vols,
                    mode='markers',
                    marker=dict(size=3, color='red'),
                    name='Market Data'
                ))
                
            except Exception as interp_error:
                self.logger.warning(f"Interpolation failed, using scatter plot: {interp_error}")
                
                # Fallback to scatter plot
                fig = go.Figure(data=[
                    go.Scatter3d(
                        x=strikes,
                        y=times,
                        z=vols,
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=vols,
                            colorscale='Viridis',
                            showscale=True
                        ),
                        name='Market Data'
                    )
                ])
            
            # Update layout
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='Strike Price',
                    yaxis_title='Days to Expiry',
                    zaxis_title='Implied Volatility (%)',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                height=600
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create 3D surface plot: {e}")
            # Return empty figure
            return go.Figure()
    
    def create_skew_plot(
        self,
        skew_analysis: Dict[str, Any],
        title: str = "Volatility Skew"
    ) -> go.Figure:
        """
        Create volatility skew plot.
        
        Args:
            skew_analysis: Skew analysis results
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        try:
            if "error" in skew_analysis:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Error: {skew_analysis['error']}",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False
                )
                return fig
            
            moneyness = skew_analysis['moneyness']
            iv_values = [iv * 100 for iv in skew_analysis['implied_vols']]  # Convert to percentage
            
            fig = go.Figure()
            
            # Add skew curve
            fig.add_trace(go.Scatter(
                x=moneyness,
                y=iv_values,
                mode='lines+markers',
                name=f"{skew_analysis['tenor_days']}D Skew",
                line=dict(width=2),
                marker=dict(size=6)
            ))
            
            # Add ATM line
            fig.add_hline(
                y=skew_analysis['atm_volatility'] * 100,
                line_dash="dash",
                annotation_text="ATM Vol",
                annotation_position="top right"
            )
            
            # Add vertical line at moneyness = 1.0
            fig.add_vline(
                x=1.0,
                line_dash="dot",
                annotation_text="ATM",
                annotation_position="top"
            )
            
            fig.update_layout(
                title=title,
                xaxis_title='Moneyness (Strike/Spot)',
                yaxis_title='Implied Volatility (%)',
                height=400,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create skew plot: {e}")
            return go.Figure()
    
    def _calculate_implied_volatility(
        self,
        market_price: float,
        spot_price: float,
        strike: float,
        time_to_expiry: float,
        option_type: str,
        risk_free_rate: float = 0.05
    ) -> Optional[float]:
        """Calculate implied volatility using numerical methods."""
        try:
            if market_price <= 0 or time_to_expiry <= 0:
                return None
            
            option_type_enum = OptionType.CALL if option_type.lower() == 'call' else OptionType.PUT
            
            def objective(vol):
                if vol <= 0:
                    return 1e6
                try:
                    params = OptionParameters(
                        spot_price=spot_price,
                        strike_price=strike,
                        time_to_expiry=time_to_expiry,
                        volatility=vol,
                        risk_free_rate=risk_free_rate,
                        option_type=option_type_enum
                    )
                    theoretical_price = self.bs_model.calculate_option_price(params).option_price
                    return abs(theoretical_price - market_price)
                except:
                    return 1e6
            
            # Use bounded optimization
            result = minimize_scalar(objective, bounds=(0.01, 5.0), method='bounded')
            
            if result.success and result.fun < market_price * 0.01:  # 1% tolerance
                return result.x
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"IV calculation failed: {e}")
            return None
    
    def _calculate_atm_volatility(
        self,
        surface_points: List[VolatilitySurfacePoint],
        spot_price: float
    ) -> float:
        """Calculate ATM volatility from surface points."""
        try:
            # Find points closest to ATM
            atm_points = []
            for point in surface_points:
                if 0.95 <= point.moneyness <= 1.05:  # Within 5% of ATM
                    atm_points.append(point)
            
            if not atm_points:
                # Fallback to all points
                atm_points = surface_points
            
            # Volume-weighted average if volume data available
            total_volume = sum(p.volume for p in atm_points)
            if total_volume > 0:
                weighted_vol = sum(p.implied_vol * p.volume for p in atm_points) / total_volume
            else:
                weighted_vol = np.mean([p.implied_vol for p in atm_points])
            
            return weighted_vol
            
        except Exception as e:
            self.logger.warning(f"ATM volatility calculation failed: {e}")
            return 0.5  # Default fallback
    
    def _calculate_skew_metrics(
        self,
        surface_points: List[VolatilitySurfacePoint],
        spot_price: float
    ) -> Dict[str, float]:
        """Calculate various skew metrics."""
        try:
            # Group by tenor
            tenor_groups = {}
            for point in surface_points:
                tenor_key = round(point.time_to_expiry * 365.25)  # Round to nearest day
                if tenor_key not in tenor_groups:
                    tenor_groups[tenor_key] = []
                tenor_groups[tenor_key].append(point)
            
            # Calculate average skew across tenors
            skew_slopes = []
            for tenor, points in tenor_groups.items():
                if len(points) >= 3:
                    moneyness = [p.moneyness for p in points]
                    ivs = [p.implied_vol for p in points]
                    slope = self._calculate_skew_slope(moneyness, ivs)
                    if slope is not None:
                        skew_slopes.append(slope)
            
            avg_skew_slope = np.mean(skew_slopes) if skew_slopes else 0.0
            
            return {
                'average_skew_slope': avg_skew_slope,
                'skew_observations': len(skew_slopes),
                'tenors_analyzed': list(tenor_groups.keys())
            }
            
        except Exception as e:
            self.logger.warning(f"Skew metrics calculation failed: {e}")
            return {'average_skew_slope': 0.0, 'skew_observations': 0, 'tenors_analyzed': []}
    
    def _calculate_skew_slope(self, moneyness: List[float], iv_values: List[float]) -> Optional[float]:
        """Calculate skew slope using linear regression."""
        try:
            if len(moneyness) < 2:
                return None
            
            # Simple linear regression
            n = len(moneyness)
            sum_x = sum(moneyness)
            sum_y = sum(iv_values)
            sum_xy = sum(x * y for x, y in zip(moneyness, iv_values))
            sum_x2 = sum(x * x for x in moneyness)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if abs(denominator) < 1e-10:
                return None
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            return slope
            
        except Exception as e:
            self.logger.warning(f"Skew slope calculation failed: {e}")
            return None
    
    def _find_closest_atm_index(self, moneyness_values: List[float]) -> Optional[int]:
        """Find index of point closest to ATM (moneyness = 1.0)."""
        try:
            distances = [abs(m - 1.0) for m in moneyness_values]
            return distances.index(min(distances))
        except:
            return None
    
    def _calculate_risk_reversal(self, points: List[VolatilitySurfacePoint], delta: float) -> float:
        """Calculate risk reversal metric."""
        # Simplified implementation - would need delta calculation for proper implementation
        return 0.0
    
    def _calculate_butterfly(self, points: List[VolatilitySurfacePoint], delta: float) -> float:
        """Calculate butterfly metric."""
        # Simplified implementation
        return 0.0
    
    def _get_empty_surface_data(self, spot_price: float, reference_date: datetime) -> VolatilitySurfaceData:
        """Return empty surface data structure."""
        return VolatilitySurfaceData(
            surface_points=[],
            spot_price=spot_price,
            reference_date=reference_date,
            atm_volatility=0.5,
            total_volume=0.0,
            skew_metrics={'average_skew_slope': 0.0, 'skew_observations': 0, 'tenors_analyzed': []}
        )


# Convenience functions
def create_sample_options_data() -> pd.DataFrame:
    """Create sample options data for testing."""
    np.random.seed(42)
    
    base_spot = 50000.0
    strikes = np.arange(45000, 56000, 1000)
    expiries = [7, 14, 30, 60]  # Days
    
    data = []
    for days in expiries:
        tte = days / 365.25
        for strike in strikes:
            for option_type in ['call', 'put']:
                # Generate realistic implied volatility
                moneyness = strike / base_spot
                base_iv = 0.8
                skew_adjustment = (1.0 - moneyness) * 0.2
                term_structure_adj = (60 - days) / 60 * 0.1
                noise = np.random.normal(0, 0.05)
                
                iv = base_iv + skew_adjustment + term_structure_adj + noise
                iv = max(0.3, min(2.0, iv))  # Clamp between 30% and 200%
                
                data.append({
                    'strike': strike,
                    'time_to_maturity': tte,
                    'option_type': option_type,
                    'implied_volatility': iv,
                    'volume': np.random.randint(10, 100),
                    'open_interest': np.random.randint(50, 500)
                })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Test the volatility surface analyzer
    print("🧪 Testing Volatility Surface Analyzer")
    print("=" * 45)
    
    analyzer = VolatilitySurfaceAnalyzer()
    
    # Create sample data
    print("📊 Creating sample options data...")
    options_df = create_sample_options_data()
    print(f"✅ Created {len(options_df)} option data points")
    
    # Build volatility surface
    print("\n🏗️ Building volatility surface...")
    surface_data = analyzer.build_volatility_surface(options_df, 50000.0)
    print(f"✅ Surface built with {len(surface_data.surface_points)} points")
    print(f"📈 ATM Volatility: {surface_data.atm_volatility:.1%}")
    print(f"📊 Total Volume: {surface_data.total_volume:,.0f}")
    
    # Analyze volatility skew
    print("\n📈 Analyzing volatility skew...")
    skew_30d = analyzer.analyze_volatility_skew(surface_data, 30)
    if "error" not in skew_30d:
        print(f"✅ 30D Skew analysis: {skew_30d['data_points']} points")
        print(f"📊 ATM Vol: {skew_30d['atm_volatility']:.1%}")
        print(f"📈 Skew Slope: {skew_30d['skew_slope']:.4f}")
    else:
        print(f"⚠️ 30D Skew analysis failed: {skew_30d['error']}")
    
    print("\n✅ Volatility Surface Analyzer test completed!")