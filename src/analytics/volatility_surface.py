# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Volatility Surface Analysis for Qortfolio V2
3D volatility surface construction and analysis from options market data
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.interpolate import griddata, RBFInterpolator
from scipy.optimize import minimize_scalar
import warnings

from models.options.black_scholes import BlackScholesModel
from core.logging import get_logger
from core.config import get_config


@dataclass
class VolatilitySurfacePoint:
    """Single point on the volatility surface."""
    strike: float
    time_to_maturity: float
    implied_volatility: float
    moneyness: float
    option_type: str
    market_price: float
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    volume: Optional[float] = None
    open_interest: Optional[float] = None


@dataclass
class VolatilitySurfaceData:
    """Complete volatility surface data."""
    surface_points: List[VolatilitySurfacePoint]
    spot_price: float
    surface_date: datetime
    currency: str
    
    # Interpolated surface
    strikes_grid: Optional[np.ndarray] = None
    times_grid: Optional[np.ndarray] = None
    volatility_grid: Optional[np.ndarray] = None
    
    # Surface metrics
    atm_volatility: Optional[float] = None
    volatility_skew: Optional[Dict[str, float]] = None
    term_structure: Optional[Dict[str, float]] = None


class VolatilitySurfaceAnalyzer:
    """
    Volatility surface analyzer for options market data.
    
    Features:
    - 3D volatility surface construction
    - Volatility smile/skew analysis
    - ATM term structure
    - Surface interpolation and smoothing
    - Market data quality filtering
    """
    
    def __init__(self):
        """Initialize volatility surface analyzer."""
        self.config = get_config()
        self.logger = get_logger("volatility_surface")
        self.bs_model = BlackScholesModel()
        
        # Configuration parameters
        self.min_time_to_maturity = self.config.get(
            'options_config.default_params.min_time_to_maturity', 1/365.25
        )
        self.max_time_to_maturity = 2.0  # 2 years max
        self.min_moneyness = 0.5  # 50% of spot
        self.max_moneyness = 2.0  # 200% of spot
        
        self.logger.info("Volatility surface analyzer initialized")
    
    def build_volatility_surface(self, options_data: pd.DataFrame, 
                                spot_price: float,
                                filter_data: bool = True) -> VolatilitySurfaceData:
        """
        Build volatility surface from options market data.
        
        Args:
            options_data: DataFrame with options market data
            spot_price: Current spot price
            filter_data: Apply data quality filters
            
        Returns:
            Complete volatility surface data
        """
        # Validate input data
        required_columns = ['strike', 'time_to_maturity', 'mark_price', 'option_type']
        if not all(col in options_data.columns for col in required_columns):
            raise ValueError(f"Options data missing required columns: {required_columns}")
        
        # Filter data if requested
        if filter_data:
            filtered_data = self._filter_options_data(options_data, spot_price)
        else:
            filtered_data = options_data.copy()
        
        if len(filtered_data) == 0:
            raise ValueError("No valid options data after filtering")
        
        # Calculate implied volatilities
        surface_points = self._calculate_implied_volatilities(filtered_data, spot_price)
        
        if len(surface_points) == 0:
            raise ValueError("Failed to calculate implied volatilities")
        
        # Create surface data
        surface_data = VolatilitySurfaceData(
            surface_points=surface_points,
            spot_price=spot_price,
            surface_date=datetime.now(),
            currency=filtered_data.get('Currency', ['Unknown']).iloc[0] if 'Currency' in filtered_data.columns else 'Unknown'
        )
        
        # Build interpolated surface
        self._build_interpolated_surface(surface_data)
        
        # Calculate surface metrics
        self._calculate_surface_metrics(surface_data)
        
        self.logger.info(f"Volatility surface built successfully", extra={
            "surface_points": len(surface_points),
            "spot_price": spot_price,
            "atm_volatility": surface_data.atm_volatility
        })
        
        return surface_data
    
    def plot_volatility_surface_3d(self, surface_data: VolatilitySurfaceData,
                                  title: Optional[str] = None) -> go.Figure:
        """
        Create 3D plotly visualization of volatility surface.
        
        Args:
            surface_data: Volatility surface data
            title: Plot title
            
        Returns:
            Plotly 3D surface figure
        """
        if surface_data.volatility_grid is None:
            raise ValueError("Interpolated surface not available")
        
        # Create meshgrid for plotting
        strikes_mesh, times_mesh = np.meshgrid(
            surface_data.strikes_grid, 
            surface_data.times_grid, 
            indexing='ij'
        )
        
        # Create 3D surface plot
        fig = go.Figure()
        
        # Add surface
        fig.add_trace(go.Surface(
            x=strikes_mesh,
            y=times_mesh * 365.25,  # Convert to days
            z=surface_data.volatility_grid * 100,  # Convert to percentage
            colorscale='Viridis',
            name='Volatility Surface',
            hovertemplate='Strike: %{x:.0f}<br>Days: %{y:.0f}<br>IV: %{z:.1f}%<extra></extra>'
        ))
        
        # Add market data points
        market_strikes = [p.strike for p in surface_data.surface_points]
        market_times = [p.time_to_maturity * 365.25 for p in surface_data.surface_points]
        market_vols = [p.implied_volatility * 100 for p in surface_data.surface_points]
        
        fig.add_trace(go.Scatter3d(
            x=market_strikes,
            y=market_times,
            z=market_vols,
            mode='markers',
            marker=dict(
                size=4,
                color='red',
                symbol='circle'
            ),
            name='Market Data',
            hovertemplate='Strike: %{x:.0f}<br>Days: %{y:.0f}<br>IV: %{z:.1f}%<extra></extra>'
        ))
        
        # Update layout
        title = title or f'Volatility Surface - {surface_data.currency} (Spot: {surface_data.spot_price:,.0f})'
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Strike Price',
                yaxis_title='Days to Expiry',
                zaxis_title='Implied Volatility (%)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def plot_volatility_smile(self, surface_data: VolatilitySurfaceData,
                             expiry_days: Optional[List[int]] = None) -> go.Figure:
        """
        Plot volatility smile for specific expiries.
        
        Args:
            surface_data: Volatility surface data
            expiry_days: List of expiry days to plot (auto-select if None)
            
        Returns:
            Plotly figure with volatility smiles
        """
        if expiry_days is None:
            # Auto-select representative expiries
            all_days = [p.time_to_maturity * 365.25 for p in surface_data.surface_points]
            unique_days = sorted(set([int(d) for d in all_days if d >= 1]))
            
            # Select up to 5 representative expiries
            if len(unique_days) <= 5:
                expiry_days = unique_days
            else:
                # Select spread across available expiries
                indices = np.linspace(0, len(unique_days)-1, 5, dtype=int)
                expiry_days = [unique_days[i] for i in indices]
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, target_days in enumerate(expiry_days):
            # Find points close to target expiry
            target_time = target_days / 365.25
            tolerance = 5 / 365.25  # 5 days tolerance
            
            expiry_points = [
                p for p in surface_data.surface_points
                if abs(p.time_to_maturity - target_time) <= tolerance
            ]
            
            if not expiry_points:
                continue
            
            # Sort by moneyness
            expiry_points.sort(key=lambda x: x.moneyness)
            
            moneyness = [p.moneyness for p in expiry_points]
            volatilities = [p.implied_volatility * 100 for p in expiry_points]
            
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=moneyness,
                y=volatilities,
                mode='lines+markers',
                name=f'{target_days}D',
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
                hovertemplate=f'{target_days}D: Moneyness=%{{x:.2f}}, IV=%{{y:.1f}}%<extra></extra>'
            ))
        
        # Add ATM line
        fig.add_vline(x=1.0, line_dash="dash", line_color="gray", 
                     annotation_text="ATM", annotation_position="top")
        
        fig.update_layout(
            title=f'Volatility Smile - {surface_data.currency}',
            xaxis_title='Moneyness (Strike/Spot)',
            yaxis_title='Implied Volatility (%)',
            hovermode='closest',
            width=700,
            height=500
        )
        
        return fig
    
    def plot_term_structure(self, surface_data: VolatilitySurfaceData) -> go.Figure:
        """
        Plot ATM volatility term structure.
        
        Args:
            surface_data: Volatility surface data
            
        Returns:
            Plotly figure with term structure
        """
        # Find ATM or close-to-ATM points for each expiry
        atm_points = []
        
        # Group by expiry
        expiry_groups = {}
        for point in surface_data.surface_points:
            days = int(point.time_to_maturity * 365.25)
            if days not in expiry_groups:
                expiry_groups[days] = []
            expiry_groups[days].append(point)
        
        # Find ATM vol for each expiry
        for days, points in expiry_groups.items():
            if not points:
                continue
            
            # Find closest to ATM (moneyness = 1.0)
            atm_point = min(points, key=lambda p: abs(p.moneyness - 1.0))
            
            # Only include if reasonably close to ATM
            if abs(atm_point.moneyness - 1.0) < 0.2:  # Within 20% of ATM
                atm_points.append((days, atm_point.implied_volatility * 100))
        
        if not atm_points:
            raise ValueError("No ATM points found for term structure")
        
        # Sort by days
        atm_points.sort(key=lambda x: x[0])
        days, volatilities = zip(*atm_points)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=days,
            y=volatilities,
            mode='lines+markers',
            name='ATM Volatility',
            line=dict(color='blue', width=3),
            marker=dict(size=8, color='blue'),
            hovertemplate='Days: %{x}<br>ATM IV: %{y:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'ATM Volatility Term Structure - {surface_data.currency}',
            xaxis_title='Days to Expiry',
            yaxis_title='ATM Implied Volatility (%)',
            hovermode='closest',
            width=700,
            height=400
        )
        
        return fig
    
    def analyze_volatility_skew(self, surface_data: VolatilitySurfaceData,
                              target_expiry_days: int = 30) -> Dict[str, float]:
        """
        Analyze volatility skew for a specific expiry.
        
        Args:
            surface_data: Volatility surface data
            target_expiry_days: Target expiry in days
            
        Returns:
            Dictionary with skew metrics
        """
        target_time = target_expiry_days / 365.25
        tolerance = 7 / 365.25  # 7 days tolerance
        
        # Find points for target expiry
        expiry_points = [
            p for p in surface_data.surface_points
            if abs(p.time_to_maturity - target_time) <= tolerance
        ]
        
        if len(expiry_points) < 3:
            return {"error": f"Insufficient data for {target_expiry_days}D expiry"}
        
        # Sort by moneyness
        expiry_points.sort(key=lambda x: x.moneyness)
        
        # Calculate skew metrics
        moneyness_values = [p.moneyness for p in expiry_points]
        volatilities = [p.implied_volatility for p in expiry_points]
        
        # Find ATM, 25 delta put, 25 delta call equivalents (approximate)
        atm_vol = None
        put_25d_vol = None
        call_25d_vol = None
        
        # ATM (closest to 1.0 moneyness)
        atm_idx = min(range(len(moneyness_values)), key=lambda i: abs(moneyness_values[i] - 1.0))
        atm_vol = volatilities[atm_idx]
        
        # 25-delta put (approximately 0.9 moneyness for short-term options)
        put_targets = [i for i, m in enumerate(moneyness_values) if 0.85 <= m <= 0.95]
        if put_targets:
            put_25d_vol = volatilities[put_targets[0]]
        
        # 25-delta call (approximately 1.1 moneyness for short-term options)  
        call_targets = [i for i, m in enumerate(moneyness_values) if 1.05 <= m <= 1.15]
        if call_targets:
            call_25d_vol = volatilities[call_targets[0]]
        
        skew_metrics = {
            "expiry_days": target_expiry_days,
            "atm_volatility": atm_vol * 100 if atm_vol else None,
            "data_points": len(expiry_points),
            "moneyness_range": [min(moneyness_values), max(moneyness_values)]
        }
        
        # Calculate skew if we have the required points
        if put_25d_vol and call_25d_vol and atm_vol:
            skew_metrics.update({
                "put_25d_vol": put_25d_vol * 100,
                "call_25d_vol": call_25d_vol * 100,
                "risk_reversal": (call_25d_vol - put_25d_vol) * 100,  # RR = Call25 - Put25
                "butterfly": ((put_25d_vol + call_25d_vol) / 2 - atm_vol) * 100  # BF = (Put25+Call25)/2 - ATM
            })
        
        # Calculate slope (linear approximation)
        if len(expiry_points) >= 2:
            # Simple linear regression for skew slope
            x = np.array(moneyness_values)
            y = np.array(volatilities)
            slope = np.polyfit(x, y, 1)[0]
            skew_metrics["skew_slope"] = slope * 100  # Slope in vol% per moneyness unit
        
        return skew_metrics
    
    # Private helper methods
    
    def _filter_options_data(self, options_data: pd.DataFrame, spot_price: float) -> pd.DataFrame:
        """Filter options data for quality and relevance."""
        filtered = options_data.copy()
        
        # Filter by time to maturity
        if 'time_to_maturity' in filtered.columns:
            filtered = filtered[
                (filtered['time_to_maturity'] >= self.min_time_to_maturity) &
                (filtered['time_to_maturity'] <= self.max_time_to_maturity)
            ]
        
        # Filter by moneyness
        if 'strike' in filtered.columns:
            moneyness = filtered['strike'] / spot_price
            filtered = filtered[
                (moneyness >= self.min_moneyness) &
                (moneyness <= self.max_moneyness)
            ]
        
        # Filter by price quality
        if 'mark_price' in filtered.columns:
            # Remove zero or negative prices
            filtered = filtered[filtered['mark_price'] > 0]
            
            # Remove options with prices below minimum tick (if very cheap)
            min_price = spot_price * 0.0001  # 0.01% of spot
            filtered = filtered[filtered['mark_price'] >= min_price]
        
        # Filter by bid-ask spread if available
        if 'bid_price' in filtered.columns and 'ask_price' in filtered.columns:
            # Calculate spread
            spread = filtered['ask_price'] - filtered['bid_price']
            spread_pct = spread / filtered['mark_price']
            
            # Remove options with very wide spreads (>50%)
            filtered = filtered[spread_pct <= 0.5]
        
        self.logger.debug(f"Filtered options data: {len(options_data)} -> {len(filtered)} options")
        
        return filtered
    
    def _calculate_implied_volatilities(self, options_data: pd.DataFrame, 
                                      spot_price: float) -> List[VolatilitySurfacePoint]:
        """Calculate implied volatilities for options data."""
        surface_points = []
        
        risk_free_rate = self.config.get('options_config.default_params.risk_free_rate', 0.05)
        
        for _, row in options_data.iterrows():
            try:
                strike = row['strike']
                time_to_maturity = row['time_to_maturity']
                market_price = row['mark_price']
                option_type = row['option_type'].lower()
                
                # Calculate implied volatility
                implied_vol = self.bs_model.calculate_implied_volatility(
                    market_price=market_price,
                    spot=spot_price,
                    strike=strike,
                    time_to_maturity=time_to_maturity,
                    option_type=option_type,
                    risk_free_rate=risk_free_rate,
                    max_iterations=50,
                    tolerance=1e-5
                )
                
                if implied_vol is not None and 0.01 <= implied_vol <= 5.0:  # Reasonable IV bounds
                    moneyness = strike / spot_price
                    
                    point = VolatilitySurfacePoint(
                        strike=strike,
                        time_to_maturity=time_to_maturity,
                        implied_volatility=implied_vol,
                        moneyness=moneyness,
                        option_type=option_type,
                        market_price=market_price,
                        bid_price=row.get('bid_price'),
                        ask_price=row.get('ask_price'),
                        volume=row.get('volume'),
                        open_interest=row.get('open_interest')
                    )
                    
                    surface_points.append(point)
                    
            except Exception as e:
                self.logger.debug(f"Failed to calculate IV for option: {e}")
                continue
        
        self.logger.info(f"Calculated implied volatilities for {len(surface_points)} options")
        
        return surface_points
    
    def _build_interpolated_surface(self, surface_data: VolatilitySurfaceData):
        """Build interpolated volatility surface."""
        if len(surface_data.surface_points) < 4:
            self.logger.warning("Insufficient points for surface interpolation")
            return
        
        # Extract data points
        strikes = [p.strike for p in surface_data.surface_points]
        times = [p.time_to_maturity for p in surface_data.surface_points]
        volatilities = [p.implied_volatility for p in surface_data.surface_points]
        
        # Create coordinate arrays
        points = np.column_stack((strikes, times))
        values = np.array(volatilities)
        
        # Create interpolation grid
        strike_min, strike_max = min(strikes), max(strikes)
        time_min, time_max = min(times), max(times)
        
        # Create a reasonable grid
        n_strikes = min(50, int((strike_max - strike_min) / (surface_data.spot_price * 0.05)))
        n_times = min(30, len(set([int(t * 365.25) for t in times])))
        
        n_strikes = max(10, n_strikes)
        n_times = max(5, n_times)
        
        surface_data.strikes_grid = np.linspace(strike_min, strike_max, n_strikes)
        surface_data.times_grid = np.linspace(time_min, time_max, n_times)
        
        # Create meshgrid for interpolation
        strikes_mesh, times_mesh = np.meshgrid(
            surface_data.strikes_grid, 
            surface_data.times_grid, 
            indexing='ij'
        )
        
        grid_points = np.column_stack((strikes_mesh.ravel(), times_mesh.ravel()))
        
        try:
            # Use RBF interpolation for smooth surface
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Try RBF interpolation first
                rbf = RBFInterpolator(points, values, kernel='thin_plate_spline', smoothing=0.01)
                interpolated = rbf(grid_points)
                
            surface_data.volatility_grid = interpolated.reshape(strikes_mesh.shape)
            
            # Ensure reasonable bounds
            surface_data.volatility_grid = np.clip(surface_data.volatility_grid, 0.01, 5.0)
            
        except Exception as e:
            self.logger.warning(f"RBF interpolation failed, using griddata: {e}")
            
            # Fallback to griddata
            try:
                interpolated = griddata(
                    points, values, grid_points, 
                    method='cubic', fill_value=np.nan
                )
                surface_data.volatility_grid = interpolated.reshape(strikes_mesh.shape)
                
                # Fill NaN values with nearest neighbor interpolation
                if np.any(np.isnan(surface_data.volatility_grid)):
                    interpolated_linear = griddata(
                        points, values, grid_points, 
                        method='nearest'
                    )
                    mask = np.isnan(surface_data.volatility_grid.ravel())
                    surface_data.volatility_grid.ravel()[mask] = interpolated_linear[mask]
                
            except Exception as e2:
                self.logger.error(f"Surface interpolation failed: {e2}")
                surface_data.volatility_grid = None
    
    def _calculate_surface_metrics(self, surface_data: VolatilitySurfaceData):
        """Calculate surface metrics like ATM vol and skew."""
        # Find ATM volatility (closest to spot price, shortest expiry)
        atm_candidates = [
            p for p in surface_data.surface_points
            if abs(p.moneyness - 1.0) < 0.1  # Within 10% of ATM
        ]
        
        if atm_candidates:
            # Choose shortest expiry ATM option
            atm_point = min(atm_candidates, key=lambda p: p.time_to_maturity)
            surface_data.atm_volatility = atm_point.implied_volatility
        
        # Calculate basic term structure
        term_structure = {}
        standard_expiries = [7, 14, 30, 60, 90, 180, 365]
        
        for days in standard_expiries:
            target_time = days / 365.25
            tolerance = 3 / 365.25  # 3 days tolerance
            
            candidates = [
                p for p in surface_data.surface_points
                if abs(p.time_to_maturity - target_time) <= tolerance
                and abs(p.moneyness - 1.0) < 0.15  # Near ATM
            ]
            
            if candidates:
                closest = min(candidates, key=lambda p: abs(p.moneyness - 1.0))
                term_structure[f"{days}D"] = closest.implied_volatility
        
        surface_data.term_structure = term_structure


# Convenience functions
def analyze_options_volatility(options_data: pd.DataFrame, spot_price: float) -> VolatilitySurfaceData:
    """
    Quick volatility surface analysis.
    
    Args:
        options_data: Options market data
        spot_price: Current spot price
        
    Returns:
        Volatility surface data
    """
    analyzer = VolatilitySurfaceAnalyzer()
    return analyzer.build_volatility_surface(options_data, spot_price)


def create_volatility_dashboard(surface_data: VolatilitySurfaceData) -> Dict[str, go.Figure]:
    """
    Create complete volatility analysis dashboard.
    
    Args:
        surface_data: Volatility surface data
        
    Returns:
        Dictionary of plotly figures
    """
    analyzer = VolatilitySurfaceAnalyzer()
    
    figures = {}
    
    try:
        figures['surface_3d'] = analyzer.plot_volatility_surface_3d(surface_data)
    except Exception as e:
        print(f"Failed to create 3D surface: {e}")
    
    try:
        figures['volatility_smile'] = analyzer.plot_volatility_smile(surface_data)
    except Exception as e:
        print(f"Failed to create volatility smile: {e}")
    
    try:
        figures['term_structure'] = analyzer.plot_term_structure(surface_data)
    except Exception as e:
        print(f"Failed to create term structure: {e}")
    
    return figures


if __name__ == "__main__":
    # Test the volatility surface analyzer
    print("🧪 Testing Volatility Surface Analyzer")
    print("=" * 40)
    
    # Create sample options data for testing
    np.random.seed(42)
    
    spot_price = 50000.0
    sample_data = []
    
    # Generate sample options data
    strikes = np.arange(40000, 65000, 2000)  # Range of strikes
    expiries = [7, 14, 30, 60, 90]  # Days to expiry
    
    for days in expiries:
        tte = days / 365.25
        for strike in strikes:
            moneyness = strike / spot_price
            
            # Create realistic implied volatility pattern
            # Higher vol for OTM options and shorter expiries
            base_vol = 0.8
            skew_adjustment = (1.0 - moneyness) * 0.3  # Volatility skew
            term_adjustment = max(0, (0.25 - tte) * 0.2)  # Term structure
            noise = np.random.normal(0, 0.05)  # Random noise
            
            iv = base_vol + skew_adjustment + term_adjustment + noise
            iv = max(0.2, min(2.0, iv))  # Bound between 20% and 200%
            
            # Calculate Black-Scholes price for this IV
            from models.options.black_scholes import price_option
            
            for option_type in ['call', 'put']:
                try:
                    price = price_option(spot_price, strike, tte, iv, option_type, 0.05)
                    
                    if price > 0:
                        sample_data.append({
                            'strike': strike,
                            'time_to_maturity': tte,
                            'mark_price': price,
                            'option_type': option_type,
                            'Currency': 'BTC'
                        })
                except:
                    continue
    
    options_df = pd.DataFrame(sample_data)
    print(f"Generated {len(options_df)} sample options for testing")
    
    # Test volatility surface analysis
    try:
        analyzer = VolatilitySurfaceAnalyzer()
        surface_data = analyzer.build_volatility_surface(options_df, spot_price)
        
        print(f"✅ Volatility surface built successfully")
        print(f"   Surface points: {len(surface_data.surface_points)}")
        print(f"   ATM volatility: {surface_data.atm_volatility:.1%}")
        print(f"   Currency: {surface_data.currency}")
        
        # Test skew analysis
        skew_analysis = analyzer.analyze_volatility_skew(surface_data, 30)
        print(f"   30D skew analysis: {skew_analysis}")
        
        # Test plotting (won't display in terminal but will create figures)
        print("   Creating visualization plots...")
        
        try:
            fig_3d = analyzer.plot_volatility_surface_3d(surface_data)
            print("   ✅ 3D surface plot created")
        except Exception as e:
            print(f"   ⚠️ 3D plot failed: {e}")
        
        try:
            fig_smile = analyzer.plot_volatility_smile(surface_data)
            print("   ✅ Volatility smile plot created")
        except Exception as e:
            print(f"   ⚠️ Smile plot failed: {e}")
        
        try:
            fig_term = analyzer.plot_term_structure(surface_data)
            print("   ✅ Term structure plot created")
        except Exception as e:
            print(f"   ⚠️ Term structure plot failed: {e}")
        
    except Exception as e:
        print(f"❌ Volatility surface test failed: {e}")
    
    print("\n✅ Volatility surface analyzer test completed!")