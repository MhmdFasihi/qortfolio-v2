# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Qortfolio V2 - Dark Theme Implementation
Location: src/dashboard/main_dashboard.py

Step 1: Complete dark theme with smokey gray and smokey royal purple
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import warnings
from scipy.stats import norm
from scipy.interpolate import griddata
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# ===================== CONFIGURATION =====================

st.set_page_config(
    page_title="Qortfolio V2",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"  # No sidebar
)

# Dark Smokey Theme CSS
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #2D2D34 0%, #1A1A2E 50%, #16213E 100%);
        color: #E8E8E8;
    }
    
    /* Main content area */
    .main {
        background: transparent;
        padding: 1rem;
    }
    
    /* Hide sidebar completely */
    .css-1d391kg {
        display: none;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #4A4A5A 0%, #6B46C1 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(107, 70, 193, 0.3);
        border: 1px solid rgba(232, 232, 232, 0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #3A3A42 0%, #5B21B6 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: #E8E8E8;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(232, 232, 232, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Chart containers */
    .chart-container {
        background: linear-gradient(135deg, #2D2D34 0%, #3A3A42 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(232, 232, 232, 0.1);
        margin: 1rem 0;
    }
    
    /* Navigation tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(135deg, #3A3A42 0%, #4A4A5A 100%);
        border-radius: 10px;
        padding: 0.5rem;
        border: 1px solid rgba(232, 232, 232, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #B8B8B8;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6B46C1 0%, #8B5CF6 100%);
        color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(107, 70, 193, 0.4);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #E8E8E8;
        font-weight: 600;
    }
    
    h1 {
        background: linear-gradient(135deg, #8B5CF6 0%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Input components */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #3A3A42 0%, #4A4A5A 100%);
        border: 2px solid rgba(107, 70, 193, 0.3);
        border-radius: 8px;
        color: #E8E8E8;
    }
    
    .stSelectbox > div > div:hover {
        border-color: rgba(139, 92, 246, 0.6);
    }
    
    .stNumberInput > div > div > input {
        background: linear-gradient(135deg, #3A3A42 0%, #4A4A5A 100%);
        border: 2px solid rgba(107, 70, 193, 0.3);
        border-radius: 8px;
        color: #E8E8E8;
    }
    
    .stSlider > div > div {
        background: linear-gradient(135deg, #3A3A42 0%, #4A4A5A 100%);
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #6B46C1 0%, #8B5CF6 100%);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6B46C1 0%, #8B5CF6 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(107, 70, 193, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #8B5CF6 0%, #A78BFA 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(107, 70, 193, 0.5);
    }
    
    /* Dataframes */
    .stDataFrame {
        background: linear-gradient(135deg, #2D2D34 0%, #3A3A42 100%);
        border-radius: 10px;
        border: 1px solid rgba(232, 232, 232, 0.1);
    }
    
    /* Metrics */
    .stMetric {
        background: linear-gradient(135deg, #3A3A42 0%, #4A4A5A 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(232, 232, 232, 0.1);
    }
    
    .stMetric > div {
        color: #E8E8E8;
    }
    
    /* Alerts and messages */
    .stAlert {
        background: linear-gradient(135deg, #3A3A42 0%, #4A4A5A 100%);
        border: 1px solid rgba(232, 232, 232, 0.1);
        border-radius: 10px;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #065F46 0%, #047857 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .stError {
        background: linear-gradient(135deg, #7F1D1D 0%, #991B1B 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .stWarning {
        background: linear-gradient(135deg, #78350F 0%, #92400E 100%);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .stInfo {
        background: linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    /* Multiselect */
    .stMultiSelect > div > div {
        background: linear-gradient(135deg, #3A3A42 0%, #4A4A5A 100%);
        border: 2px solid rgba(107, 70, 193, 0.3);
        border-radius: 8px;
    }
    
    /* Checkbox */
    .stCheckbox > label {
        color: #E8E8E8;
    }
    
    /* Remove default streamlit styling */
    .css-1aumxhk {
        background: transparent;
    }
    
    /* Navigation styling */
    .nav-container {
        background: linear-gradient(135deg, #3A3A42 0%, #4A4A5A 100%);
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid rgba(232, 232, 232, 0.1);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Page content */
    .page-content {
        background: linear-gradient(135deg, #2D2D34 0%, #3A3A42 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(232, 232, 232, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-top: 3rem;
        border: 1px solid rgba(232, 232, 232, 0.1);
        text-align: center;
        color: #B8B8B8;
    }
</style>
""", unsafe_allow_html=True)

# ===================== DARK THEME PLOTLY CONFIG =====================

def get_dark_theme_layout():
    """Get consistent dark theme layout for all plotly charts."""
    return {
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'font': {
            'color': '#E8E8E8',
            'family': 'Arial, sans-serif',
            'size': 12
        },
        'colorway': [
            '#8B5CF6',  # Purple
            '#A78BFA',  # Light purple
            '#C4B5FD',  # Lighter purple
            '#6B46C1',  # Dark purple
            '#7C3AED',  # Medium purple
            '#9333EA',  # Another purple
            '#AB78FF',  # Soft purple
            '#B794F6'   # Lavender
        ],
        'xaxis': {
            'gridcolor': 'rgba(232, 232, 232, 0.1)',
            'linecolor': 'rgba(232, 232, 232, 0.2)',
            'tickcolor': 'rgba(232, 232, 232, 0.2)',
            'titlefont': {'color': '#E8E8E8'},
            'tickfont': {'color': '#B8B8B8'}
        },
        'yaxis': {
            'gridcolor': 'rgba(232, 232, 232, 0.1)',
            'linecolor': 'rgba(232, 232, 232, 0.2)',
            'tickcolor': 'rgba(232, 232, 232, 0.2)',
            'titlefont': {'color': '#E8E8E8'},
            'tickfont': {'color': '#B8B8B8'}
        },
        'legend': {
            'font': {'color': '#E8E8E8'},
            'bgcolor': 'rgba(58, 58, 66, 0.8)',
            'bordercolor': 'rgba(232, 232, 232, 0.1)',
            'borderwidth': 1
        }
    }

# ===================== DATA HANDLERS =====================

class RealTimeDataManager:
    """Real-time data manager with dark theme integration."""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 30
        
    def get_current_price(self, symbol: str) -> float:
        """Get current crypto price with caching."""
        cache_key = f"price_{symbol}"
        now = time.time()
        
        if cache_key in self.cache:
            price, timestamp = self.cache[cache_key]
            if now - timestamp < self.cache_timeout:
                return price
        
        try:
            if symbol == "BTC":
                ticker = yf.Ticker("BTC-USD")
            elif symbol == "ETH":
                ticker = yf.Ticker("ETH-USD")
            else:
                return 0.0
            
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                price = float(data['Close'].iloc[-1])
                self.cache[cache_key] = (price, now)
                return price
            else:
                return 95000.0 if symbol == "BTC" else 3200.0
        except:
            return 95000.0 if symbol == "BTC" else 3200.0
    
    def get_historical_volatility(self, symbol: str, days: int = 30) -> float:
        """Calculate historical volatility."""
        try:
            ticker_symbol = f"{symbol}-USD"
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period=f"{days}d")
            
            if len(hist) < 10:
                return 0.8
            
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(365)
            return max(0.3, min(2.0, volatility))
        except:
            return 0.8
    
    def get_options_data(self, symbol: str) -> pd.DataFrame:
        """Generate realistic options data for BTC/ETH."""
        spot_price = self.get_current_price(symbol)
        hv = self.get_historical_volatility(symbol)
        
        expiry_dates = []
        base_date = datetime.now()
        
        for i in range(1, 9):
            weekly = base_date + timedelta(weeks=i)
            expiry_dates.append(weekly.strftime("%Y-%m-%d"))
        
        for i in range(1, 7):
            monthly = base_date + timedelta(days=30*i)
            expiry_dates.append(monthly.strftime("%Y-%m-%d"))
        
        expiry_dates = sorted(list(set(expiry_dates)))[:12]
        
        strike_range = 0.4
        num_strikes = 20
        strikes = np.linspace(
            spot_price * (1 - strike_range),
            spot_price * (1 + strike_range),
            num_strikes
        )
        
        options_data = []
        risk_free_rate = 0.05
        
        for expiry_str in expiry_dates:
            expiry = datetime.strptime(expiry_str, "%Y-%m-%d")
            days_to_expiry = max(1, (expiry - base_date).days)
            time_to_maturity = days_to_expiry / 365.25
            
            for strike in strikes:
                moneyness = strike / spot_price
                iv_base = hv
                
                if moneyness < 0.9 or moneyness > 1.1:
                    iv_adjustment = 0.1 + 0.2 * abs(moneyness - 1)
                else:
                    iv_adjustment = 0.05
                
                iv = iv_base + iv_adjustment
                
                call_price, put_price = self.black_scholes_both(
                    spot_price, strike, time_to_maturity, risk_free_rate, iv
                )
                
                call_greeks = self.calculate_greeks(
                    spot_price, strike, time_to_maturity, risk_free_rate, iv, "call"
                )
                put_greeks = self.calculate_greeks(
                    spot_price, strike, time_to_maturity, risk_free_rate, iv, "put"
                )
                
                options_data.append({
                    'symbol': f'{symbol}-{expiry_str}-{int(strike)}-C',
                    'underlying': symbol,
                    'type': 'call',
                    'strike': strike,
                    'expiry': expiry_str,
                    'days_to_expiry': days_to_expiry,
                    'time_to_maturity': time_to_maturity,
                    'spot_price': spot_price,
                    'price': call_price,
                    'bid': call_price * 0.98,
                    'ask': call_price * 1.02,
                    'iv': iv,
                    'delta': call_greeks['delta'],
                    'gamma': call_greeks['gamma'],
                    'theta': call_greeks['theta'],
                    'vega': call_greeks['vega'],
                    'rho': call_greeks['rho']
                })
                
                options_data.append({
                    'symbol': f'{symbol}-{expiry_str}-{int(strike)}-P',
                    'underlying': symbol,
                    'type': 'put',
                    'strike': strike,
                    'expiry': expiry_str,
                    'days_to_expiry': days_to_expiry,
                    'time_to_maturity': time_to_maturity,
                    'spot_price': spot_price,
                    'price': put_price,
                    'bid': put_price * 0.98,
                    'ask': put_price * 1.02,
                    'iv': iv,
                    'delta': put_greeks['delta'],
                    'gamma': put_greeks['gamma'],
                    'theta': put_greeks['theta'],
                    'vega': put_greeks['vega'],
                    'rho': put_greeks['rho']
                })
        
        return pd.DataFrame(options_data)
    
    def black_scholes_both(self, S, K, T, r, sigma):
        """Calculate both call and put prices using Black-Scholes."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(0, call_price), max(0, put_price)
    
    def calculate_greeks(self, S, K, T, r, sigma, option_type):
        """Calculate option Greeks."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        phi_d1 = norm.pdf(d1)
        Phi_d1 = norm.cdf(d1)
        Phi_d2 = norm.cdf(d2)
        
        if option_type == "call":
            delta = Phi_d1
            rho = K * T * np.exp(-r * T) * Phi_d2 / 100
        else:
            delta = Phi_d1 - 1
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        gamma = phi_d1 / (S * sigma * np.sqrt(T))
        theta = ((-S * phi_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * Phi_d2) / 365
        vega = S * phi_d1 * np.sqrt(T) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

# Initialize data manager
@st.cache_resource
def get_data_manager():
    return RealTimeDataManager()

data_manager = get_data_manager()

# ===================== DASHBOARD PAGES =====================

def render_header():
    """Render minimal dark theme header."""
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem; text-align: center;">
            🚀 Qortfolio V2
        </h1>
        <p style="margin: 0.5rem 0 0 0; text-align: center; color: #B8B8B8; font-size: 1.1rem;">
            Professional Options Analytics Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Real-time market data header
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        btc_price = data_manager.get_current_price("BTC")
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0 0 0.5rem 0; color: #B8B8B8;">BTC Price</h4>
            <h2 style="margin: 0; color: #8B5CF6;">${btc_price:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        eth_price = data_manager.get_current_price("ETH")
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0 0 0.5rem 0; color: #B8B8B8;">ETH Price</h4>
            <h2 style="margin: 0; color: #A78BFA;">${eth_price:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        btc_vol = data_manager.get_historical_volatility("BTC") * 100
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0 0 0.5rem 0; color: #B8B8B8;">BTC Volatility</h4>
            <h2 style="margin: 0; color: #C4B5FD;">{btc_vol:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0 0 0.5rem 0; color: #B8B8B8;">Last Update</h4>
            <h2 style="margin: 0; color: #6B46C1;">{current_time}</h2>
        </div>
        """, unsafe_allow_html=True)

def market_overview_page():
    """Market Overview with dark theme."""
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    
    st.header("📈 Market Overview")
    
    # Controls
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_assets = st.multiselect(
            "Select Assets", 
            ["BTC", "ETH"], 
            default=["BTC", "ETH"]
        )
    with col2:
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=True)
    
    if auto_refresh:
        time.sleep(1)
        st.rerun()
    
    # Price charts
    st.subheader("💰 Price Charts")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("BTC Price (24h)", "ETH Price (24h)", "BTC vs ETH Correlation", "Volume Analysis")
    )
    
    colors = ['#8B5CF6', '#A78BFA']
    
    for i, asset in enumerate(["BTC", "ETH"]):
        if asset in selected_assets:
            try:
                ticker = yf.Ticker(f"{asset}-USD")
                hist = ticker.history(period="1d", interval="5m")
                
                if not hist.empty:
                    row = 1
                    col = 1 if asset == "BTC" else 2
                    
                    fig.add_trace(
                        go.Scatter(
                            x=hist.index,
                            y=hist['Close'],
                            name=f"{asset} Price",
                            line=dict(color=colors[i], width=3),
                            fill='tonexty' if i > 0 else None
                        ),
                        row=row, col=col
                    )
            except:
                st.warning(f"Could not load {asset} data")
    
    # Apply dark theme
    fig.update_layout(get_dark_theme_layout())
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Real-Time Market Analysis"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def options_chain_page():
    """Options Chain with dark theme (placeholder for Step 2)."""
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.header("⛓️ Options Chain Analysis")
    st.info("Options Chain redesign will be implemented in Step 2")
    st.markdown('</div>', unsafe_allow_html=True)

def volatility_surface_page():
    """Volatility Surface with dark theme (placeholder for Step 3)."""
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.header("🌋 Volatility Surface Analysis")
    st.info("Volatility Surface fixes will be implemented in Step 3")
    st.markdown('</div>', unsafe_allow_html=True)

def pnl_analysis_page():
    """PnL Analysis with dark theme (placeholder for Step 7)."""
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.header("💰 PnL Analysis")
    st.info("PnL Analysis integration will be implemented in Step 7")
    st.markdown('</div>', unsafe_allow_html=True)

def risk_management_page():
    """Risk Management with dark theme (placeholder for Step 8)."""
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.header("⚠️ Risk Management")
    st.info("Risk Management integration will be implemented in Step 8")
    st.markdown('</div>', unsafe_allow_html=True)

def portfolio_analytics_page():
    """Portfolio Analytics with dark theme (placeholder for Step 4)."""
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.header("💼 Portfolio Management")
    st.info("Portfolio Management core will be implemented in Step 4")
    st.markdown('</div>', unsafe_allow_html=True)

def asset_allocation_page():
    """Asset Allocation with dark theme (placeholder for Step 6)."""
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.header("📊 Asset Allocation")
    st.info("Asset Allocation with HRP/HERC will be implemented in Step 6")
    st.markdown('</div>', unsafe_allow_html=True)

# ===================== MAIN APPLICATION =====================

def main():
    """Main dark theme dashboard application."""
    
    # Render header
    render_header()
    
    # Navigation using tabs (no sidebar)
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    
    tabs = st.tabs([
        "📈 Market Overview",
        "⛓️ Options Chain", 
        "🌋 Volatility Surface",
        "💰 PnL Analysis",
        "⚠️ Risk Management", 
        "💼 Portfolio Management",
        "📊 Asset Allocation"
    ])
    
    with tabs[0]:
        market_overview_page()
    
    with tabs[1]:
        options_chain_page()
    
    with tabs[2]:
        volatility_surface_page()
    
    with tabs[3]:
        pnl_analysis_page()
    
    with tabs[4]:
        risk_management_page()
    
    with tabs[5]:
        portfolio_analytics_page()
    
    with tabs[6]:
        asset_allocation_page()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Auto-refresh
    if st.button("🔄 Refresh Data", key="main_refresh"):
        st.cache_data.clear()
        st.rerun()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🚀 <strong>Qortfolio V2</strong> - Professional Options Analytics Platform</p>
        <p>Copyright © 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()