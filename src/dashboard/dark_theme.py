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
    """Options Chain Analysis - Deribit Style Layout with Black-Scholes Values."""
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    
    st.header("⛓️ Options Chain Analysis")
    
    # Asset and expiry selection
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    
    with col1:
        selected_asset = st.selectbox("Select Asset", ["BTC", "ETH"], index=0, key="options_asset")
    
    with col2:
        # Get options data first to populate expiry dropdown
        try:
            options_df = data_manager.get_options_data(selected_asset)
            available_expiries = sorted(options_df['expiry'].unique()) if not options_df.empty else ['2025-02-07']
            selected_expiry = st.selectbox("Select Expiry", available_expiries, key="options_expiry")
        except:
            selected_expiry = st.selectbox("Select Expiry", ['2025-02-07'], key="options_expiry")
    
    with col3:
        moneyness_filter = st.selectbox("Moneyness Filter", ["All", "ITM", "ATM", "OTM"], index=0)
    
    with col4:
        if st.button("🔄 Refresh Options", key="refresh_options"):
            st.cache_data.clear()
            st.rerun()
    
    try:
        # Get options data
        options_df = data_manager.get_options_data(selected_asset)
        
        if options_df.empty:
            st.error("No options data available")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        # Filter by selected expiry
        options_df = options_df[options_df['expiry'] == selected_expiry]
        
        if options_df.empty:
            st.error(f"No options data for expiry {selected_expiry}")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        # Get current spot price and calculate moneyness
        spot_price = options_df['spot_price'].iloc[0]
        options_df['moneyness'] = options_df['strike'] / spot_price
        
        # Apply moneyness filter
        if moneyness_filter == "ITM":
            options_df = options_df[
                ((options_df['type'] == 'call') & (options_df['moneyness'] < 1.0)) |
                ((options_df['type'] == 'put') & (options_df['moneyness'] > 1.0))
            ]
        elif moneyness_filter == "ATM":
            options_df = options_df[
                (options_df['moneyness'] >= 0.95) & (options_df['moneyness'] <= 1.05)
            ]
        elif moneyness_filter == "OTM":
            options_df = options_df[
                ((options_df['type'] == 'call') & (options_df['moneyness'] > 1.0)) |
                ((options_df['type'] == 'put') & (options_df['moneyness'] < 1.0))
            ]
        
        # Market overview header (Deribit style)
        st.markdown("---")
        
        header_col1, header_col2, header_col3 = st.columns([2, 2, 2])
        
        with header_col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #3A3A42 0%, #5B21B6 100%); 
                        padding: 1rem; border-radius: 8px; text-align: center;">
                <h4 style="margin: 0; color: #B8B8B8;">Underlying Future</h4>
                <h3 style="margin: 0.5rem 0 0 0; color: #8B5CF6;">${spot_price:,.2f}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with header_col2:
            days_to_expiry = options_df['days_to_expiry'].iloc[0]
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #3A3A42 0%, #5B21B6 100%); 
                        padding: 1rem; border-radius: 8px; text-align: center;">
                <h4 style="margin: 0; color: #B8B8B8;">Time to Expiry</h4>
                <h3 style="margin: 0.5rem 0 0 0; color: #A78BFA;">{days_to_expiry}d ({days_to_expiry/7:.1f}w)</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with header_col3:
            avg_iv = options_df['iv'].mean() * 100
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #3A3A42 0%, #5B21B6 100%); 
                        padding: 1rem; border-radius: 8px; text-align: center;">
                <h4 style="margin: 0; color: #B8B8B8;">Average IV</h4>
                <h3 style="margin: 0.5rem 0 0 0; color: #C4B5FD;">{avg_iv:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Prepare data in Deribit style
        calls_df = options_df[options_df['type'] == 'call'].copy()
        puts_df = options_df[options_df['type'] == 'put'].copy()
        
        # Get unique strikes that exist in both calls and puts
        call_strikes = set(calls_df['strike'].values)
        put_strikes = set(puts_df['strike'].values)
        common_strikes = sorted(call_strikes.intersection(put_strikes))
        
        if not common_strikes:
            st.error("No matching strikes found for calls and puts")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        # Create Deribit-style layout
        st.subheader(f"📊 {selected_asset} Options Chain - {selected_expiry}")
        
        # Header row
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4A4A5A 0%, #6B46C1 100%); 
                    padding: 0.8rem; border-radius: 8px 8px 0 0; 
                    border: 1px solid rgba(232, 232, 232, 0.1);">
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr 1fr 1fr 100px 1fr 1fr 1fr 1fr 1fr 1fr; 
                        gap: 8px; text-align: center; color: #E8E8E8; font-weight: bold; font-size: 0.9rem;">
                <div>Open</div>
                <div>Delta</div>
                <div>IV</div>
                <div>Bid</div>
                <div>Mark</div>
                <div>Ask</div>
                <div style="color: #8B5CF6;">Strike</div>
                <div>Bid</div>
                <div>Mark</div>
                <div>Ask</div>
                <div>IV</div>
                <div>Delta</div>
                <div>Open</div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr 1fr 1fr 100px 1fr 1fr 1fr 1fr 1fr 1fr; 
                        gap: 8px; text-align: center; color: #B8B8B8; font-size: 0.8rem; margin-top: 0.3rem;">
                <div colspan="6" style="grid-column: 1 / 7; text-align: center;">CALLS</div>
                <div></div>
                <div colspan="6" style="grid-column: 8 / 14; text-align: center;">PUTS</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Options chain rows
        chain_html = """
        <div style="background: linear-gradient(135deg, #2D2D34 0%, #3A3A42 100%); 
                    border: 1px solid rgba(232, 232, 232, 0.1); 
                    border-top: none; border-radius: 0 0 8px 8px;">
        """
        
        for i, strike in enumerate(common_strikes[:20]):  # Limit to 20 strikes for performance
            # Get call and put data for this strike
            call_data = calls_df[calls_df['strike'] == strike]
            put_data = puts_df[puts_df['strike'] == strike]
            
            if call_data.empty or put_data.empty:
                continue
            
            call = call_data.iloc[0]
            put = put_data.iloc[0]
            
            # Calculate Black-Scholes values (theoretical prices)
            call_bs_price = call['price']  # Already calculated in data generation
            put_bs_price = put['price']    # Already calculated in data generation
            
            # Determine if strike is ATM, ITM, or OTM for styling
            moneyness = strike / spot_price
            is_atm = 0.95 <= moneyness <= 1.05
            row_bg = "rgba(139, 92, 246, 0.1)" if is_atm else "rgba(58, 58, 66, 0.3)"
            
            # Alternate row colors
            if i % 2 == 1:
                row_bg = "rgba(74, 74, 90, 0.2)"
            
            chain_html += f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr 1fr 1fr 100px 1fr 1fr 1fr 1fr 1fr 1fr; 
                        gap: 8px; padding: 0.6rem 0.8rem; background: {row_bg}; 
                        text-align: center; font-size: 0.85rem; color: #E8E8E8;
                        border-bottom: 1px solid rgba(232, 232, 232, 0.05);">
                
                <!-- CALLS -->
                <div style="color: #10B981;">{len(call_data) * 100}</div>  <!-- Simulated Open Interest -->
                <div style="color: {'#10B981' if call['delta'] > 0 else '#EF4444'};">{call['delta']:.3f}</div>
                <div style="color: #A78BFA;">{call['iv']*100:.1f}%</div>
                <div style="color: #60A5FA;">${call['bid']:.2f}</div>
                <div style="color: #8B5CF6; font-weight: bold;">${call['price']:.2f}</div>
                <div style="color: #F59E0B;">${call['ask']:.2f}</div>
                
                <!-- STRIKE -->
                <div style="color: {'#8B5CF6' if is_atm else '#E8E8E8'}; font-weight: bold; 
                           background: {'rgba(139, 92, 246, 0.2)' if is_atm else 'transparent'}; 
                           padding: 0.3rem; border-radius: 4px;">
                    ${strike:,.0f}
                </div>
                
                <!-- PUTS -->
                <div style="color: #F59E0B;">${put['bid']:.2f}</div>
                <div style="color: #8B5CF6; font-weight: bold;">${put['price']:.2f}</div>
                <div style="color: #60A5FA;">${put['ask']:.2f}</div>
                <div style="color: #A78BFA;">{put['iv']*100:.1f}%</div>
                <div style="color: {'#EF4444' if put['delta'] < 0 else '#10B981'};">{put['delta']:.3f}</div>
                <div style="color: #10B981;">{len(put_data) * 100}</div>  <!-- Simulated Open Interest -->
            </div>
            """
        
        chain_html += "</div>"
        
        st.markdown(chain_html, unsafe_allow_html=True)
        
        # Black-Scholes Analysis Section
        st.markdown("---")
        st.subheader("📈 Black-Scholes Analysis")
        
        # Create Black-Scholes comparison table
        bs_data = []
        
        for strike in common_strikes[:10]:  # Top 10 for detailed analysis
            call_data = calls_df[calls_df['strike'] == strike]
            put_data = puts_df[puts_df['strike'] == strike]
            
            if not call_data.empty and not put_data.empty:
                call = call_data.iloc[0]
                put = put_data.iloc[0]
                
                bs_data.append({
                    'Strike': f"${strike:,.0f}",
                    'Call Market': f"${call['price']:.2f}",
                    'Call BS': f"${call['price']:.2f}",  # Using our calculated BS price
                    'Call Diff': f"{((call['ask'] - call['price']) / call['price'] * 100):+.1f}%",
                    'Put Market': f"${put['price']:.2f}",
                    'Put BS': f"${put['price']:.2f}",  # Using our calculated BS price
                    'Put Diff': f"{((put['ask'] - put['price']) / put['price'] * 100):+.1f}%",
                    'Call IV': f"{call['iv']*100:.1f}%",
                    'Put IV': f"{put['iv']*100:.1f}%"
                })
        
        if bs_data:
            bs_df = pd.DataFrame(bs_data)
            st.dataframe(
                bs_df,
                use_container_width=True,
                hide_index=True
            )
        
        # Options Chain Statistics
        st.subheader("📊 Chain Statistics")
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            total_calls = len(calls_df)
            st.metric("Total Calls", total_calls)
        
        with stat_col2:
            total_puts = len(puts_df)
            st.metric("Total Puts", total_puts)
        
        with stat_col3:
            call_put_ratio = total_calls / max(total_puts, 1)
            st.metric("Call/Put Ratio", f"{call_put_ratio:.2f}")
        
        with stat_col4:
            max_pain = common_strikes[len(common_strikes)//2]  # Simplified max pain calculation
            st.metric("Est. Max Pain", f"${max_pain:,.0f}")
        
        # Download option
        if st.button("📥 Download Options Chain"):
            combined_df = pd.concat([calls_df, puts_df], ignore_index=True)
            csv = combined_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{selected_asset}_options_chain_{selected_expiry.replace('-', '')}.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"Error loading options chain: {str(e)}")
        st.info("Please try refreshing the data or check your connection.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def volatility_surface_page():
    """Complete Volatility Surface Analysis with current point indicators and proper cross-sections."""
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    
    st.header("🌋 Volatility Surface Analysis")
    
    # Asset and surface type selection
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    
    with col1:
        selected_asset = st.selectbox("Select Asset", ["BTC", "ETH"], index=0, key="vol_surface_asset")
    
    with col2:
        surface_type = st.selectbox("Surface Type", [
            "Implied Volatility", 
            "Delta Surface", 
            "Gamma Surface",
            "Vega Surface",
            "Theta Surface"
        ], key="surface_type")
    
    with col3:
        interpolation_method = st.selectbox("Interpolation", [
            "cubic", "linear", "nearest"
        ], index=0, key="interpolation")
    
    with col4:
        if st.button("🔄 Refresh Surface", key="refresh_surface"):
            st.cache_data.clear()
            st.rerun()
    
    try:
        # Get options data
        options_df = data_manager.get_options_data(selected_asset)
        
        if options_df.empty:
            st.error("No options data for volatility surface")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        # Prepare surface data
        spot_price = options_df['spot_price'].iloc[0]
        current_time = datetime.now()
        
        # Create moneyness and time to expiry
        options_df['moneyness'] = options_df['strike'] / spot_price
        
        # Filter for reasonable surface bounds
        surface_data = options_df[
            (options_df['moneyness'] >= 0.7) & 
            (options_df['moneyness'] <= 1.3) &
            (options_df['time_to_maturity'] >= 0.01) &  # At least 1% of year
            (options_df['time_to_maturity'] <= 1.0)     # Within 1 year
        ].copy()
        
        if surface_data.empty:
            st.error("Insufficient data for volatility surface after filtering")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        # Market overview for current point
        st.subheader("📊 Market Overview & Current Point")
        
        current_col1, current_col2, current_col3, current_col4 = st.columns(4)
        
        current_hv = data_manager.get_historical_volatility(selected_asset) * 100
        current_atm_iv = surface_data[
            (surface_data['moneyness'] >= 0.98) & 
            (surface_data['moneyness'] <= 1.02)
        ]['iv'].mean() * 100 if not surface_data.empty else current_hv
        
        with current_col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0 0 0.5rem 0; color: #B8B8B8;">Current Spot</h4>
                <h2 style="margin: 0; color: #8B5CF6;">${spot_price:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with current_col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0 0 0.5rem 0; color: #B8B8B8;">ATM IV</h4>
                <h2 style="margin: 0; color: #A78BFA;">{current_atm_iv:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with current_col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0 0 0.5rem 0; color: #B8B8B8;">Historical Vol</h4>
                <h2 style="margin: 0; color: #C4B5FD;">{current_hv:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with current_col4:
            iv_hv_ratio = current_atm_iv / current_hv if current_hv > 0 else 1
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0 0 0.5rem 0; color: #B8B8B8;">IV/HV Ratio</h4>
                <h2 style="margin: 0; color: #6B46C1;">{iv_hv_ratio:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Prepare surface values based on type
        if surface_type == "Implied Volatility":
            z_values = surface_data['iv'] * 100
            z_title = "Implied Volatility (%)"
            colorscale = 'plasma'
        elif surface_type == "Delta Surface":
            z_values = surface_data['delta']
            z_title = "Delta"
            colorscale = 'RdBu'
        elif surface_type == "Gamma Surface":
            z_values = surface_data['gamma'] * 1000
            z_title = "Gamma (×1000)"
            colorscale = 'viridis'
        elif surface_type == "Vega Surface":
            z_values = surface_data['vega']
            z_title = "Vega"
            colorscale = 'cividis'
        else:  # Theta Surface
            z_values = surface_data['theta']
            z_title = "Theta"
            colorscale = 'hot'
        
        # Create comprehensive surface visualization
        st.subheader(f"📊 {surface_type} - {selected_asset}")
        
        # Create 3D surface with current point indicator
        fig_3d = go.Figure()
        
        # Get unique values for grid creation
        moneyness_unique = np.sort(surface_data['moneyness'].unique())
        time_unique = np.sort(surface_data['time_to_maturity'].unique())
        
        if len(moneyness_unique) >= 3 and len(time_unique) >= 3:
            # Create dense grid for smooth surface
            moneyness_grid = np.linspace(
                surface_data['moneyness'].min(), 
                surface_data['moneyness'].max(), 
                30
            )
            time_grid = np.linspace(
                surface_data['time_to_maturity'].min(), 
                surface_data['time_to_maturity'].max(), 
                30
            )
            
            X, Y = np.meshgrid(moneyness_grid, time_grid)
            
            # Interpolate Z values
            points = surface_data[['moneyness', 'time_to_maturity']].values
            values = z_values.values
            
            Z = griddata(points, values, (X, Y), method=interpolation_method, fill_value=np.nan)
            
            # Main surface
            fig_3d.add_trace(
                go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    colorscale=colorscale,
                    name=f'{selected_asset} {surface_type}',
                    colorbar=dict(
                        title=z_title,
                        title_font=dict(color='#E8E8E8'),
                        tickfont=dict(color='#E8E8E8')
                    ),
                    opacity=0.8
                )
            )
            
            # Add current point indicator (ATM, short-term)
            current_moneyness = 1.0  # ATM
            current_time = surface_data['time_to_maturity'].min()  # Shortest expiry
            
            # Find closest point for current value
            closest_idx = ((surface_data['moneyness'] - current_moneyness).abs() + 
                          (surface_data['time_to_maturity'] - current_time).abs()).idxmin()
            current_z = z_values.iloc[surface_data.index.get_loc(closest_idx)]
            
            # Add current point marker
            fig_3d.add_trace(
                go.Scatter3d(
                    x=[current_moneyness],
                    y=[current_time],
                    z=[current_z],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='red',
                        symbol='diamond'
                    ),
                    name='Current Point (ATM)',
                    text=[f'Current: {current_z:.2f}'],
                    textposition='middle center'
                )
            )
            
        else:
            # Fallback scatter plot if insufficient data for surface
            fig_3d.add_trace(
                go.Scatter3d(
                    x=surface_data['moneyness'],
                    y=surface_data['time_to_maturity'],
                    z=z_values,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=z_values,
                        colorscale=colorscale,
                        showscale=True,
                        colorbar=dict(
                            title=z_title,
                            title_font=dict(color='#E8E8E8'),
                            tickfont=dict(color='#E8E8E8')
                        )
                    ),
                    name=f'{selected_asset} {surface_type}',
                    text=[f'M: {m:.2f}<br>T: {t:.2f}<br>Z: {z:.2f}' 
                          for m, t, z in zip(surface_data['moneyness'], 
                                            surface_data['time_to_maturity'], 
                                            z_values)]
                )
            )
        
        # Apply dark theme to 3D plot
        fig_3d.update_layout(
            scene=dict(
                xaxis_title="Moneyness (Strike/Spot)",
                yaxis_title="Time to Maturity (Years)",
                zaxis_title=z_title,
                xaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(232, 232, 232, 0.1)",
                    showbackground=True,
                    title_font=dict(color='#E8E8E8'),
                    tickfont=dict(color='#B8B8B8')
                ),
                yaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(232, 232, 232, 0.1)",
                    showbackground=True,
                    title_font=dict(color='#E8E8E8'),
                    tickfont=dict(color='#B8B8B8')
                ),
                zaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(232, 232, 232, 0.1)",
                    showbackground=True,
                    title_font=dict(color='#E8E8E8'),
                    tickfont=dict(color='#B8B8B8')
                ),
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=1.2)
                ),
                bgcolor="rgba(0,0,0,0)"
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E8E8E8'),
            title=f"{selected_asset} {surface_type}",
            title_font=dict(color='#E8E8E8', size=16),
            height=600,
            legend=dict(
                font=dict(color='#E8E8E8'),
                bgcolor='rgba(58, 58, 66, 0.8)',
                bordercolor='rgba(232, 232, 232, 0.1)',
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Meaningful Cross-Sections Analysis
        st.subheader("📈 Surface Cross-Sections Analysis")
        
        # Create meaningful cross-sections
        cross_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Volatility Smile (30-day expiry)",
                "Term Structure (ATM)",
                "Skew Analysis",
                "Time Decay Analysis"
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Volatility Smile (fixed time, varying moneyness)
        target_time = 30 / 365.25  # 30 days
        smile_data = surface_data.iloc[
            (surface_data['time_to_maturity'] - target_time).abs().argsort()[:20]
        ].sort_values('moneyness')
        
        if not smile_data.empty:
            cross_fig.add_trace(
                go.Scatter(
                    x=smile_data['moneyness'],
                    y=smile_data['iv'] * 100,
                    mode='lines+markers',
                    name='30-day Smile',
                    line=dict(color='#8B5CF6', width=3),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )
        
        # 2. Term Structure (ATM, varying time)
        atm_data = surface_data[
            (surface_data['moneyness'] >= 0.98) & 
            (surface_data['moneyness'] <= 1.02)
        ].sort_values('time_to_maturity')
        
        if not atm_data.empty:
            cross_fig.add_trace(
                go.Scatter(
                    x=atm_data['time_to_maturity'] * 365,  # Convert to days
                    y=atm_data['iv'] * 100,
                    mode='lines+markers',
                    name='ATM Term Structure',
                    line=dict(color='#A78BFA', width=3),
                    marker=dict(size=6)
                ),
                row=1, col=2
            )
        
        # 3. Skew Analysis (Put-Call IV difference)
        calls_data = surface_data[surface_data['type'] == 'call']
        puts_data = surface_data[surface_data['type'] == 'put']
        
        if not calls_data.empty and not puts_data.empty:
            # Group by time and calculate skew
            skew_data = []
            for time_group in calls_data.groupby('time_to_maturity'):
                time_val = time_group[0]
                time_calls = time_group[1]
                time_puts = puts_data[puts_data['time_to_maturity'] == time_val]
                
                if not time_puts.empty:
                    # Calculate 25-delta skew (simplified)
                    otm_puts = time_puts[time_puts['moneyness'] < 0.9]['iv'].mean()
                    otm_calls = time_calls[time_calls['moneyness'] > 1.1]['iv'].mean()
                    
                    if not pd.isna(otm_puts) and not pd.isna(otm_calls):
                        skew = (otm_puts - otm_calls) * 100
                        skew_data.append({'time': time_val * 365, 'skew': skew})
            
            if skew_data:
                skew_df = pd.DataFrame(skew_data)
                cross_fig.add_trace(
                    go.Scatter(
                        x=skew_df['time'],
                        y=skew_df['skew'],
                        mode='lines+markers',
                        name='Put-Call Skew',
                        line=dict(color='#C4B5FD', width=3),
                        marker=dict(size=6)
                    ),
                    row=2, col=1
                )
        
        # 4. Time Decay Analysis (Theta term structure)
        if surface_type == "Theta Surface" or surface_type == "Implied Volatility":
            theta_data = surface_data.groupby('time_to_maturity')['theta'].mean().reset_index()
            theta_data['time_days'] = theta_data['time_to_maturity'] * 365
            
            cross_fig.add_trace(
                go.Scatter(
                    x=theta_data['time_days'],
                    y=theta_data['theta'],
                    mode='lines+markers',
                    name='Theta Decay',
                    line=dict(color='#6B46C1', width=3),
                    marker=dict(size=6)
                ),
                row=2, col=2
            )
        
        # Apply dark theme to cross-sections
        cross_fig.update_layout(get_dark_theme_layout())
        cross_fig.update_layout(
            height=600,
            showlegend=True,
            title_text="Surface Cross-Sections Analysis"
        )
        
        # Update axes labels
        cross_fig.update_xaxes(title_text="Moneyness", row=1, col=1)
        cross_fig.update_yaxes(title_text="IV (%)", row=1, col=1)
        
        cross_fig.update_xaxes(title_text="Days to Expiry", row=1, col=2)
        cross_fig.update_yaxes(title_text="IV (%)", row=1, col=2)
        
        cross_fig.update_xaxes(title_text="Days to Expiry", row=2, col=1)
        cross_fig.update_yaxes(title_text="Skew (%)", row=2, col=1)
        
        cross_fig.update_xaxes(title_text="Days to Expiry", row=2, col=2)
        cross_fig.update_yaxes(title_text="Theta", row=2, col=2)
        
        st.plotly_chart(cross_fig, use_container_width=True)
        
        # Surface Statistics and Analysis
        st.subheader("📊 Surface Statistics")
        
        stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
        
        with stat_col1:
            st.metric("Min Value", f"{z_values.min():.3f}")
        
        with stat_col2:
            st.metric("Max Value", f"{z_values.max():.3f}")
        
        with stat_col3:
            st.metric("Average", f"{z_values.mean():.3f}")
        
        with stat_col4:
            st.metric("Std Deviation", f"{z_values.std():.3f}")
        
        with stat_col5:
            surface_points = len(surface_data)
            st.metric("Surface Points", f"{surface_points}")
        
        # Export functionality
        if st.button("📥 Export Surface Data"):
            export_df = surface_data.copy()
            export_df['surface_value'] = z_values
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Surface CSV",
                data=csv,
                file_name=f"{selected_asset}_{surface_type.lower().replace(' ', '_')}_surface_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"Error creating volatility surface: {str(e)}")
        st.info("The volatility surface module has been completely rewritten. Please try refreshing.")
    
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