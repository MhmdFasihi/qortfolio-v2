# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Qortfolio V2 - Main Interactive Dashboard
Streamlit-based web interface for comprehensive options analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import sys
import os
from pathlib import Path

# Add src to Python path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

# Import Qortfolio V2 modules
try:
    from data import get_data_manager, collect_market_data, get_spot_price
    from models.options.black_scholes import price_option, calculate_greeks
    from models.options.greeks_calculator import GreeksCalculator, analyze_portfolio_risk
    from analytics.pnl_simulator import TaylorPnLSimulator, MarketScenario
    from analytics.volatility_surface import VolatilitySurfaceAnalyzer, analyze_options_volatility
    from core.config import get_config
    from core.logging import setup_logging, get_logger
except ImportError as e:
    st.error(f"Failed to import Qortfolio V2 modules: {e}")
    st.error("Please ensure all module files are created and properly structured.")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="Qortfolio V2 - Options Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logging
setup_logging({"level": "INFO", "console": False, "file_enabled": False})
logger = get_logger("dashboard")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-warning {
        background-color: #ffe6e6;
        padding: 1rem;
        border-left: 4px solid #ff4444;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e6ffe6;
        padding: 1rem;
        border-left: 4px solid #44ff44;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class QortfolioDashboard:
    """Main dashboard class for Qortfolio V2."""
    
    def __init__(self):
        """Initialize dashboard components."""
        self.config = get_config()
        self.data_manager = get_data_manager()
        self.greeks_calculator = GreeksCalculator()
        self.pnl_simulator = TaylorPnLSimulator()
        self.vol_analyzer = VolatilitySurfaceAnalyzer()
        
        # Initialize session state
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = []
        if 'market_data_cache' not in st.session_state:
            st.session_state.market_data_cache = {}
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
    
    def run(self):
        """Run the main dashboard."""
        
        # Header
        st.markdown('<h1 class="main-header">📊 Qortfolio V2 - Options Analytics Platform</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Select Analysis", 
            ["Market Overview", "Options Chain", "Portfolio Analytics", 
             "PnL Simulation", "Volatility Surface", "Risk Management", "System Status"]
        )
        
        # Main content based on selection
        if page == "Market Overview":
            self.market_overview_page()
        elif page == "Options Chain":
            self.options_chain_page()
        elif page == "Portfolio Analytics":
            self.portfolio_analytics_page()
        elif page == "PnL Simulation":
            self.pnl_simulation_page()
        elif page == "Volatility Surface":
            self.volatility_surface_page()
        elif page == "Risk Management":
            self.risk_management_page()
        elif page == "System Status":
            self.system_status_page()
    
    def market_overview_page(self):
        """Market overview page with current prices and key metrics."""
        st.header("📈 Market Overview")
        
        # Symbol selection
        enabled_cryptos = self.config.enabled_cryptocurrencies
        symbols = [crypto.symbol for crypto in enabled_cryptos]
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            selected_symbols = st.multiselect(
                "Select Cryptocurrencies",
                symbols,
                default=["BTC", "ETH"] if len(symbols) >= 2 else symbols[:1]
            )
        
        with col2:
            auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)
        
        with col3:
            if st.button("🔄 Refresh Data"):
                st.session_state.market_data_cache = {}
        
        if not selected_symbols:
            st.warning("Please select at least one cryptocurrency.")
            return
        
        # Auto refresh logic
        if auto_refresh:
            time.sleep(1)
            st.rerun()
        
        # Get market data
        with st.spinner("Loading market data..."):
            try:
                market_data = collect_market_data(
                    symbols=selected_symbols,
                    include_options=True,
                    include_historical=True,
                    period="30d",
                    interval="1d"
                )
                
                st.session_state.last_update = datetime.now()
                
            except Exception as e:
                st.error(f"Failed to load market data: {e}")
                return
        
        # Display current prices
        st.subheader("💰 Current Prices")
        
        price_cols = st.columns(len(selected_symbols))
        
        for i, symbol in enumerate(selected_symbols):
            with price_cols[i]:
                try:
                    current_price = get_spot_price(symbol)
                    
                    if current_price:
                        st.metric(
                            label=f"{symbol} Price",
                            value=f"${current_price:,.2f}",
                            delta=None  # TODO: Add price change calculation
                        )
                    else:
                        st.error(f"No price data for {symbol}")
                        
                except Exception as e:
                    st.error(f"Error loading {symbol}: {e}")
        
        # Options availability
        st.subheader("🎯 Options Availability")
        
        deribit_currencies = self.config.deribit_currencies
        options_available = [s for s in selected_symbols if s in deribit_currencies]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Options Available:** {', '.join(options_available) if options_available else 'None'}")
        
        with col2:
            st.info(f"**Spot Data Only:** {', '.join([s for s in selected_symbols if s not in deribit_currencies])}")
        
        # Historical price chart
        if market_data.historical_data is not None and not market_data.historical_data.empty:
            st.subheader("📊 Historical Prices (30 Days)")
            
            fig = go.Figure()
            
            for symbol in selected_symbols:
                symbol_data = market_data.historical_data[
                    market_data.historical_data['Symbol'] == symbol
                ]
                
                if not symbol_data.empty and 'Close' in symbol_data.columns:
                    fig.add_trace(go.Scatter(
                        x=symbol_data['Timestamp'] if 'Timestamp' in symbol_data.columns else symbol_data.index,
                        y=symbol_data['Close'],
                        mode='lines',
                        name=symbol,
                        line=dict(width=2)
                    ))
            
            fig.update_layout(
                title="Price History",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Market statistics
        if market_data.historical_data is not None:
            st.subheader("📈 Market Statistics")
            
            stats_cols = st.columns(len(selected_symbols))
            
            for i, symbol in enumerate(selected_symbols):
                with stats_cols[i]:
                    symbol_data = market_data.historical_data[
                        market_data.historical_data['Symbol'] == symbol
                    ]
                    
                    if not symbol_data.empty and 'Close' in symbol_data.columns:
                        prices = symbol_data['Close']
                        returns = prices.pct_change().dropna()
                        
                        volatility_annual = returns.std() * np.sqrt(365) * 100
                        
                        st.markdown(f"**{symbol} Stats:**")
                        st.write(f"30D High: ${prices.max():,.2f}")
                        st.write(f"30D Low: ${prices.min():,.2f}")
                        st.write(f"30D Volatility: {volatility_annual:.1f}%")
        
        # Last update info
        if st.session_state.last_update:
            st.caption(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def options_chain_page(self):
        """Options chain analysis page."""
        st.header("🎯 Options Chain Analysis")
        
        # Symbol selection for options
        deribit_currencies = self.config.deribit_currencies
        
        if not deribit_currencies:
            st.error("No cryptocurrencies with options available in configuration.")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.selectbox("Select Cryptocurrency", deribit_currencies, index=0)
        
        with col2:
            option_type_filter = st.selectbox("Option Type", ["All", "Calls", "Puts"])
        
        with col3:
            if st.button("🔄 Load Options Chain"):
                st.session_state.options_data_cache = {}
        
        # Load options data
        with st.spinner(f"Loading {symbol} options chain..."):
            try:
                market_data = collect_market_data(
                    symbols=[symbol],
                    include_options=True,
                    include_historical=False
                )
                
                if market_data.options_data is None or market_data.options_data.empty:
                    st.warning(f"No options data available for {symbol}")
                    return
                
                options_df = market_data.options_data
                current_spot = get_spot_price(symbol)
                
            except Exception as e:
                st.error(f"Failed to load options data: {e}")
                return
        
        # Filter options by type
        if option_type_filter == "Calls":
            options_df = options_df[options_df['option_type'].str.lower() == 'call']
        elif option_type_filter == "Puts":
            options_df = options_df[options_df['option_type'].str.lower() == 'put']
        
        # Display current spot price
        st.metric(f"{symbol} Spot Price", f"${current_spot:,.2f}" if current_spot else "N/A")
        
        # Options chain table
        st.subheader("📋 Options Chain")
        
        if not options_df.empty:
            # Prepare display columns
            display_columns = ['instrument_name', 'option_type', 'strike', 'mark_price', 
                             'bid_price', 'ask_price', 'TimeToMaturity']
            
            available_columns = [col for col in display_columns if col in options_df.columns]
            
            display_df = options_df[available_columns].copy()
            
            # Format for display
            if 'TimeToMaturity' in display_df.columns:
                display_df['Days to Expiry'] = (display_df['TimeToMaturity'] * 365.25).round(0)
            
            if 'strike' in display_df.columns and current_spot:
                display_df['Moneyness'] = (display_df['strike'] / current_spot).round(3)
            
            # Sort by strike and expiry
            if 'strike' in display_df.columns:
                display_df = display_df.sort_values(['strike'])
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Options analytics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Strike Distribution")
                
                if 'strike' in options_df.columns:
                    fig = px.histogram(
                        options_df, 
                        x='strike',
                        color='option_type',
                        title='Options by Strike Price',
                        nbins=20
                    )
                    
                    # Add vertical line for current spot
                    if current_spot:
                        fig.add_vline(x=current_spot, line_dash="dash", 
                                    annotation_text="Current Spot")
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("⏰ Expiry Distribution")
                
                if 'TimeToMaturity' in options_df.columns:
                    expiry_days = options_df['TimeToMaturity'] * 365.25
                    
                    fig = px.histogram(
                        x=expiry_days,
                        title='Options by Days to Expiry',
                        nbins=15
                    )
                    
                    fig.update_xaxis(title="Days to Expiry")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Quick options pricing
            st.subheader("🧮 Quick Options Pricing")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                quick_strike = st.number_input("Strike Price", 
                                             value=int(current_spot) if current_spot else 50000,
                                             step=1000)
            
            with col2:
                quick_days = st.number_input("Days to Expiry", value=30, min_value=1, max_value=365)
            
            with col3:
                quick_vol = st.number_input("Implied Volatility (%)", value=80.0, min_value=1.0, max_value=500.0)
            
            with col4:
                quick_type = st.selectbox("Type", ["call", "put"])
            
            if current_spot and st.button("Calculate Option Price"):
                try:
                    tte = quick_days / 365.25
                    vol = quick_vol / 100
                    
                    option_price = price_option(current_spot, quick_strike, tte, vol, quick_type)
                    greeks = calculate_greeks(current_spot, quick_strike, tte, vol, quick_type)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Option Price", f"${option_price:.2f}")
                        st.metric("Delta", f"{greeks['delta']:.4f}")
                    
                    with col2:
                        st.metric("Gamma", f"{greeks['gamma']:.6f}")
                        st.metric("Theta", f"{greeks['theta']:.4f}")
                    
                    with col3:
                        st.metric("Vega", f"{greeks['vega']:.4f}")
                        st.metric("Intrinsic", f"${greeks['intrinsic']:.2f}")
                
                except Exception as e:
                    st.error(f"Calculation failed: {e}")
        
        else:
            st.warning("No options data available for the selected filters.")
    
    def portfolio_analytics_page(self):
        """Portfolio management and analytics page."""
        st.header("💼 Portfolio Analytics")
        
        # Portfolio management
        st.subheader("🔧 Portfolio Management")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("Add positions to your portfolio for comprehensive analysis.")
        
        with col2:
            if st.button("Clear Portfolio"):
                st.session_state.portfolio = []
                st.success("Portfolio cleared!")
        
        # Add new position
        with st.expander("➕ Add New Position", expanded=len(st.session_state.portfolio) == 0):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                pos_symbol = st.selectbox("Symbol", self.config.deribit_currencies)
                pos_quantity = st.number_input("Quantity", value=1, step=1, 
                                             help="Positive for long, negative for short")
            
            with col2:
                pos_spot = st.number_input("Spot Price", value=50000.0, step=100.0)
                pos_strike = st.number_input("Strike Price", value=52000.0, step=100.0)
            
            with col3:
                pos_days = st.number_input("Days to Expiry", value=30, min_value=1, max_value=365)
                pos_vol = st.number_input("Volatility (%)", value=80.0, min_value=1.0, max_value=300.0)
            
            with col4:
                pos_type = st.selectbox("Option Type", ["call", "put"])
                
                if st.button("Add Position"):
                    position = {
                        'symbol': pos_symbol,
                        'quantity': pos_quantity,
                        'spot_price': pos_spot,
                        'strike_price': pos_strike,
                        'time_to_maturity': pos_days / 365.25,
                        'volatility': pos_vol / 100,
                        'option_type': pos_type,
                        'risk_free_rate': 0.05
                    }
                    
                    st.session_state.portfolio.append(position)
                    st.success(f"Added {pos_quantity} {pos_symbol} {pos_strike} {pos_type}")
                    st.rerun()
        
        # Display current portfolio
        if st.session_state.portfolio:
            st.subheader("📊 Current Portfolio")
            
            # Portfolio table
            portfolio_df = pd.DataFrame(st.session_state.portfolio)
            portfolio_df['Days to Expiry'] = (portfolio_df['time_to_maturity'] * 365.25).round(0)
            portfolio_df['Volatility %'] = (portfolio_df['volatility'] * 100).round(1)
            
            display_cols = ['symbol', 'quantity', 'option_type', 'strike_price', 
                          'Days to Expiry', 'Volatility %']
            st.dataframe(portfolio_df[display_cols], use_container_width=True)
            
            # Portfolio analytics
            try:
                portfolio_greeks = self.greeks_calculator.calculate_portfolio_greeks(st.session_state.portfolio)
                
                st.subheader("📈 Portfolio Greeks")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Delta", f"{portfolio_greeks.delta:.4f}")
                
                with col2:
                    st.metric("Gamma", f"{portfolio_greeks.gamma:.6f}")
                
                with col3:
                    st.metric("Theta", f"{portfolio_greeks.theta:.4f}")
                
                with col4:
                    st.metric("Vega", f"{portfolio_greeks.vega:.4f}")
                
                with col5:
                    st.metric("Portfolio Value", f"${portfolio_greeks.portfolio_value:.2f}")
                
                # Risk metrics
                st.subheader("⚠️ Risk Analysis")
                
                if len(st.session_state.portfolio) > 0:
                    avg_spot = np.mean([p['spot_price'] for p in st.session_state.portfolio])
                    risk_metrics = self.greeks_calculator.calculate_risk_metrics(portfolio_greeks, avg_spot)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Max Daily Theta Loss", f"${risk_metrics.max_loss_1_day:.2f}")
                    
                    with col2:
                        st.metric("Risk from 1% Move", f"${risk_metrics.max_loss_1_percent_move:.2f}")
                    
                    with col3:
                        st.metric("Gamma Risk (5% move)", f"${risk_metrics.gamma_risk:.2f}")
                
            except Exception as e:
                st.error(f"Portfolio analysis failed: {e}")
        
        else:
            st.info("Portfolio is empty. Add some positions to see analytics.")
    
    def pnl_simulation_page(self):
        """PnL simulation using Taylor expansion."""
        st.header("💰 PnL Simulation (Taylor Expansion)")
        
        if not st.session_state.portfolio:
            st.warning("Please add positions to your portfolio first.")
            return
        
        st.subheader("🎯 Taylor Expansion Formula")
        st.latex(r"\Delta C \approx \delta \Delta S + \frac{1}{2}\gamma (\Delta S)^2 + \theta \Delta t + \nu \Delta \sigma + \rho \Delta r")
        
        # Simulation parameters
        st.subheader("⚙️ Simulation Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            spot_moves = st.multiselect(
                "Spot Price Changes (%)",
                [-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20],
                default=[-10, -5, 0, 5, 10]
            )
        
        with col2:
            vol_changes = st.multiselect(
                "Volatility Changes (%)",
                [-30, -20, -10, -5, 0, 5, 10, 20, 30],
                default=[-20, -10, 0, 10, 20]
            )
        
        with col3:
            time_horizons = st.multiselect(
                "Time Horizons (days)",
                [1, 3, 7, 14, 30],
                default=[1, 7, 30]
            )
        
        # Run simulation
        if st.button("🚀 Run PnL Simulation"):
            with st.spinner("Running Taylor expansion PnL simulation..."):
                try:
                    # Create scenarios
                    scenarios = []
                    
                    avg_spot = np.mean([p['spot_price'] for p in st.session_state.portfolio])
                    
                    scenario_id = 1
                    for spot_pct in spot_moves:
                        for vol_pct in vol_changes:
                            for days in time_horizons:
                                spot_change = avg_spot * (spot_pct / 100)
                                vol_change = vol_pct / 100
                                
                                scenario = MarketScenario(
                                    spot_change=spot_change,
                                    time_change=days,
                                    volatility_change=vol_change,
                                    rate_change=0.0,
                                    scenario_name=f"Scenario_{scenario_id}"
                                )
                                scenarios.append(scenario)
                                scenario_id += 1
                    
                    # Run simulation
                    pnl_results = self.pnl_simulator.simulate_pnl(
                        st.session_state.portfolio,
                        scenarios,
                        include_second_order=True,
                        validate_with_bs=True
                    )
                    
                    # Display results
                    st.subheader("📊 Simulation Results")
                    
                    if pnl_results:
                        # Summary statistics
                        pnls = [r.taylor_total_pnl for r in pnl_results]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Max Gain", f"${max(pnls):.2f}")
                        
                        with col2:
                            st.metric("Max Loss", f"${min(pnls):.2f}")
                        
                        with col3:
                            st.metric("Average P&L", f"${np.mean(pnls):.2f}")
                        
                        with col4:
                            st.metric("Scenarios", len(pnl_results))
                        
                        # P&L distribution
                        st.subheader("📈 P&L Distribution")
                        
                        fig = px.histogram(
                            x=pnls,
                            nbins=30,
                            title="Taylor Expansion P&L Distribution"
                        )
                        fig.update_xaxis(title="P&L ($)")
                        fig.update_yaxis(title="Frequency")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Accuracy analysis
                        if any(r.actual_pnl is not None for r in pnl_results):
                            st.subheader("🎯 Taylor Expansion Accuracy")
                            
                            accuracy_analysis = self.pnl_simulator.analyze_taylor_accuracy(pnl_results)
                            
                            if "error_statistics" in accuracy_analysis:
                                stats = accuracy_analysis["error_statistics"]
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Mean Absolute Error", f"${stats['mean_absolute_error']:.2f}")
                                
                                with col2:
                                    st.metric("Max Absolute Error", f"${stats['max_absolute_error']:.2f}")
                                
                                with col3:
                                    st.metric("Mean Relative Error", f"{stats['mean_relative_error']:.1%}")
                        
                        # Detailed results table
                        st.subheader("📋 Detailed Results")
                        
                        results_data = []
                        for r in pnl_results[:20]:  # Show first 20 results
                            results_data.append({
                                'Scenario': r.scenario.scenario_name,
                                'Spot Change $': r.scenario.spot_change,
                                'Vol Change %': r.scenario.volatility_change * 100,
                                'Days': r.scenario.time_change,
                                'Taylor P&L $': r.taylor_total_pnl,
                                'Delta P&L $': r.delta_pnl,
                                'Gamma P&L $': r.gamma_pnl,
                                'Theta P&L $': r.theta_pnl,
                                'Vega P&L $': r.vega_pnl,
                                'Actual P&L $': r.actual_pnl,
                                'Error $': r.taylor_error
                            })
                        
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True)
                        
                    else:
                        st.error("No simulation results generated.")
                
                except Exception as e:
                    st.error(f"Simulation failed: {e}")
    
    def volatility_surface_page(self):
        """Volatility surface analysis page."""
        st.header("🌊 Volatility Surface Analysis")
        
        # Symbol selection
        deribit_currencies = self.config.deribit_currencies
        
        if not deribit_currencies:
            st.error("No cryptocurrencies with options available.")
            return
        
        symbol = st.selectbox("Select Cryptocurrency", deribit_currencies, index=0)
        
        if st.button("📊 Build Volatility Surface"):
            with st.spinner(f"Building volatility surface for {symbol}..."):
                try:
                    # Get options data
                    market_data = collect_market_data(
                        symbols=[symbol],
                        include_options=True,
                        include_historical=False
                    )
                    
                    if market_data.options_data is None or market_data.options_data.empty:
                        st.error(f"No options data available for {symbol}")
                        return
                    
                    current_spot = get_spot_price(symbol)
                    if not current_spot:
                        st.error(f"Could not get spot price for {symbol}")
                        return
                    
                    # Build volatility surface
                    surface_data = analyze_options_volatility(market_data.options_data, current_spot)
                    
                    # Display surface metrics
                    st.subheader("📊 Surface Metrics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Surface Points", len(surface_data.surface_points))
                    
                    with col2:
                        st.metric("Spot Price", f"${surface_data.spot_price:,.2f}")
                    
                    with col3:
                        atm_vol = surface_data.atm_volatility
                        st.metric("ATM Volatility", f"{atm_vol:.1%}" if atm_vol else "N/A")
                    
                    with col4:
                        st.metric("Currency", surface_data.currency)
                    
                    # 3D Volatility Surface
                    st.subheader("🏔️ 3D Volatility Surface")
                    
                    try:
                        fig_3d = self.vol_analyzer.plot_volatility_surface_3d(surface_data)
                        st.plotly_chart(fig_3d, use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to create 3D surface: {e}")
                    
                    # Volatility Smile
                    st.subheader("😊 Volatility Smile")
                    
                    try:
                        fig_smile = self.vol_analyzer.plot_volatility_smile(surface_data)
                        st.plotly_chart(fig_smile, use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to create volatility smile: {e}")
                    
                    # Term Structure
                    st.subheader("📈 ATM Term Structure")
                    
                    try:
                        fig_term = self.vol_analyzer.plot_term_structure(surface_data)
                        st.plotly_chart(fig_term, use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to create term structure: {e}")
                    
                    # Skew analysis
                    st.subheader("📐 Volatility Skew Analysis")
                    
                    skew_analysis = self.vol_analyzer.analyze_volatility_skew(surface_data, 30)
                    
                    if "error" not in skew_analysis:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if skew_analysis.get("atm_volatility"):
                                st.metric("30D ATM Vol", f"{skew_analysis['atm_volatility']:.1f}%")
                            
                            if skew_analysis.get("risk_reversal"):
                                st.metric("Risk Reversal", f"{skew_analysis['risk_reversal']:.1f}%")
                        
                        with col2:
                            if skew_analysis.get("butterfly"):
                                st.metric("Butterfly", f"{skew_analysis['butterfly']:.1f}%")
                            
                            if skew_analysis.get("skew_slope"):
                                st.metric("Skew Slope", f"{skew_analysis['skew_slope']:.2f}")
                    
                except Exception as e:
                    st.error(f"Volatility surface analysis failed: {e}")
    
    def risk_management_page(self):
        """Risk management and monitoring page."""
        st.header("⚠️ Risk Management")
        
        if not st.session_state.portfolio:
            st.warning("Please add positions to your portfolio first.")
            return
        
        # Portfolio risk analysis
        try:
            portfolio_greeks = self.greeks_calculator.calculate_portfolio_greeks(st.session_state.portfolio)
            avg_spot = np.mean([p['spot_price'] for p in st.session_state.portfolio])
            risk_metrics = self.greeks_calculator.calculate_risk_metrics(portfolio_greeks, avg_spot)
            
            # Risk dashboard
            st.subheader("🎛️ Risk Dashboard")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Time Decay Risk**")
                daily_theta = abs(portfolio_greeks.theta)
                st.metric("Daily Theta Decay", f"${daily_theta:.2f}")
                
                if daily_theta > 100:
                    st.markdown('<div class="risk-warning">⚠️ High theta exposure</div>', 
                              unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Delta Risk**")
                delta_exposure = abs(portfolio_greeks.delta * avg_spot)
                st.metric("Delta Exposure", f"${delta_exposure:.2f}")
                
                if abs(portfolio_greeks.delta) > 0.5:
                    st.markdown('<div class="risk-warning">⚠️ High directional exposure</div>', 
                              unsafe_allow_html=True)
            
            with col3:
                st.markdown("**Gamma Risk**")
                st.metric("Gamma Risk (5% move)", f"${risk_metrics.gamma_risk:.2f}")
                
                if risk_metrics.gamma_risk > 500:
                    st.markdown('<div class="risk-warning">⚠️ High gamma exposure</div>', 
                              unsafe_allow_html=True)
            
            # Scenario stress testing
            st.subheader("🔥 Stress Testing")
            
            stress_scenarios = [
                ("Market Crash (-20%)", -0.2, 0, 1),
                ("Volatility Spike", 0, 0.3, 1),
                ("Time Decay (7 days)", 0, 0, 7),
                ("Combined Stress", -0.15, 0.2, 3)
            ]
            
            stress_results = []
            
            for name, spot_pct, vol_change, days in stress_scenarios:
                spot_change = avg_spot * spot_pct
                
                pnl_attr = self.greeks_calculator.calculate_pnl_attribution(
                    portfolio_greeks,
                    spot_change=spot_change,
                    time_decay=days,
                    volatility_change=vol_change
                )
                
                stress_results.append({
                    'Scenario': name,
                    'Total P&L': pnl_attr['total_explained'],
                    'Delta P&L': pnl_attr['delta_pnl'],
                    'Gamma P&L': pnl_attr['gamma_pnl'],
                    'Theta P&L': pnl_attr['theta_pnl'],
                    'Vega P&L': pnl_attr['vega_pnl']
                })
            
            stress_df = pd.DataFrame(stress_results)
            
            # Color code by P&L
            def highlight_pnl(val):
                if isinstance(val, (int, float)):
                    if val < -100:
                        return 'background-color: #ffcccc'
                    elif val > 100:
                        return 'background-color: #ccffcc'
                return ''
            
            styled_df = stress_df.style.applymap(highlight_pnl, subset=['Total P&L'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Risk limits and alerts
            st.subheader("🚨 Risk Limits & Alerts")
            
            col1, col2 = st.columns(2)
            
            with col1:
                max_delta_limit = st.number_input("Max Delta Exposure ($)", value=10000, step=1000)
                max_theta_limit = st.number_input("Max Daily Theta ($)", value=500, step=50)
            
            with col2:
                max_gamma_limit = st.number_input("Max Gamma Risk ($)", value=2000, step=500)
                
                if st.button("Check Risk Limits"):
                    violations = []
                    
                    if delta_exposure > max_delta_limit:
                        violations.append(f"Delta exposure ${delta_exposure:.0f} exceeds limit ${max_delta_limit}")
                    
                    if daily_theta > max_theta_limit:
                        violations.append(f"Daily theta ${daily_theta:.0f} exceeds limit ${max_theta_limit}")
                    
                    if risk_metrics.gamma_risk > max_gamma_limit:
                        violations.append(f"Gamma risk ${risk_metrics.gamma_risk:.0f} exceeds limit ${max_gamma_limit}")
                    
                    if violations:
                        for violation in violations:
                            st.error(f"⚠️ {violation}")
                    else:
                        st.success("✅ All risk limits within bounds")
        
        except Exception as e:
            st.error(f"Risk analysis failed: {e}")
    
    def system_status_page(self):
        """System status and diagnostics page."""
        st.header("🔧 System Status")
        
        # Configuration status
        st.subheader("⚙️ Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Enabled Cryptocurrencies:** {len(self.config.enabled_cryptocurrencies)}")
            
            enabled_list = [crypto.symbol for crypto in self.config.enabled_cryptocurrencies]
            st.write(", ".join(enabled_list))
        
        with col2:
            st.info(f"**Deribit Currencies:** {len(self.config.deribit_currencies)}")
            st.write(", ".join(self.config.deribit_currencies))
        
        # Data collection status
        st.subheader("📊 Data Collection Status")
        
        try:
            stats = self.data_manager.get_collector_stats()
            
            for collector_name, collector_stats in stats.items():
                st.markdown(f"**{collector_name.replace('_', ' ').title()}:**")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Requests", collector_stats.get('total_requests', 0))
                
                with col2:
                    st.metric("Successful", collector_stats.get('successful_requests', 0))
                
                with col3:
                    success_rate = collector_stats.get('success_rate', 0)
                    st.metric("Success Rate", f"{success_rate:.1%}")
                
                with col4:
                    avg_time = collector_stats.get('average_response_time', 0)
                    st.metric("Avg Response", f"{avg_time:.2f}s")
        
        except Exception as e:
            st.error(f"Failed to get collector stats: {e}")
        
        # System health check
        st.subheader("🩺 System Health Check")
        
        if st.button("🔍 Run Health Check"):
            with st.spinner("Running system health check..."):
                health_results = []
                
                # Test configuration
                try:
                    config_summary = self.config.get_config_summary()
                    health_results.append(("Configuration", "✅ OK", f"Loaded {len(config_summary['loaded_files'])} files"))
                except Exception as e:
                    health_results.append(("Configuration", "❌ Error", str(e)))
                
                # Test data collection
                try:
                    test_price = get_spot_price("BTC")
                    if test_price:
                        health_results.append(("Crypto Data", "✅ OK", f"BTC: ${test_price:,.2f}"))
                    else:
                        health_results.append(("Crypto Data", "⚠️ Warning", "No price data"))
                except Exception as e:
                    health_results.append(("Crypto Data", "❌ Error", str(e)))
                
                # Test options data
                try:
                    if self.config.deribit_currencies:
                        market_data = collect_market_data(
                            symbols=[self.config.deribit_currencies[0]],
                            include_options=True,
                            include_historical=False
                        )
                        
                        if market_data.options_data is not None and not market_data.options_data.empty:
                            health_results.append(("Options Data", "✅ OK", f"{len(market_data.options_data)} options"))
                        else:
                            health_results.append(("Options Data", "⚠️ Warning", "No options data"))
                    else:
                        health_results.append(("Options Data", "⚠️ Warning", "No supported currencies"))
                        
                except Exception as e:
                    health_results.append(("Options Data", "❌ Error", str(e)))
                
                # Test financial calculations
                try:
                    test_price = price_option(50000, 52000, 30/365.25, 0.8, 'call')
                    health_results.append(("Financial Calcs", "✅ OK", f"Test option: ${test_price:.2f}"))
                except Exception as e:
                    health_results.append(("Financial Calcs", "❌ Error", str(e)))
                
                # Display results
                for component, status, details in health_results:
                    col1, col2, col3 = st.columns([2, 1, 3])
                    
                    with col1:
                        st.write(f"**{component}**")
                    
                    with col2:
                        st.write(status)
                    
                    with col3:
                        st.write(details)
        
        # Performance metrics
        st.subheader("⚡ Performance Metrics")
        
        # Portfolio size
        st.metric("Portfolio Positions", len(st.session_state.portfolio))
        
        # Cache statistics
        try:
            cache_stats = self.data_manager.get_cache_stats()
            
            if cache_stats and cache_stats.get('cache_enabled', False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Cache Entries", cache_stats.get('total_items', 0))
                
                with col2:
                    st.metric("Active Entries", cache_stats.get('active_items', 0))
                
                with col3:
                    if st.button("Clear Cache"):
                        self.data_manager.clear_cache()
                        st.success("Cache cleared!")
            else:
                st.info("Caching is disabled")
                
        except Exception as e:
            st.error(f"Failed to get cache stats: {e}")


def main():
    """Main entry point for the dashboard."""
    try:
        dashboard = QortfolioDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Dashboard initialization failed: {e}")
        st.error("Please ensure all Qortfolio V2 modules are properly installed and configured.")


if __name__ == "__main__":
    main()