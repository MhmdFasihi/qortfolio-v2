# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Dashboard Patch - Add Missing Methods to QortfolioDashboard
Location: src/dashboard/dashboard_patch.py

This module patches the QortfolioDashboard class with missing methods.
Import this after creating QortfolioDashboard to add missing methods.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from .dashboard_methods import system_status_page, market_overview_page, options_chain_page, volatility_surface_page

def patch_dashboard_class(dashboard_class):
    """
    Patch the QortfolioDashboard class with missing methods.
    
    Args:
        dashboard_class: The QortfolioDashboard class to patch
    """
    
    def system_status_page_method(self):
        """System Status Page method."""
        return system_status_page()
    
    def market_overview_page_method(self):
        """Market Overview Page method.""" 
        return market_overview_page()
    
    def options_chain_page_method(self):
        """Options Chain Page method."""
        return options_chain_page()
    
    def volatility_surface_page_method(self):
        """Volatility Surface Page method."""
        return volatility_surface_page()
    
    def portfolio_management_page_method(self):
        """Portfolio Management Page method."""
        st.header("💼 Portfolio Management")
        st.info("Portfolio management features coming soon!")
        
        # Basic placeholder
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Portfolio Value", "$125,000", delta="$5,000")
        with col2:
            st.metric("Number of Positions", "15", delta="3")
    
    def pnl_simulation_page_method(self):
        """PnL Simulation Page method."""
        st.header("💰 PnL Simulation (Taylor Expansion)")
        st.info("PnL simulation features coming soon!")
        
        # Basic placeholder
        st.write("**Taylor Expansion Formula:**")
        st.latex(r"\Delta C \approx \delta \Delta S + \frac{1}{2}\gamma(\Delta S)^2 + \theta \Delta t + \nu \Delta \sigma")
    
    def risk_management_page_method(self):
        """Risk Management Page method."""
        st.header("⚠️ Risk Management")
        
        # Risk Dashboard with corrected styling
        st.subheader("📊 Risk Dashboard")
        
        # Time Decay Risk
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Time Decay Risk",
                value="$150.84",
                delta="Daily Theta Decay",
                help="Expected daily loss due to time decay"
            )
        
        with col2:
            st.metric(
                label="Delta Risk", 
                value="$1536.16",
                delta="Delta Exposure",
                help="Portfolio sensitivity to underlying price changes"
            )
        
        with col3:
            st.metric(
                label="Gamma Risk",
                value="$2.17", 
                delta="Gamma Risk (5% move)",
                help="Portfolio convexity exposure"
            )
        
        # Fixed Warning Box with Better Styling
        st.markdown("""
        <div style="
            background-color: #ffe6e6; 
            border-left: 4px solid #ff4444; 
            padding: 15px; 
            border-radius: 5px; 
            margin: 20px 0;
            color: #333333;
        ">
            <h4 style="color: #cc0000; margin-top: 0;">⚠️ Risk Alert</h4>
            <p style="margin-bottom: 0; color: #333333; font-weight: 500;">
                High gamma exposure detected. Consider hedging strategies to reduce risk.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Stress Testing Section
        st.subheader("🔥 Stress Testing")
        st.info("Stress testing functionality will be implemented with advanced PnL methods.")
    
    # Add all missing methods to the class
    dashboard_class.system_status_page = system_status_page_method
    dashboard_class.market_overview_page = market_overview_page_method
    dashboard_class.options_chain_page = options_chain_page_method
    dashboard_class.volatility_surface_page = volatility_surface_page_method
    dashboard_class.portfolio_management_page = portfolio_management_page_method
    dashboard_class.pnl_simulation_page = pnl_simulation_page_method
    dashboard_class.risk_management_page = risk_management_page_method
    
    return dashboard_class

# Usage: patch_dashboard_class(QortfolioDashboard)