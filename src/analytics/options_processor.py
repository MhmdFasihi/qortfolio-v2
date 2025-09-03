# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.

"""
Options Analytics Processor - Integration with Real Data
File: src/analytics/options_processor.py
Integrates financial models with real Deribit data and MongoDB
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import asyncio
from dataclasses import dataclass, asdict
import json

# Import our financial models
from src.models.options.black_scholes import BlackScholesModel, OptionParameters, OptionType
from src.models.options.greeks_calculator import GreeksCalculator, PortfolioGreeks, RiskMetrics
from src.models.options.options_chain import OptionsChainProcessor, OptionChainMetrics

# Import data collectors
from src.data.collectors.deribit_collector import DeribitCollector
from src.data.collectors.data_manager import DataManager

# Import database operations
from src.core.database.operations import DatabaseOperations
from src.core.database.connection import DatabaseConnection

# Import utilities
from src.core.utils.time_utils import TimeUtils
from src.core.logging import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class OptionsAnalytics:
    """Complete options analytics results."""
    timestamp: datetime
    underlying: str
    spot_price: float
    options_count: int
    chain_metrics: Dict[str, Any]
    portfolio_greeks: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    top_opportunities: List[Dict[str, Any]]
    processing_time: float


class RealTimeOptionsProcessor:
    """
    Process real Deribit options data with Greeks calculation and storage.
    Designed to work with your actual 582 BTC options data.
    """
    
    def __init__(self):
        """Initialize processor with all components."""
        self.logger = setup_logger(__name__)
        
        # Initialize models
        self.bs_model = BlackScholesModel()
        self.greeks_calc = GreeksCalculator(self.bs_model)
        self.chain_processor = OptionsChainProcessor(self.bs_model, self.greeks_calc)
        
        # Initialize collectors
        self.deribit_collector = DeribitCollector()
        self.data_manager = DataManager()
        
        # Initialize database
        self.db_ops = DatabaseOperations()
        
        self.logger.info("Options Processor initialized with all components")
    
    async def process_live_options(self, currency: str = 'BTC') -> OptionsAnalytics:
        """
        Process live options data from Deribit.
        
        Args:
            currency: 'BTC' or 'ETH'
            
        Returns:
            Complete analytics results
        """
        start_time = datetime.now()
        
        try:
            # Fetch real options data from Deribit
            self.logger.info(f"Fetching live {currency} options from Deribit...")
            options_data = await self._fetch_real_deribit_data(currency)
            
            if not options_data:
                self.logger.warning(f"No options data received for {currency}")
                return None
            
            self.logger.info(f"Received {len(options_data)} {currency} options")
            
            # Process options chain with Greeks
            chain_df = self.chain_processor.process_deribit_chain(options_data)
            self.logger.info(f"Processed {len(chain_df)} options with Greeks")
            
            # Calculate chain metrics
            chain_metrics = self.chain_processor.analyze_chain_metrics(chain_df)
            
            # Calculate portfolio Greeks (assuming we hold all ATM options)
            portfolio_greeks = self._calculate_portfolio_greeks(chain_df)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(chain_df)
            
            # Identify opportunities
            opportunities = self.chain_processor.identify_opportunities(chain_df)
            
            # Store in MongoDB
            await self._store_to_mongodb(chain_df, chain_metrics)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create analytics result
            analytics = OptionsAnalytics(
                timestamp=datetime.now(),
                underlying=currency,
                spot_price=chain_df['underlying_price'].iloc[0] if not chain_df.empty else 0,
                options_count=len(chain_df),
                chain_metrics=self._metrics_to_dict(chain_metrics),
                portfolio_greeks=portfolio_greeks,
                risk_metrics=risk_metrics,
                top_opportunities=self._get_top_opportunities(opportunities),
                processing_time=processing_time
            )
            
            self.logger.info(f"Analytics complete in {processing_time:.2f} seconds")
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error processing live options: {e}")
            raise
    
    async def _fetch_real_deribit_data(self, currency: str) -> List[Dict]:
        """
        Fetch real options data from Deribit.
        This uses your actual Deribit collector.
        """
        try:
            # Use the real Deribit collector
            options_data = self.deribit_collector.get_options_data(currency)
            
            # If no live data, try to get from database
            if not options_data:
                self.logger.info("No live data, checking database for recent data...")
                options_data = self._get_recent_from_db(currency)
            
            return options_data
            
        except Exception as e:
            self.logger.error(f"Error fetching Deribit data: {e}")
            # Fallback to database
            return self._get_recent_from_db(currency)
    
    def _get_recent_from_db(self, currency: str) -> List[Dict]:
        """Get recent options data from MongoDB."""
        try:
            # Query recent options from database
            query = {
                'underlying': currency,
                'timestamp': {'$gte': datetime.now() - timedelta(hours=1)}
            }
            
            cursor = self.db_ops.db.options_data.find(query).limit(1000)
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"Error fetching from database: {e}")
            return []
    
    def _calculate_portfolio_greeks(self, chain_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate portfolio Greeks for ATM options.
        """
        if chain_df.empty:
            return {}
        
        # Select ATM options for portfolio
        atm_options = chain_df[chain_df['is_atm'] == True].copy()
        
        if atm_options.empty:
            # If no ATM, select closest to money
            chain_df['distance_from_atm'] = abs(1 - chain_df['moneyness'])
            atm_options = chain_df.nsmallest(10, 'distance_from_atm')
        
        # Create portfolio positions
        positions = []
        for _, option in atm_options.iterrows():
            positions.append({
                'quantity': 1,  # Assume 1 contract each
                'spot_price': option['underlying_price'],
                'strike_price': option['strike'],
                'time_to_maturity': option['time_to_maturity'],
                'volatility': option['implied_volatility'],
                'option_type': option['option_type'],
                'underlying': option['underlying'],
                'is_coin_based': True
            })
        
        # Calculate portfolio Greeks
        if positions:
            portfolio_greeks = self.greeks_calc.calculate_portfolio_greeks(positions)
            return portfolio_greeks.get_summary()
        
        return {}
    
    def _calculate_risk_metrics(self, chain_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics from chain."""
        if chain_df.empty:
            return {}
        
        # Aggregate risk metrics
        total_gamma = chain_df['gamma'].sum() if 'gamma' in chain_df else 0
        total_vega = chain_df['vega'].sum() if 'vega' in chain_df else 0
        total_theta = chain_df['theta'].sum() if 'theta' in chain_df else 0
        
        # Calculate gamma exposure
        spot = chain_df['underlying_price'].iloc[0]
        gamma_exposure = total_gamma * spot * spot / 100
        
        # Find max pain
        strikes = chain_df.groupby('strike')['open_interest'].sum()
        max_pain_strike = strikes.idxmax() if not strikes.empty else 0
        
        return {
            'total_gamma': total_gamma,
            'total_vega': total_vega,
            'total_theta': total_theta,
            'gamma_exposure': gamma_exposure,
            'max_pain_strike': max_pain_strike,
            'options_count': len(chain_df),
            'unique_strikes': chain_df['strike'].nunique(),
            'unique_expiries': chain_df['expiry'].nunique()
        }
    
    def _metrics_to_dict(self, metrics: OptionChainMetrics) -> Dict[str, Any]:
        """Convert metrics object to dictionary."""
        return {
            'total_volume': metrics.total_volume,
            'total_open_interest': metrics.total_open_interest,
            'put_call_ratio': metrics.put_call_ratio,
            'average_iv': metrics.average_iv,
            'iv_skew': metrics.iv_skew,
            'atm_iv': metrics.atm_iv,
            'max_pain_strike': metrics.max_pain_strike,
            'gamma_max_strike': metrics.gamma_max_strike,
            'total_gamma_exposure': metrics.total_gamma_exposure
        }
    
    def _get_top_opportunities(self, opportunities: Dict[str, List], top_n: int = 5) -> List[Dict]:
        """Get top opportunities from all categories."""
        top_opps = []
        
        for opp_type, opps in opportunities.items():
            for opp in opps[:top_n]:
                top_opps.append({
                    'type': opp_type,
                    'details': opp
                })
        
        return top_opps[:top_n]
    
    async def _store_to_mongodb(self, chain_df: pd.DataFrame, metrics: OptionChainMetrics):
        """Store processed data to MongoDB."""
        try:
            timestamp = datetime.now()
            
            # Store options with Greeks
            if not chain_df.empty:
                records = chain_df.to_dict('records')
                for record in records:
                    record['processed_at'] = timestamp
                    record['has_greeks'] = True
                    
                    # Clean up NaN values for MongoDB
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                
                # Insert to MongoDB
                result = self.db_ops.db.options_data.insert_many(records)
                self.logger.info(f"Stored {len(result.inserted_ids)} options with Greeks to MongoDB")
            
            # Store chain metrics
            metrics_doc = {
                'timestamp': timestamp,
                'metrics': self._metrics_to_dict(metrics),
                'options_count': len(chain_df)
            }
            
            self.db_ops.db.chain_metrics.insert_one(metrics_doc)
            self.logger.info("Stored chain metrics to MongoDB")
            
        except Exception as e:
            self.logger.error(f"Error storing to MongoDB: {e}")
    
    def get_current_portfolio_risk(self) -> Dict[str, Any]:
        """Get current portfolio risk metrics from database."""
        try:
            # Get latest options from database
            latest_options = list(
                self.db_ops.db.options_data.find(
                    {'has_greeks': True}
                ).sort('processed_at', -1).limit(100)
            )
            
            if not latest_options:
                return {}
            
            # Calculate aggregate risk
            total_delta = sum(opt.get('delta', 0) for opt in latest_options)
            total_gamma = sum(opt.get('gamma', 0) for opt in latest_options)
            total_theta = sum(opt.get('theta', 0) for opt in latest_options)
            total_vega = sum(opt.get('vega', 0) for opt in latest_options)
            
            return {
                'total_delta': total_delta,
                'total_gamma': total_gamma,
                'total_theta': total_theta,
                'total_vega': total_vega,
                'options_count': len(latest_options),
                'last_update': latest_options[0].get('processed_at')
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio risk: {e}")
            return {}


class OptionsDataValidator:
    """Validate options data quality."""
    
    @staticmethod
    def validate_deribit_data(options_data: List[Dict]) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Validate and clean Deribit options data.
        
        Returns:
            Tuple of (cleaned_data, validation_report)
        """
        cleaned = []
        issues = {
            'missing_fields': 0,
            'invalid_prices': 0,
            'invalid_iv': 0,
            'expired': 0,
            'total': len(options_data)
        }
        
        for option in options_data:
            # Check required fields
            required = ['instrument_name', 'mark_price', 'underlying_price']
            if not all(field in option for field in required):
                issues['missing_fields'] += 1
                continue
            
            # Validate prices
            if option.get('mark_price', 0) <= 0:
                issues['invalid_prices'] += 1
                continue
            
            # Validate IV
            iv = option.get('mark_iv', 0)
            if iv <= 0 or iv > 500:  # IV should be between 0 and 500%
                issues['invalid_iv'] += 1
                option['mark_iv'] = 80  # Default to 80% if invalid
            
            # Check expiration
            try:
                instrument = option['instrument_name']
                if 'PERPETUAL' not in instrument:
                    # Parse expiry date
                    parts = instrument.split('-')
                    if len(parts) >= 2:
                        expiry_str = parts[1]
                        # Simple check if expired (you can make this more sophisticated)
                        # For now, just add to cleaned
                        cleaned.append(option)
                else:
                    cleaned.append(option)
            except:
                cleaned.append(option)
        
        validation_report = {
            'total_options': issues['total'],
            'valid_options': len(cleaned),
            'issues': issues,
            'validity_rate': len(cleaned) / issues['total'] if issues['total'] > 0 else 0
        }
        
        return cleaned, validation_report


# Async wrapper for processing
async def process_options_async(currency: str = 'BTC') -> OptionsAnalytics:
    """Async wrapper for processing options."""
    processor = RealTimeOptionsProcessor()
    return await processor.process_live_options(currency)


# Synchronous wrapper for easy use
def process_options(currency: str = 'BTC') -> OptionsAnalytics:
    """Synchronous wrapper for processing options."""
    return asyncio.run(process_options_async(currency))


if __name__ == "__main__":
    # Example usage with real data
    print("Processing real Deribit options data...")
    
    try:
        # Process BTC options
        analytics = process_options('BTC')
        
        if analytics:
            print(f"\n✅ Processed {analytics.options_count} BTC options")
            print(f"Spot Price: ${analytics.spot_price:,.2f}")
            print(f"Average IV: {analytics.chain_metrics['average_iv']*100:.1f}%")
            print(f"Put/Call Ratio: {analytics.chain_metrics['put_call_ratio']:.3f}")
            print(f"Processing Time: {analytics.processing_time:.2f} seconds")
            
            if analytics.portfolio_greeks:
                print(f"\nPortfolio Greeks:")
                print(f"  Total Delta: {analytics.portfolio_greeks.get('total_delta', 0):.4f}")
                print(f"  Total Gamma: {analytics.portfolio_greeks.get('total_gamma', 0):.6f}")
        else:
            print("No analytics generated - check data source")
            
    except Exception as e:
        print(f"Error: {e}")