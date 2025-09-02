# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Options Chain Processor for Deribit Data
Processes options chains, calculates Greeks, and identifies trading opportunities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

# Import our models
from .black_scholes import BlackScholesModel, OptionParameters, OptionType
from .greeks_calculator import GreeksCalculator, PortfolioGreeks

# Import database operations
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.core.database.operations import DatabaseOperations

logger = logging.getLogger(__name__)


@dataclass
class OptionChainMetrics:
    """Metrics for an options chain."""
    total_volume: float
    total_open_interest: float
    put_call_ratio: float
    average_iv: float
    iv_skew: float  # Difference between OTM put and call IV
    atm_iv: float
    term_structure: Dict[str, float]  # IV by expiry
    max_pain_strike: float
    gamma_max_strike: float
    total_gamma_exposure: float


@dataclass
class OptionContract:
    """Single option contract with all data."""
    instrument_name: str
    underlying: str
    strike: float
    expiry: datetime
    option_type: str
    bid: float
    ask: float
    mark_price: float
    underlying_price: float
    volume: float
    open_interest: float
    implied_volatility: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    time_to_maturity: Optional[float] = None
    moneyness: Optional[float] = None  # S/K
    is_itm: Optional[bool] = None
    is_atm: Optional[bool] = None
    is_otm: Optional[bool] = None


class OptionsChainProcessor:
    """
    Process and analyze options chains from Deribit.
    """
    
    def __init__(self, bs_model: Optional[BlackScholesModel] = None,
                 greeks_calc: Optional[GreeksCalculator] = None,
                 db_ops: Optional[DatabaseOperations] = None):
        """
        Initialize options chain processor.
        
        Args:
            bs_model: Black-Scholes model instance
            greeks_calc: Greeks calculator instance
            db_ops: Database operations instance
        """
        self.bs_model = bs_model or BlackScholesModel()
        self.greeks_calc = greeks_calc or GreeksCalculator(self.bs_model)
        self.db_ops = db_ops
        self.logger = logging.getLogger(__name__)
    
    def process_deribit_chain(self, options_data: List[Dict]) -> pd.DataFrame:
        """
        Process raw Deribit options data into structured chain.
        
        Args:
            options_data: List of option dictionaries from Deribit
            
        Returns:
            DataFrame with processed options chain
        """
        if not options_data:
            self.logger.warning("No options data provided")
            return pd.DataFrame()
        
        processed_options = []
        current_time = datetime.now()
        
        for option in options_data:
            try:
                # Parse instrument name (e.g., "BTC-28JUN24-50000-C")
                instrument_parts = option['instrument_name'].split('-')
                underlying = instrument_parts[0]
                expiry_str = instrument_parts[1]
                strike = float(instrument_parts[2])
                option_type = 'call' if instrument_parts[3] == 'C' else 'put'
                
                # Parse expiry date
                expiry = self._parse_deribit_expiry(expiry_str)
                time_to_maturity = (expiry - current_time).total_seconds() / (365.25 * 24 * 3600)
                
                # Get prices (Deribit prices are in crypto terms)
                underlying_price = option.get('underlying_price', option.get('index_price', 0))
                mark_price = option.get('mark_price', 0)
                
                # Create option contract
                contract = OptionContract(
                    instrument_name=option['instrument_name'],
                    underlying=underlying,
                    strike=strike,
                    expiry=expiry,
                    option_type=option_type,
                    bid=option.get('best_bid_price', 0),
                    ask=option.get('best_ask_price', 0),
                    mark_price=mark_price,
                    underlying_price=underlying_price,
                    volume=option.get('volume', 0),
                    open_interest=option.get('open_interest', 0),
                    implied_volatility=option.get('mark_iv', 0) / 100,  # Convert from percentage
                    time_to_maturity=time_to_maturity,
                    moneyness=underlying_price / strike if strike > 0 else 0
                )
                
                # Classify moneyness
                moneyness_threshold = 0.02  # 2% threshold for ATM
                if option_type == 'call':
                    contract.is_itm = contract.moneyness > (1 + moneyness_threshold)
                    contract.is_atm = abs(contract.moneyness - 1) <= moneyness_threshold
                    contract.is_otm = contract.moneyness < (1 - moneyness_threshold)
                else:  # put
                    contract.is_itm = contract.moneyness < (1 - moneyness_threshold)
                    contract.is_atm = abs(contract.moneyness - 1) <= moneyness_threshold
                    contract.is_otm = contract.moneyness > (1 + moneyness_threshold)
                
                # Calculate Greeks if we have valid data
                if underlying_price > 0 and strike > 0 and time_to_maturity > 0 and contract.implied_volatility > 0:
                    greeks = self._calculate_contract_greeks(contract)
                    contract.delta = greeks['delta']
                    contract.gamma = greeks['gamma']
                    contract.theta = greeks['theta']
                    contract.vega = greeks['vega']
                    contract.rho = greeks['rho']
                
                processed_options.append(contract)
                
            except Exception as e:
                self.logger.error(f"Error processing option {option.get('instrument_name', 'UNKNOWN')}: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame([vars(c) for c in processed_options])
        
        # Sort by expiry and strike
        if not df.empty:
            df = df.sort_values(['expiry', 'strike', 'option_type'])
        
        self.logger.info(f"Processed {len(df)} options from chain")
        return df
    
    def _parse_deribit_expiry(self, expiry_str: str) -> datetime:
        """
        Parse Deribit expiry string (e.g., '28JUN24').
        
        Args:
            expiry_str: Expiry string from Deribit
            
        Returns:
            Expiry datetime
        """
        # Handle different formats (e.g., '3SEP25' or '28JUN24')
        expiry_str = expiry_str.strip()
        if len(expiry_str) == 7:  # e.g., '28JUN24'
            day = int(expiry_str[:2])
            month_str = expiry_str[2:5]
            year = 2000 + int(expiry_str[5:7])
        elif len(expiry_str) == 6:  # e.g., '3SEP25'
            day = int(expiry_str[:1])
            month_str = expiry_str[1:4]
            year = 2000 + int(expiry_str[4:6])
        else:
            # Handle other formats if needed
            raise ValueError(f"Unknown expiry format: {expiry_str}")
        
        # Convert month string to number
        months = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        month = months.get(month_str.upper(), 1)
        
        # Deribit expiries are at 08:00 UTC
        return datetime(year, month, day, 8, 0, 0)
    
    def _calculate_contract_greeks(self, contract: OptionContract) -> Dict[str, float]:
        """
        Calculate Greeks for a single contract.
        
        Args:
            contract: Option contract
            
        Returns:
            Dictionary of Greeks
        """
        params = OptionParameters(
            spot_price=contract.underlying_price,
            strike_price=contract.strike,
            time_to_maturity=contract.time_to_maturity,
            volatility=contract.implied_volatility,
            risk_free_rate=0.05,  # Could be adjusted for crypto
            option_type=OptionType.CALL if contract.option_type == 'call' else OptionType.PUT,
            is_coin_based=True  # Deribit options are coin-based
        )
        
        pricing = self.bs_model.calculate_option_price(params)
        
        return {
            'delta': pricing.delta,
            'gamma': pricing.gamma,
            'theta': pricing.theta,
            'vega': pricing.vega,
            'rho': pricing.rho
        }
    
    def analyze_chain_metrics(self, chain_df: pd.DataFrame) -> OptionChainMetrics:
        """
        Calculate comprehensive metrics for options chain.
        
        Args:
            chain_df: Processed options chain DataFrame
            
        Returns:
            Option chain metrics
        """
        if chain_df.empty:
            return OptionChainMetrics(
                total_volume=0, total_open_interest=0, put_call_ratio=0,
                average_iv=0, iv_skew=0, atm_iv=0, term_structure={},
                max_pain_strike=0, gamma_max_strike=0, total_gamma_exposure=0
            )
        
        # Calculate basic metrics
        total_volume = chain_df['volume'].sum()
        total_open_interest = chain_df['open_interest'].sum()
        
        # Put-Call ratio
        puts = chain_df[chain_df['option_type'] == 'put']
        calls = chain_df[chain_df['option_type'] == 'call']
        put_volume = puts['volume'].sum()
        call_volume = calls['volume'].sum()
        put_call_ratio = put_volume / call_volume if call_volume > 0 else 0
        
        # IV metrics
        average_iv = chain_df['implied_volatility'].mean()
        
        # ATM IV (options closest to spot)
        atm_options = chain_df[chain_df['is_atm'] == True]
        atm_iv = atm_options['implied_volatility'].mean() if not atm_options.empty else average_iv
        
        # IV Skew (25 delta put IV - 25 delta call IV)
        iv_skew = self._calculate_iv_skew(chain_df)
        
        # Term structure
        term_structure = {}
        for expiry in chain_df['expiry'].unique():
            expiry_options = chain_df[chain_df['expiry'] == expiry]
            term_structure[str(expiry)] = expiry_options['implied_volatility'].mean()
        
        # Max pain (strike with maximum open interest pain)
        max_pain_strike = self._calculate_max_pain(chain_df)
        
        # Gamma maximum strike
        gamma_max_strike = self._find_gamma_max_strike(chain_df)
        
        # Total gamma exposure
        total_gamma_exposure = self._calculate_total_gamma_exposure(chain_df)
        
        return OptionChainMetrics(
            total_volume=total_volume,
            total_open_interest=total_open_interest,
            put_call_ratio=put_call_ratio,
            average_iv=average_iv,
            iv_skew=iv_skew,
            atm_iv=atm_iv,
            term_structure=term_structure,
            max_pain_strike=max_pain_strike,
            gamma_max_strike=gamma_max_strike,
            total_gamma_exposure=total_gamma_exposure
        )
    
    def _calculate_iv_skew(self, chain_df: pd.DataFrame) -> float:
        """Calculate IV skew from chain."""
        try:
            # Find 25 delta options
            otm_puts = chain_df[(chain_df['option_type'] == 'put') & 
                               (chain_df['delta'].between(-0.3, -0.2))]
            otm_calls = chain_df[(chain_df['option_type'] == 'call') & 
                                (chain_df['delta'].between(0.2, 0.3))]
            
            if not otm_puts.empty and not otm_calls.empty:
                put_iv = otm_puts['implied_volatility'].mean()
                call_iv = otm_calls['implied_volatility'].mean()
                return put_iv - call_iv
        except:
            pass
        return 0
    
    def _calculate_max_pain(self, chain_df: pd.DataFrame) -> float:
        """Calculate max pain strike."""
        if chain_df.empty:
            return 0
        
        strikes = chain_df['strike'].unique()
        max_pain = 0
        max_pain_value = float('inf')
        
        for strike in strikes:
            pain = 0
            # Calculate pain for calls
            calls = chain_df[(chain_df['option_type'] == 'call') & (chain_df['strike'] < strike)]
            pain += ((strike - calls['strike']) * calls['open_interest']).sum()
            
            # Calculate pain for puts
            puts = chain_df[(chain_df['option_type'] == 'put') & (chain_df['strike'] > strike)]
            pain += ((puts['strike'] - strike) * puts['open_interest']).sum()
            
            if pain < max_pain_value:
                max_pain_value = pain
                max_pain = strike
        
        return max_pain
    
    def _find_gamma_max_strike(self, chain_df: pd.DataFrame) -> float:
        """Find strike with maximum gamma."""
        if chain_df.empty or 'gamma' not in chain_df.columns:
            return 0
        
        gamma_by_strike = chain_df.groupby('strike')['gamma'].sum()
        if not gamma_by_strike.empty:
            return gamma_by_strike.idxmax()
        return 0
    
    def _calculate_total_gamma_exposure(self, chain_df: pd.DataFrame) -> float:
        """Calculate total gamma exposure in dollar terms."""
        if chain_df.empty or 'gamma' not in chain_df.columns:
            return 0
        
        # Gamma exposure = gamma * open_interest * spot^2 / 100
        chain_df['gamma_exposure'] = (
            chain_df['gamma'] * 
            chain_df['open_interest'] * 
            chain_df['underlying_price']**2 / 100
        )
        
        return chain_df['gamma_exposure'].sum()
    
    def identify_opportunities(self, chain_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify trading opportunities in the options chain.
        
        Args:
            chain_df: Processed options chain
            
        Returns:
            Dictionary of identified opportunities
        """
        opportunities = {
            'high_iv_premium': [],
            'low_iv_discount': [],
            'high_volume': [],
            'gamma_concentration': [],
            'calendar_spreads': [],
            'skew_trades': []
        }
        
        if chain_df.empty:
            return opportunities
        
        # Find high IV premium opportunities (IV > historical average)
        avg_iv = chain_df['implied_volatility'].mean()
        std_iv = chain_df['implied_volatility'].std()
        
        high_iv = chain_df[chain_df['implied_volatility'] > avg_iv + 1.5 * std_iv]
        for _, option in high_iv.iterrows():
            opportunities['high_iv_premium'].append({
                'instrument': option['instrument_name'],
                'iv': option['implied_volatility'],
                'z_score': (option['implied_volatility'] - avg_iv) / std_iv if std_iv > 0 else 0
            })
        
        # Find low IV discount opportunities
        low_iv = chain_df[chain_df['implied_volatility'] < avg_iv - 1.5 * std_iv]
        for _, option in low_iv.iterrows():
            opportunities['low_iv_discount'].append({
                'instrument': option['instrument_name'],
                'iv': option['implied_volatility'],
                'z_score': (option['implied_volatility'] - avg_iv) / std_iv if std_iv > 0 else 0
            })
        
        # High volume activity
        volume_threshold = chain_df['volume'].quantile(0.95)
        high_volume = chain_df[chain_df['volume'] > volume_threshold]
        for _, option in high_volume.iterrows():
            opportunities['high_volume'].append({
                'instrument': option['instrument_name'],
                'volume': option['volume'],
                'oi': option['open_interest'],
                'volume_oi_ratio': option['volume'] / option['open_interest'] if option['open_interest'] > 0 else 0
            })
        
        # Gamma concentration
        if 'gamma' in chain_df.columns:
            gamma_threshold = chain_df['gamma'].quantile(0.95)
            high_gamma = chain_df[chain_df['gamma'] > gamma_threshold]
            for _, option in high_gamma.iterrows():
                opportunities['gamma_concentration'].append({
                    'instrument': option['instrument_name'],
                    'strike': option['strike'],
                    'gamma': option['gamma'],
                    'gamma_exposure': option['gamma'] * option['open_interest'] * option['underlying_price']**2 / 100
                })
        
        # Calendar spread opportunities (same strike, different expiries)
        for strike in chain_df['strike'].unique():
            strike_options = chain_df[chain_df['strike'] == strike]
            if len(strike_options['expiry'].unique()) > 1:
                # Find IV differences between expiries
                iv_by_expiry = strike_options.groupby('expiry')['implied_volatility'].mean()
                if len(iv_by_expiry) > 1:
                    max_iv_expiry = iv_by_expiry.idxmax()
                    min_iv_expiry = iv_by_expiry.idxmin()
                    iv_spread = iv_by_expiry.max() - iv_by_expiry.min()
                    
                    if iv_spread > 0.1:  # 10% IV difference
                        opportunities['calendar_spreads'].append({
                            'strike': strike,
                            'sell_expiry': max_iv_expiry,
                            'buy_expiry': min_iv_expiry,
                            'iv_spread': iv_spread
                        })
        
        return opportunities
    
    def save_to_database(self, chain_df: pd.DataFrame, metrics: OptionChainMetrics) -> bool:
        """
        Save processed chain and metrics to database.
        
        Args:
            chain_df: Processed options chain
            metrics: Chain metrics
            
        Returns:
            Success status
        """
        if self.db_ops is None:
            self.logger.warning("No database operations configured")
            return False
        
        try:
            # Convert DataFrame to records
            options_records = chain_df.to_dict('records')
            
            # Add timestamp
            timestamp = datetime.now()
            for record in options_records:
                record['processed_at'] = timestamp
                record['chain_metrics'] = vars(metrics)
            
            # Store in database
            result = self.db_ops.store_options_data(options_records)
            
            self.logger.info(f"Saved {len(options_records)} options to database")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to save to database: {e}")
            return False


# Convenience functions
def process_deribit_options(options_data: List[Dict]) -> pd.DataFrame:
    """
    Quick function to process Deribit options data.
    
    Args:
        options_data: Raw options data from Deribit
        
    Returns:
        Processed DataFrame with Greeks
    """
    processor = OptionsChainProcessor()
    return processor.process_deribit_chain(options_data)


def analyze_options_chain(chain_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze an options chain for metrics and opportunities.
    
    Args:
        chain_df: Processed options chain DataFrame
        
    Returns:
        Dictionary with metrics and opportunities
    """
    processor = OptionsChainProcessor()
    
    metrics = processor.analyze_chain_metrics(chain_df)
    opportunities = processor.identify_opportunities(chain_df)
    
    return {
        'metrics': vars(metrics),
        'opportunities': opportunities
    }


if __name__ == "__main__":
    # Example usage
    print("Options Chain Processor for Deribit")
    print("=" * 50)
    
    # Sample Deribit options data
    sample_options = [
        {
            'instrument_name': 'BTC-28JUN24-50000-C',
            'underlying_price': 50000,
            'index_price': 50000,
            'mark_price': 0.0523,  # In BTC
            'mark_iv': 80,  # 80% IV
            'best_bid_price': 0.0520,
            'best_ask_price': 0.0526,
            'volume': 125.5,
            'open_interest': 523.2
        },
        {
            'instrument_name': 'BTC-28JUN24-52000-C',
            'underlying_price': 50000,
            'index_price': 50000,
            'mark_price': 0.0385,  # In BTC
            'mark_iv': 82,  # 82% IV
            'best_bid_price': 0.0380,
            'best_ask_price': 0.0390,
            'volume': 89.3,
            'open_interest': 412.7
        },
        {
            'instrument_name': 'BTC-28JUN24-48000-P',
            'underlying_price': 50000,
            'index_price': 50000,
            'mark_price': 0.0412,  # In BTC
            'mark_iv': 85,  # 85% IV
            'best_bid_price': 0.0408,
            'best_ask_price': 0.0416,
            'volume': 156.8,
            'open_interest': 687.3
        }
    ]
    
    # Process the chain
    processor = OptionsChainProcessor()
    chain_df = processor.process_deribit_chain(sample_options)
    
    print(f"\n1. Processed {len(chain_df)} options")
    print("\nSample processed data:")
    print(chain_df[['instrument_name', 'strike', 'option_type', 'mark_price', 
                   'implied_volatility', 'delta', 'gamma']].head())
    
    # Analyze metrics
    metrics = processor.analyze_chain_metrics(chain_df)
    print(f"\n2. Chain Metrics:")
    print(f"   Total Volume: {metrics.total_volume:.2f}")
    print(f"   Put/Call Ratio: {metrics.put_call_ratio:.2f}")
    print(f"   Average IV: {metrics.average_iv:.2%}")
    print(f"   ATM IV: {metrics.atm_iv:.2%}")
    print(f"   Max Pain Strike: ${metrics.max_pain_strike:,.0f}")
    
    # Identify opportunities
    opportunities = processor.identify_opportunities(chain_df)
    print(f"\n3. Identified Opportunities:")
    for opp_type, opps in opportunities.items():
        if opps:
            print(f"   {opp_type}: {len(opps)} opportunities found")
    
    print("\n✅ Options Chain Processor ready for Qortfolio V2!")