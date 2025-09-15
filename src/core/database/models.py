# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
MongoDB database models for Qortfolio V2.
Defines schemas for all collections.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class OptionType(Enum):
    """Option type enumeration."""
    CALL = "call"
    PUT = "put"

class OrderType(Enum):
    """Order type enumeration."""
    BUY = "buy"
    SELL = "sell"

@dataclass
class OptionsData:
    """Options data model."""
    symbol: str
    underlying: str
    strike: float
    expiry: datetime
    option_type: OptionType
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for MongoDB."""
        data = asdict(self)
        data['option_type'] = self.option_type.value
        return data

@dataclass
class PriceData:
    """Price data model."""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime
    source: str = "yfinance"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for MongoDB."""
        return asdict(self)

@dataclass
class PortfolioPosition:
    """Portfolio position model."""
    user_id: str
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    position_type: str  # 'spot', 'option'
    entry_date: datetime
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()
    
    @property
    def pnl(self) -> float:
        """Calculate P&L."""
        return (self.current_price - self.entry_price) * self.quantity
    
    @property
    def pnl_percentage(self) -> float:
        """Calculate P&L percentage."""
        if self.entry_price == 0:
            return 0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for MongoDB."""
        data = asdict(self)
        data['pnl'] = self.pnl
        data['pnl_percentage'] = self.pnl_percentage
        return data

@dataclass
class RiskMetrics:
    """Risk metrics model."""
    portfolio_id: str
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    cvar_95: float  # Conditional VaR 95%
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    beta: float
    correlation_matrix: Dict[str, Dict[str, float]]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict:
        """Convert to dictionary for MongoDB."""
        return asdict(self)


@dataclass
class PortfolioData:
    """Portfolio configuration and allocation data model."""
    portfolio_id: str
    user_id: str
    assets: List[str]
    weights: Dict[str, float]
    total_value: float
    cash_position: float
    currency: str = "USD"
    creation_date: datetime = None
    last_updated: datetime = None

    def __post_init__(self):
        if self.creation_date is None:
            self.creation_date = datetime.utcnow()
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()

    def to_dict(self) -> Dict:
        """Convert to dictionary for MongoDB."""
        return asdict(self)


@dataclass
class PerformanceReport:
    """Performance analytics report model."""
    portfolio_id: str
    analysis_date: datetime
    lookback_days: int

    # Basic returns
    total_return: float
    annual_return: float
    annual_volatility: float

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float

    # Risk metrics
    max_drawdown: float
    value_at_risk: float
    conditional_value_at_risk: float

    # Distribution metrics
    skewness: float
    kurtosis: float

    # Win/Loss metrics
    win_rate: float
    profit_factor: float

    # Optional fields with default values
    benchmark_symbol: Optional[str] = None
    information_ratio: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    tracking_error: Optional[float] = None
    up_capture_ratio: Optional[float] = None
    down_capture_ratio: Optional[float] = None
    kelly_criterion: Optional[float] = None
    ulcer_index: Optional[float] = None
    recovery_factor: Optional[float] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict:
        """Convert to dictionary for MongoDB."""
        return asdict(self)


@dataclass
class SectorAllocation:
    """Crypto sector allocation model."""
    portfolio_id: str
    sector_name: str
    assets: List[str]
    allocation_percentage: float
    current_value: float
    target_value: float
    rebalance_needed: bool
    last_rebalance_date: Optional[datetime] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict:
        """Convert to dictionary for MongoDB."""
        return asdict(self)


@dataclass
class VolatilitySurfaceData:
    """Volatility surface data model."""
    currency: str
    spot_price: float
    surface_data: Dict[str, Any]  # Grid data for interpolation
    atm_term_structure: Dict[str, float]  # ATM vol by expiry
    skew_data: Dict[str, Dict[str, float]]  # Vol skew by expiry
    quality_metrics: Dict[str, float]  # Surface quality metrics
    data_points_count: int
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict:
        """Convert to dictionary for MongoDB."""
        return asdict(self)


@dataclass
class OptionsChainAnalytics:
    """Options chain analytics model."""
    currency: str
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
    iv_rank: float
    term_structure_slope: float
    skew_25d: float
    flow_direction: str  # bullish, bearish, neutral
    unusual_activity: List[Dict[str, Any]]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict:
        """Convert to dictionary for MongoDB."""
        return asdict(self)


@dataclass
class GreeksSnapshot:
    """Portfolio Greeks snapshot model."""
    portfolio_id: str
    currency: str
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    total_rho: float
    delta_dollars: float
    gamma_dollars: float
    portfolio_value: float
    positions_count: int
    by_underlying: Dict[str, Dict[str, float]]
    by_expiry: Dict[str, Dict[str, float]]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict:
        """Convert to dictionary for MongoDB."""
        return asdict(self)


@dataclass
class ImpliedVolatilityPoint:
    """Individual IV data point model."""
    currency: str
    symbol: str
    strike: float
    expiry: datetime
    option_type: str
    spot_price: float
    market_price: float
    implied_volatility: float
    moneyness: float
    time_to_maturity: float
    volume: int
    open_interest: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict:
        """Convert to dictionary for MongoDB."""
        return asdict(self)

# Collection schemas for validation
SCHEMAS = {
    "options_data": {
        "symbol": str,
        "underlying": str,
        "strike": float,
        "expiry": datetime,
        "option_type": str,
        "bid": float,
        "ask": float,
        "last_price": float,
        "volume": int,
        "open_interest": int,
        "implied_volatility": float,
        "timestamp": datetime
    },
    "price_data": {
        "symbol": str,
        "open": float,
        "high": float,
        "low": float,
        "close": float,
        "volume": float,
        "timestamp": datetime,
        "source": str
    },
    "portfolio_positions": {
        "user_id": str,
        "symbol": str,
        "quantity": float,
        "entry_price": float,
        "current_price": float,
        "position_type": str,
        "entry_date": datetime,
        "last_updated": datetime
    },
    "portfolio_data": {
        "portfolio_id": str,
        "user_id": str,
        "assets": list,
        "weights": dict,
        "total_value": float,
        "cash_position": float,
        "currency": str,
        "creation_date": datetime,
        "last_updated": datetime
    },
    "risk_metrics": {
        "portfolio_id": str,
        "metrics": dict,
        "calculated_by": str,
        "timestamp": datetime
    },
    "performance_reports": {
        "portfolio_id": str,
        "performance_report": dict,
        "generated_by": str,
        "timestamp": datetime
    },
    "sector_risk_metrics": {
        "portfolio_id": str,
        "sector_allocation_risk": dict,
        "timestamp": datetime
    },
    "portfolio_comparisons": {
        "portfolio_metrics": dict,
        "correlation_matrix": dict,
        "comparison_date": datetime,
        "portfolios_compared": int,
        "lookback_days": int
    },
    "sector_allocations": {
        "portfolio_id": str,
        "sector_name": str,
        "assets": list,
        "allocation_percentage": float,
        "current_value": float,
        "target_value": float,
        "rebalance_needed": bool,
        "timestamp": datetime
    },
    "volatility_surfaces": {
        "currency": str,
        "spot_price": float,
        "surface_data": dict,
        "atm_term_structure": dict,
        "skew_data": dict,
        "quality_metrics": dict,
        "data_points_count": int,
        "timestamp": datetime
    },
    "volatility_surfaces_history": {
        "currency": str,
        "spot_price": float,
        "surface_data": dict,
        "timestamp": datetime
    },
    "options_chain_analytics": {
        "currency": str,
        "total_call_volume": int,
        "total_put_volume": int,
        "call_put_ratio": float,
        "max_pain_strike": float,
        "gamma_exposure": float,
        "flow_direction": str,
        "timestamp": datetime
    },
    "greeks_snapshots": {
        "portfolio_id": str,
        "currency": str,
        "total_delta": float,
        "total_gamma": float,
        "total_theta": float,
        "total_vega": float,
        "portfolio_value": float,
        "timestamp": datetime
    },
    "implied_volatility_points": {
        "currency": str,
        "symbol": str,
        "strike": float,
        "implied_volatility": float,
        "moneyness": float,
        "volume": int,
        "timestamp": datetime
    },
    "performance_attribution": {
        "portfolio_id": str,
        "attribution_type": str,
        "attribution_results": dict,
        "lookback_days": int,
        "timestamp": datetime
    },
    "risk_adjusted_metrics": {
        "portfolio_id": str,
        "risk_adjusted_metrics": dict,
        "risk_free_rate": float,
        "lookback_days": int,
        "timestamp": datetime
    },
    "comprehensive_tearsheets": {
        "portfolio_id": str,
        "comprehensive_tearsheet": dict,
        "timestamp": datetime
    }
}

if __name__ == "__main__":
    # Test models
    option = OptionsData(
        symbol="BTC-31JAN25-100000-C",
        underlying="BTC",
        strike=100000,
        expiry=datetime(2025, 1, 31),
        option_type=OptionType.CALL,
        bid=5000,
        ask=5100,
        last_price=5050,
        volume=100,
        open_interest=500,
        implied_volatility=0.65
    )
    
    print("Options Data Model:")
    print(option.to_dict())
