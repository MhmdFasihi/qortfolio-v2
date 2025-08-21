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
    "risk_metrics": {
        "portfolio_id": str,
        "var_95": float,
        "var_99": float,
        "cvar_95": float,
        "sharpe_ratio": float,
        "sortino_ratio": float,
        "max_drawdown": float,
        "beta": float,
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
