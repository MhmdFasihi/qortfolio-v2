# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Mathematical utilities for financial calculations.
Provides common mathematical functions used across the platform.
"""

import numpy as np
from typing import Union, List, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class MathUtils:
    """Mathematical utility functions for finance."""
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with zero handling."""
        if abs(denominator) < 1e-10:
            return default
        return numerator / denominator
    
    @staticmethod
    def calculate_returns(prices: np.ndarray, method: str = "simple") -> np.ndarray:
        """
        Calculate returns from price series.
        
        Args:
            prices: Array of prices
            method: 'simple' or 'log' returns
        """
        if len(prices) < 2:
            return np.array([])
        
        if method == "simple":
            returns = np.diff(prices) / prices[:-1]
        elif method == "log":
            returns = np.diff(np.log(prices))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return returns
    
    @staticmethod
    def calculate_volatility(returns: np.ndarray, annualize: bool = True) -> float:
        """
        Calculate volatility from returns.
        
        Args:
            returns: Array of returns
            annualize: Whether to annualize (assumes daily returns)
        """
        vol = np.std(returns)
        if annualize:
            vol *= np.sqrt(365)  # Trading days per year
        return vol
    
    @staticmethod
    def normal_cdf(x: float) -> float:
        """Standard normal cumulative distribution function."""
        return stats.norm.cdf(x)
    
    @staticmethod
    def normal_pdf(x: float) -> float:
        """Standard normal probability density function."""
        return stats.norm.pdf(x)
    
    @staticmethod
    def clip_value(value: float, min_val: float = -1e10, max_val: float = 1e10) -> float:
        """Clip value to prevent numerical overflow."""
        return np.clip(value, min_val, max_val)
    
    @staticmethod
    def is_close(a: float, b: float, tolerance: float = 1e-9) -> bool:
        """Check if two floats are close within tolerance."""
        return abs(a - b) < tolerance

if __name__ == "__main__":
    # Test math utilities
    prices = np.array([100, 102, 101, 103, 102])
    returns = MathUtils.calculate_returns(prices)
    volatility = MathUtils.calculate_volatility(returns)
    print(f"Returns: {returns}")
    print(f"Annualized Volatility: {volatility:.2%}")
