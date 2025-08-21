# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Custom exception classes for Qortfolio V2.
Provides specific error handling for different components.
"""

class QortfolioException(Exception):
    """Base exception class for Qortfolio."""
    pass

# Database Exceptions
class DatabaseConnectionError(QortfolioException):
    """Raised when database connection fails."""
    pass

class DatabaseOperationError(QortfolioException):
    """Raised when database operation fails."""
    pass

# Data Collection Exceptions
class DataCollectionError(QortfolioException):
    """Raised when data collection fails."""
    pass

class APIConnectionError(DataCollectionError):
    """Raised when API connection fails."""
    pass

class RateLimitError(DataCollectionError):
    """Raised when API rate limit is exceeded."""
    pass

# Financial Calculation Exceptions
class CalculationError(QortfolioException):
    """Raised when financial calculation fails."""
    pass

class InvalidOptionDataError(CalculationError):
    """Raised when option data is invalid for calculations."""
    pass

class TimeCalculationError(CalculationError):
    """Raised when time-to-maturity calculation fails."""
    pass

# Validation Exceptions
class ValidationError(QortfolioException):
    """Raised when data validation fails."""
    pass

class InvalidTickerError(ValidationError):
    """Raised when ticker symbol is invalid."""
    pass

class InvalidDateRangeError(ValidationError):
    """Raised when date range is invalid."""
    pass

# Portfolio Exceptions
class PortfolioError(QortfolioException):
    """Raised when portfolio operation fails."""
    pass

class InsufficientDataError(PortfolioError):
    """Raised when insufficient data for portfolio calculations."""
    pass

class AllocationError(PortfolioError):
    """Raised when portfolio allocation is invalid."""
    pass

if __name__ == "__main__":
    # Test exception hierarchy
    print("Qortfolio Exception Hierarchy:")
    print("- QortfolioException")
    print("  - DatabaseConnectionError")
    print("  - DatabaseOperationError")
    print("  - DataCollectionError")
    print("    - APIConnectionError")
    print("    - RateLimitError")
    print("  - CalculationError")
    print("    - InvalidOptionDataError")
    print("    - TimeCalculationError")
    print("  - ValidationError")
    print("    - InvalidTickerError")
    print("    - InvalidDateRangeError")
    print("  - PortfolioError")
    print("    - InsufficientDataError")
    print("    - AllocationError")
