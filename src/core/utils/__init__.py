# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
src/core/utils/__init__.py
Core utilities package initialization.
"""

from .time_utils import (
    calculate_time_to_maturity,
    calculate_time_to_maturity_vectorized,
    fix_legacy_time_calculation,
    validate_time_calculation,
    get_business_days_to_maturity,
    SECONDS_PER_YEAR,
    DAYS_PER_YEAR,
    TimeCalculationError
)

__all__ = [
    'calculate_time_to_maturity',
    'calculate_time_to_maturity_vectorized', 
    'fix_legacy_time_calculation',
    'validate_time_calculation',
    'get_business_days_to_maturity',
    'SECONDS_PER_YEAR',
    'DAYS_PER_YEAR',
    'TimeCalculationError'
]