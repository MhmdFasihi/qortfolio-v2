# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
src/core/__init__.py
Core package initialization.
"""

# Import key utilities that are commonly used
from .utils.time_utils import (
    calculate_time_to_maturity,
    SECONDS_PER_YEAR,
    TimeCalculationError
)

__version__ = "0.1.0"

__all__ = [
    'calculate_time_to_maturity',
    'SECONDS_PER_YEAR', 
    'TimeCalculationError'
]