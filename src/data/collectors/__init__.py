# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Data collectors package initialization - Clean imports only.
"""

# Only import what actually exists
try:
    from .deribit_collector import DeribitCollector, get_deribit_collector
    __all__ = ['DeribitCollector', 'get_deribit_collector']
except ImportError:
    # If import fails, provide empty module to prevent crashes
    __all__ = []