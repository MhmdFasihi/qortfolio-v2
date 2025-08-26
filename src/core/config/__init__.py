# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.

"""Configuration package for Qortfolio V2.

This package exposes crypto sector configuration and re-exports the main
`config` object from `src.core.settings` to avoid name conflicts.
"""

from ..settings import config  # re-export config instance
from ..crypto.crypto_sectors import crypto_sectors, CryptoSectorsManager

__all__ = ['config', 'crypto_sectors', 'CryptoSectorsManager']
