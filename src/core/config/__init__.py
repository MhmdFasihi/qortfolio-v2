# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.

"""Configuration package for Qortfolio V2.

Provides lazy access to the global `config` to avoid circular imports during
module initialization and exposes crypto sector utilities.
"""

from ..crypto.crypto_sectors import crypto_sectors, CryptoSectorsManager


class _ConfigProxy:
    """Lazy proxy for src.core.settings.config.

    This defers importing `src.core.settings` until the proxy is used,
    preventing circular import issues when modules import `src.core.config`.
    """

    def __getattr__(self, name):
        from ..settings import config as _real_config
        return getattr(_real_config, name)

    def __repr__(self) -> str:
        try:
            from ..settings import config as _real_config
            return f"ConfigProxy({repr(_real_config)})"
        except Exception:
            return "ConfigProxy(<uninitialized>)"


# Re-export a proxy named `config` for compatibility
config = _ConfigProxy()

__all__ = ['config', 'crypto_sectors', 'CryptoSectorsManager']
