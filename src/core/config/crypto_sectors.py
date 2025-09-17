# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Cryptocurrency sectors configuration for portfolio analysis.
Maps crypto assets to their respective sectors for attribution analysis.
"""

CRYPTO_SECTORS = {
    "DeFi": {
        "tickers": ["UNI", "AAVE", "CRV", "MKR", "COMP", "SNX", "SUSHI", "1INCH", "YFI", "BAL"],
        "description": "Decentralized Finance protocols and governance tokens",
        "market_cap_tier": "mixed"
    },
    "Infrastructure": {
        "tickers": ["BTC", "ETH", "SOL", "ADA", "AVAX", "LINK", "DOT", "ATOM", "NEAR", "FTM", "MATIC"],
        "description": "Blockchain infrastructure and smart contract platforms",
        "market_cap_tier": "large"
    },
    "AI": {
        "tickers": ["NEAR", "FET", "RLC", "RENDER", "TAO", "OCEAN", "AGI", "NMR", "GRT"],
        "description": "Artificial Intelligence and machine learning focused projects",
        "market_cap_tier": "mixed"
    },
    "Gaming": {
        "tickers": ["AXS", "SAND", "MANA", "ENJ", "GALA", "IMX", "GMT", "STEPN", "APE"],
        "description": "Gaming, metaverse and NFT platforms",
        "market_cap_tier": "mixed"
    },
    "Layer2": {
        "tickers": ["MATIC", "LRC", "IMX", "METIS", "ARB", "OP"],
        "description": "Layer 2 scaling solutions and rollups",
        "market_cap_tier": "mixed"
    },
    "Privacy": {
        "tickers": ["XMR", "ZEC", "DASH", "SCRT", "ROSE"],
        "description": "Privacy-focused cryptocurrencies",
        "market_cap_tier": "mixed"
    },
    "Storage": {
        "tickers": ["FIL", "AR", "STORJ", "SIA"],
        "description": "Decentralized storage solutions",
        "market_cap_tier": "mixed"
    },
    "Oracle": {
        "tickers": ["LINK", "BAND", "TRB", "API3"],
        "description": "Oracle and data feed providers",
        "market_cap_tier": "mixed"
    },
    "Exchange": {
        "tickers": ["BNB", "CRO", "FTT", "HT", "KCS"],
        "description": "Exchange tokens and centralized exchange ecosystems",
        "market_cap_tier": "large"
    },
    "Payments": {
        "tickers": ["LTC", "BCH", "XRP", "XLM", "DASH", "ZEC"],
        "description": "Digital payments and remittance focused cryptocurrencies",
        "market_cap_tier": "large"
    },
    "Meme": {
        "tickers": ["DOGE", "SHIB", "PEPE", "FLOKI", "BONK"],
        "description": "Meme coins and community-driven tokens",
        "market_cap_tier": "mixed"
    },
    "Stablecoin": {
        "tickers": ["USDT", "USDT-USD", "USDC", "USDC-USD", "DAI", "BUSD", "TUSD", "FRAX"],
        "description": "Stablecoins pegged to fiat currencies",
        "market_cap_tier": "large"
    }
}

# Market cap tiers for risk analysis
MARKET_CAP_TIERS = {
    "large": {
        "min_market_cap": 10_000_000_000,  # $10B+
        "risk_multiplier": 1.0
    },
    "medium": {
        "min_market_cap": 1_000_000_000,   # $1B - $10B
        "risk_multiplier": 1.5
    },
    "small": {
        "min_market_cap": 100_000_000,     # $100M - $1B
        "risk_multiplier": 2.0
    },
    "micro": {
        "min_market_cap": 0,               # <$100M
        "risk_multiplier": 3.0
    }
}

def get_asset_sector(asset: str) -> str:
    """
    Get the sector for a given cryptocurrency asset.

    Args:
        asset: Cryptocurrency ticker symbol

    Returns:
        Sector name or 'Other' if not found
    """
    for sector_name, sector_data in CRYPTO_SECTORS.items():
        if asset.upper() in [ticker.upper() for ticker in sector_data["tickers"]]:
            return sector_name
    return "Other"

def get_sector_assets(sector: str) -> list:
    """
    Get all assets in a given sector.

    Args:
        sector: Sector name

    Returns:
        List of asset tickers in the sector
    """
    return CRYPTO_SECTORS.get(sector, {}).get("tickers", [])

def get_portfolio_sector_allocation(portfolio_weights: dict) -> dict:
    """
    Calculate sector allocation for a portfolio.

    Args:
        portfolio_weights: Dict of {asset: weight}

    Returns:
        Dict of {sector: total_weight}
    """
    sector_allocation = {}

    for asset, weight in portfolio_weights.items():
        sector = get_asset_sector(asset)
        sector_allocation[sector] = sector_allocation.get(sector, 0) + weight

    return sector_allocation

def get_sector_risk_multiplier(sector: str) -> float:
    """
    Get risk multiplier for a sector based on typical market cap tier.

    Args:
        sector: Sector name

    Returns:
        Risk multiplier factor
    """
    sector_data = CRYPTO_SECTORS.get(sector, {})
    tier = sector_data.get("market_cap_tier", "mixed")

    if tier == "large":
        return MARKET_CAP_TIERS["large"]["risk_multiplier"]
    elif tier == "medium":
        return MARKET_CAP_TIERS["medium"]["risk_multiplier"]
    elif tier == "small":
        return MARKET_CAP_TIERS["small"]["risk_multiplier"]
    else:  # mixed or unknown
        return 1.75  # Average between medium and small cap risk