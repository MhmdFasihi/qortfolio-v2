# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Portfolio Optimization Models - Pure Backend Business Logic
All optimization calculations happen here, no UI dependencies
"""

import numpy as np
import pandas as pd
import riskfolio as rp
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging

from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

class OptimizationMethod(Enum):
    """Supported optimization methods"""
    HRP = "HRP"
    HERC = "HERC"
    MEAN_VARIANCE = "Classic"
    BLACK_LITTERMAN = "BL"
    RISK_PARITY = "RP"

class OptimizationObjective(Enum):
    """Optimization objectives"""
    SHARPE = "Sharpe"
    MIN_RISK = "MinRisk"
    UTILITY = "Utility"
    MAX_RETURN = "MaxRet"

@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization"""
    method: OptimizationMethod = OptimizationMethod.HRP
    objective: OptimizationObjective = OptimizationObjective.SHARPE
    risk_measure: str = "MV"
    risk_free_rate: float = 0.05
    risk_aversion: float = 2.0
    lookback_days: int = 252

    # HRP/HERC specific
    codependence: str = "pearson"
    linkage_method: str = "single"
    max_clusters: int = 10
    optimize_leaf_order: bool = True

    # Constraints
    min_weight: float = 0.01
    max_weight: float = 0.40
    max_sector_weight: float = 0.60

@dataclass
class OptimizationResult:
    """Result of portfolio optimization"""
    portfolio_id: str
    method_used: str
    objective_used: str
    weights: Dict[str, float]  # Asset -> Weight mapping
    expected_return: float
    volatility: float
    sharpe_ratio: float
    sector_allocation: Dict[str, float]
    risk_metrics: Dict
    optimization_time: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None

class PortfolioOptimizationModel:
    """
    Pure backend model for portfolio optimization
    Contains all business logic, no UI dependencies
    """

    # Crypto sectors mapping
    CRYPTO_SECTORS = {
        'Infrastructure': ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'AVAX-USD', 'NEAR-USD'],
        'DeFi': ['AAVE-USD', 'UNI7083-USD', 'LDO-USD', 'CRV-USD', 'MKR-USD', 'COMP5692-USD'],
        'Oracle_Infrastructure': ['LINK-USD', 'HNT-USD', 'MATIC-USD', 'DOT-USD'],
        'Web3_AI': ['ICP-USD', 'FET-USD', 'RLC-USD', 'RENDER-USD', 'TAO22974-USD'],
        'Exchange_Trading': ['XRP-USD', 'ONDO-USD', 'DYDX-USD'],
        'Layer2_Scaling': ['MATIC-USD', 'OP-USD', 'ARB11841-USD'],
        'Gaming_Metaverse': ['SAND-USD', 'MANA-USD', 'AXS-USD', 'GALA-USD'],
        'Meme': ['DOGE-USD', 'SHIB-USD', 'PEPE24478-USD', 'WIF-USD', 'BONK-USD'],
        'Privacy': ['XMR-USD', 'ZEC-USD', 'DASH-USD'],
        'Storage_Compute': ['FIL-USD', 'AR-USD', 'STORJ-USD'],
        'Enterprise_Adoption': ['XAUT-USD', 'USDC-USD', 'USDT-USD'],
        'Cross_Chain': ['ATOM-USD', 'DOT-USD', 'COSMOS-USD']
    }

    def __init__(self, db: AsyncIOMotorDatabase):
        """Initialize portfolio optimization model"""
        self.db = db

    async def prepare_market_data(self,
                                 portfolio_id: str,
                                 assets: List[str],
                                 lookback_days: int) -> pd.DataFrame:
        """
        Prepare historical returns data from database

        Args:
            portfolio_id: Portfolio identifier
            assets: List of assets to include
            lookback_days: Number of days to look back

        Returns:
            DataFrame with historical returns
        """
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=lookback_days)

            price_data = {}
            for asset in assets:
                symbol = asset.replace('-USD', '')
                cursor = self.db.crypto_prices.find({
                    "symbol": symbol,
                    "timestamp": {"$gte": start_date, "$lte": end_date}
                }).sort("timestamp", 1)

                prices = []
                timestamps = []
                async for doc in cursor:
                    prices.append(doc['price_usd'])
                    timestamps.append(doc['timestamp'])

                if prices:
                    price_series = pd.Series(prices, index=timestamps)
                    price_data[asset] = price_series

            if not price_data:
                # Fallback to sample data for testing
                return self._generate_sample_data(assets, lookback_days)

            # Create returns DataFrame
            price_df = pd.DataFrame(price_data).fillna(method='ffill').fillna(method='bfill')
            returns_df = price_df.pct_change().dropna()

            logger.info(f"Prepared market data for {len(assets)} assets with {len(returns_df)} observations")
            return returns_df

        except Exception as e:
            logger.error(f"Error preparing market data: {e}")
            # Fallback to sample data
            return self._generate_sample_data(assets, lookback_days)

    def _generate_sample_data(self, assets: List[str], n_days: int) -> pd.DataFrame:
        """Generate sample data for testing"""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

        returns = {}
        for asset in assets:
            # Realistic crypto returns
            daily_vol = 0.04 if 'BTC' in asset else 0.06
            daily_return = 0.0008 if 'BTC' in asset else 0.001
            returns[asset] = pd.Series(
                np.random.normal(daily_return, daily_vol, n_days),
                index=dates
            )

        return pd.DataFrame(returns)

    def _create_asset_classes_mapping(self, assets: List[str]) -> pd.DataFrame:
        """Create asset classes mapping for constraints"""
        assets_data = []

        for sector, tokens in self.CRYPTO_SECTORS.items():
            for token in tokens:
                if token in assets:
                    assets_data.append({'Assets': token, 'Sector': sector})

        # Add unmapped assets to "Other" sector
        mapped_assets = {item['Assets'] for item in assets_data}
        for asset in assets:
            if asset not in mapped_assets:
                assets_data.append({'Assets': asset, 'Sector': 'Other'})

        return pd.DataFrame(assets_data).sort_values(by=['Assets'])

    def _create_optimization_constraints(self,
                                       assets: List[str],
                                       asset_classes: pd.DataFrame,
                                       config: OptimizationConfig,
                                       sector_constraints: Optional[Dict] = None,
                                       asset_constraints: Optional[Dict] = None,
                                       constraint_type: str = "hrp") -> pd.DataFrame:
        """Create optimization constraints"""
        constraints_data = []

        # Default sector constraints
        default_sector_constraints = {
            'Infrastructure': {'min': 0.10, 'max': 0.50},
            'DeFi': {'max': 0.30},
            'Oracle_Infrastructure': {'max': 0.25},
            'Web3_AI': {'max': 0.20},
            'Meme': {'max': 0.15}
        }

        active_sectors = asset_classes['Sector'].unique()
        final_sector_constraints = sector_constraints or default_sector_constraints

        # Add sector constraints
        for sector in active_sectors:
            if sector in final_sector_constraints:
                limits = final_sector_constraints[sector]

                if isinstance(limits, dict):
                    if 'min' in limits:
                        constraint = {
                            'Disabled': False,
                            'Type': 'Classes',
                            'Set': 'Sector',
                            'Position': sector,
                            'Sign': '>=',
                            'Weight': limits['min']
                        }
                        if constraint_type == "mv":
                            constraint.update({
                                'Type Relative': '', 'Relative Set': '',
                                'Relative': '', 'Factor': ''
                            })
                        constraints_data.append(constraint)

                    if 'max' in limits:
                        constraint = {
                            'Disabled': False,
                            'Type': 'Classes',
                            'Set': 'Sector',
                            'Position': sector,
                            'Sign': '<=',
                            'Weight': limits['max']
                        }
                        if constraint_type == "mv":
                            constraint.update({
                                'Type Relative': '', 'Relative Set': '',
                                'Relative': '', 'Factor': ''
                            })
                        constraints_data.append(constraint)

        # Add asset constraints
        if asset_constraints:
            for asset, limits in asset_constraints.items():
                if asset in assets:
                    for sign, limit_key in [('>=', 'min'), ('<=', 'max')]:
                        if limit_key in limits:
                            constraint = {
                                'Disabled': False,
                                'Type': 'Assets',
                                'Set': '',
                                'Position': asset,
                                'Sign': sign,
                                'Weight': limits[limit_key]
                            }
                            if constraint_type == "mv":
                                constraint.update({
                                    'Type Relative': '', 'Relative Set': '',
                                    'Relative': '', 'Factor': ''
                                })
                            constraints_data.append(constraint)

        # Default min/max weights for all assets
        for asset in assets:
            for sign, weight in [('>=', config.min_weight), ('<=', config.max_weight)]:
                constraint = {
                    'Disabled': False,
                    'Type': 'Assets',
                    'Set': '',
                    'Position': asset,
                    'Sign': sign,
                    'Weight': weight
                }
                if constraint_type == "mv":
                    constraint.update({
                        'Type Relative': '', 'Relative Set': '',
                        'Relative': '', 'Factor': ''
                    })
                constraints_data.append(constraint)

        constraints = pd.DataFrame(constraints_data)

        # Format based on constraint type
        if constraint_type == "hrp":
            required_columns = ['Disabled', 'Type', 'Set', 'Position', 'Sign', 'Weight']
            constraints = constraints[required_columns]

        return constraints

    async def optimize_hrp_herc(self,
                              portfolio_id: str,
                              assets: List[str],
                              config: OptimizationConfig,
                              method: str = "HRP",
                              sector_constraints: Optional[Dict] = None,
                              asset_constraints: Optional[Dict] = None) -> OptimizationResult:
        """
        Perform HRP or HERC optimization

        Args:
            portfolio_id: Portfolio identifier
            assets: List of assets to optimize
            config: Optimization configuration
            method: "HRP" or "HERC"
            sector_constraints: Sector-level constraints
            asset_constraints: Asset-level constraints

        Returns:
            OptimizationResult with weights and metrics
        """
        start_time = datetime.now()

        try:
            # Prepare data
            returns_data = await self.prepare_market_data(portfolio_id, assets, config.lookback_days)
            asset_classes = self._create_asset_classes_mapping(assets)
            constraints = self._create_optimization_constraints(
                assets, asset_classes, config, sector_constraints, asset_constraints, "hrp"
            )

            # Optimize using riskfolio-lib
            portfolio = rp.HCPortfolio(returns=returns_data)
            weight_max, weight_min = rp.hrp_constraints(constraints, asset_classes)
            portfolio.w_max = weight_max
            portfolio.w_min = weight_min

            optimal_weights = portfolio.optimization(
                model=method,
                codependence=config.codependence,
                rm=config.risk_measure,
                rf=config.risk_free_rate,
                linkage=config.linkage_method,
                max_k=config.max_clusters,
                leaf_order=config.optimize_leaf_order
            )

            # Calculate metrics
            portfolio_return = (returns_data.mean() @ optimal_weights).iloc[0] * 252
            portfolio_vol = np.sqrt(optimal_weights.T @ returns_data.cov() @ optimal_weights).iloc[0] * np.sqrt(252)
            sharpe_ratio = (portfolio_return - config.risk_free_rate) / portfolio_vol

            # Calculate sector allocation
            sector_allocation = {}
            for sector in asset_classes['Sector'].unique():
                sector_assets = asset_classes[asset_classes['Sector'] == sector]['Assets'].tolist()
                sector_weight = optimal_weights.loc[optimal_weights.index.isin(sector_assets)].sum().iloc[0]
                sector_allocation[sector] = float(sector_weight)

            # Store result in database
            result = OptimizationResult(
                portfolio_id=portfolio_id,
                method_used=method,
                objective_used=config.objective.value,
                weights=optimal_weights.iloc[:, 0].to_dict(),
                expected_return=float(portfolio_return),
                volatility=float(portfolio_vol),
                sharpe_ratio=float(sharpe_ratio),
                sector_allocation=sector_allocation,
                risk_metrics={},
                optimization_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now(),
                success=True
            )

            await self._store_optimization_result(result)
            logger.info(f"Completed {method} optimization for {portfolio_id}")
            return result

        except Exception as e:
            logger.error(f"Error in {method} optimization: {e}")
            return OptimizationResult(
                portfolio_id=portfolio_id,
                method_used=method,
                objective_used=config.objective.value,
                weights={},
                expected_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                sector_allocation={},
                risk_metrics={},
                optimization_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )

    async def optimize_mean_variance(self,
                                   portfolio_id: str,
                                   assets: List[str],
                                   config: OptimizationConfig,
                                   objective: str = "Sharpe",
                                   sector_constraints: Optional[Dict] = None,
                                   asset_constraints: Optional[Dict] = None) -> OptimizationResult:
        """
        Perform Mean-Variance optimization

        Args:
            portfolio_id: Portfolio identifier
            assets: List of assets to optimize
            config: Optimization configuration
            objective: Optimization objective
            sector_constraints: Sector-level constraints
            asset_constraints: Asset-level constraints

        Returns:
            OptimizationResult with weights and metrics
        """
        start_time = datetime.now()

        try:
            # Prepare data
            returns_data = await self.prepare_market_data(portfolio_id, assets, config.lookback_days)
            asset_classes = self._create_asset_classes_mapping(assets)
            constraints = self._create_optimization_constraints(
                assets, asset_classes, config, sector_constraints, asset_constraints, "mv"
            )

            # Optimize using riskfolio-lib
            portfolio = rp.Portfolio(returns=returns_data)
            A, b = rp.assets_constraints(constraints, asset_classes)
            portfolio.ainequality = A
            portfolio.binequality = b

            portfolio.assets_stats(method_mu='hist', method_cov='hist')

            optimal_weights = portfolio.optimization(
                model='Classic',
                rm=config.risk_measure,
                obj=objective,
                rf=config.risk_free_rate,
                l=config.risk_aversion,
                hist=True
            )

            # Calculate metrics
            portfolio_return = (portfolio.mu @ optimal_weights).iloc[0] * 252
            portfolio_vol = np.sqrt(optimal_weights.T @ portfolio.cov @ optimal_weights).iloc[0] * np.sqrt(252)
            sharpe_ratio = (portfolio_return - config.risk_free_rate) / portfolio_vol

            # Calculate sector allocation
            sector_allocation = {}
            for sector in asset_classes['Sector'].unique():
                sector_assets = asset_classes[asset_classes['Sector'] == sector]['Assets'].tolist()
                sector_weight = optimal_weights.loc[optimal_weights.index.isin(sector_assets)].sum().iloc[0]
                sector_allocation[sector] = float(sector_weight)

            result = OptimizationResult(
                portfolio_id=portfolio_id,
                method_used='Mean-Variance',
                objective_used=objective,
                weights=optimal_weights.iloc[:, 0].to_dict(),
                expected_return=float(portfolio_return),
                volatility=float(portfolio_vol),
                sharpe_ratio=float(sharpe_ratio),
                sector_allocation=sector_allocation,
                risk_metrics={},
                optimization_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now(),
                success=True
            )

            await self._store_optimization_result(result)
            logger.info(f"Completed Mean-Variance optimization for {portfolio_id}")
            return result

        except Exception as e:
            logger.error(f"Error in Mean-Variance optimization: {e}")
            return OptimizationResult(
                portfolio_id=portfolio_id,
                method_used='Mean-Variance',
                objective_used=objective,
                weights={},
                expected_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                sector_allocation={},
                risk_metrics={},
                optimization_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )

    async def run_multi_method_comparison(self,
                                        portfolio_id: str,
                                        assets: List[str],
                                        config: OptimizationConfig,
                                        methods: List[str] = None,
                                        sector_constraints: Optional[Dict] = None,
                                        asset_constraints: Optional[Dict] = None) -> Dict[str, OptimizationResult]:
        """
        Run multiple optimization methods and compare results

        Args:
            portfolio_id: Portfolio identifier
            assets: List of assets to optimize
            config: Optimization configuration
            methods: List of methods to run
            sector_constraints: Sector-level constraints
            asset_constraints: Asset-level constraints

        Returns:
            Dictionary of optimization results by method
        """
        if methods is None:
            methods = ['HRP', 'HERC', 'Sharpe', 'MinRisk']

        results = {}

        # Run hierarchical methods
        if 'HRP' in methods:
            results['HRP'] = await self.optimize_hrp_herc(
                portfolio_id, assets, config, 'HRP', sector_constraints, asset_constraints
            )

        if 'HERC' in methods:
            results['HERC'] = await self.optimize_hrp_herc(
                portfolio_id, assets, config, 'HERC', sector_constraints, asset_constraints
            )

        # Run mean-variance methods
        mv_objectives = ['Sharpe', 'MinRisk', 'Utility', 'MaxRet']
        for obj in mv_objectives:
            if obj in methods:
                results[f'MV_{obj}'] = await self.optimize_mean_variance(
                    portfolio_id, assets, config, obj, sector_constraints, asset_constraints
                )

        logger.info(f"Completed multi-method comparison with {len(results)} methods")
        return results

    async def _store_optimization_result(self, result: OptimizationResult) -> str:
        """Store optimization result in database"""
        try:
            doc = {
                'portfolio_id': result.portfolio_id,
                'method_used': result.method_used,
                'objective_used': result.objective_used,
                'weights': result.weights,
                'expected_return': result.expected_return,
                'volatility': result.volatility,
                'sharpe_ratio': result.sharpe_ratio,
                'sector_allocation': result.sector_allocation,
                'risk_metrics': result.risk_metrics,
                'optimization_time': result.optimization_time,
                'timestamp': result.timestamp,
                'success': result.success,
                'error_message': result.error_message
            }

            insert_result = await self.db.optimization_results.insert_one(doc)
            return str(insert_result.inserted_id)

        except Exception as e:
            logger.error(f"Error storing optimization result: {e}")
            return ""

    async def get_optimization_history(self,
                                     portfolio_id: Optional[str] = None,
                                     method: Optional[str] = None,
                                     limit: int = 10) -> List[Dict]:
        """Get optimization history from database"""
        try:
            query = {}
            if portfolio_id:
                query['portfolio_id'] = portfolio_id
            if method:
                query['method_used'] = method

            cursor = self.db.optimization_results.find(query).sort("timestamp", -1).limit(limit)
            results = []
            async for doc in cursor:
                doc['_id'] = str(doc['_id'])
                results.append(doc)

            return results

        except Exception as e:
            logger.error(f"Error getting optimization history: {e}")
            return []