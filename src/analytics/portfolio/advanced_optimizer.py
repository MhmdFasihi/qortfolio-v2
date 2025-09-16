# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Advanced Portfolio Optimization Module
Integrates HRP, HERC, and classic Mean-Variance optimization with crypto sector constraints
High-performance optimization using riskfolio-lib with database integration
"""

import numpy as np
import pandas as pd
import riskfolio as rp
import warnings
import asyncio
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from motor.motor_asyncio import AsyncIOMotorDatabase
from ..risk.portfolio_risk import PortfolioRiskAnalyzer
import logging

logger = logging.getLogger(__name__)

class OptimizationMethod(Enum):
    """Supported optimization methods"""
    HRP = "HRP"  # Hierarchical Risk Parity
    HERC = "HERC"  # Hierarchical Equal Risk Contribution
    MEAN_VARIANCE = "Classic"  # Classic Mean-Variance optimization
    BLACK_LITTERMAN = "BL"  # Black-Litterman
    RISK_PARITY = "RP"  # Risk Parity

class OptimizationObjective(Enum):
    """Optimization objectives"""
    SHARPE = "Sharpe"  # Maximum Sharpe ratio
    MIN_RISK = "MinRisk"  # Minimum risk
    UTILITY = "Utility"  # Maximum utility
    MAX_RETURN = "MaxRet"  # Maximum return

@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization"""
    method: OptimizationMethod = OptimizationMethod.HRP
    objective: OptimizationObjective = OptimizationObjective.SHARPE
    risk_measure: str = "MV"  # Mean-Variance
    risk_free_rate: float = 0.05
    risk_aversion: float = 2.0
    lookback_days: int = 252
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly

    # HRP/HERC specific
    codependence: str = "pearson"
    linkage_method: str = "single"
    max_clusters: int = 10
    optimize_leaf_order: bool = True

    # Constraints
    min_weight: float = 0.01  # Minimum 1% per asset
    max_weight: float = 0.40  # Maximum 40% per asset
    max_sector_weight: float = 0.60  # Maximum 60% per sector

@dataclass
class OptimizationResult:
    """Result of portfolio optimization"""
    weights: pd.DataFrame
    expected_return: float
    volatility: float
    sharpe_ratio: float
    method_used: str
    objective_used: str
    risk_metrics: Dict
    sector_allocation: Dict
    optimization_time: float
    timestamp: datetime

class AdvancedPortfolioOptimizer:
    """
    Advanced Portfolio Optimization with multiple algorithms and crypto sector constraints
    Integrates HRP, HERC, and classic optimization methods with high performance
    """

    # Enhanced crypto sectors mapping from existing config
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

    def __init__(self, db: AsyncIOMotorDatabase, config: OptimizationConfig = None):
        """
        Initialize advanced portfolio optimizer

        Args:
            db: MongoDB database connection
            config: Optimization configuration
        """
        self.db = db
        self.config = config or OptimizationConfig()
        self.returns_data = None
        self.portfolio = None
        self.risk_analyzer = PortfolioRiskAnalyzer(db)

        # Configure settings
        pd.options.display.float_format = '{:.4%}'.format
        warnings.filterwarnings("ignore")

    async def prepare_optimization_data(self,
                                      portfolio_id: str,
                                      assets: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Prepare historical returns data for optimization

        Args:
            portfolio_id: Portfolio identifier
            assets: List of assets to include (optional)

        Returns:
            DataFrame with historical returns
        """
        try:
            if assets:
                # Use provided asset list
                self.assets = assets
            else:
                # Get assets from existing portfolio
                portfolio_data = await self.db.portfolio_data.find_one(
                    {"portfolio_id": portfolio_id}
                )
                if not portfolio_data:
                    # Default to top crypto assets if no portfolio found
                    self.assets = [
                        'BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'LINK-USD',
                        'AAVE-USD', 'UNI7083-USD', 'MATIC-USD'
                    ]
                else:
                    self.assets = [pos['symbol'] for pos in portfolio_data.get('positions', [])]

            # Fetch historical price data from database
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=self.config.lookback_days)

            price_data = {}
            for asset in self.assets:
                # Query price data from database
                cursor = self.db.crypto_prices.find({
                    "symbol": asset.replace('-USD', ''),
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
                raise ValueError("No price data available for optimization")

            # Create price DataFrame and calculate returns
            price_df = pd.DataFrame(price_data)
            price_df = price_df.fillna(method='ffill').fillna(method='bfill')
            self.returns_data = price_df.pct_change().dropna()

            logger.info(f"Prepared optimization data for {len(self.assets)} assets with {len(self.returns_data)} observations")
            return self.returns_data

        except Exception as e:
            logger.error(f"Error preparing optimization data: {e}")
            # Fallback to sample data for testing
            dates = pd.date_range(end=datetime.now(), periods=self.config.lookback_days, freq='D')
            np.random.seed(42)
            returns = {}
            for asset in self.assets:
                returns[asset] = pd.Series(
                    np.random.normal(0.0008, 0.03, len(dates)),
                    index=dates
                )
            self.returns_data = pd.DataFrame(returns)
            return self.returns_data

    def _create_sector_constraints(self,
                                 sector_constraints: Optional[Dict] = None,
                                 asset_constraints: Optional[Dict] = None,
                                 constraint_type: str = "hrp") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create sector and asset constraints for optimization

        Args:
            sector_constraints: Custom sector constraints
            asset_constraints: Custom asset constraints

        Returns:
            Tuple of (asset_classes, constraints) DataFrames
        """
        # Create asset classes mapping
        assets_data = []
        for sector, tokens in self.CRYPTO_SECTORS.items():
            for token in tokens:
                if token in self.assets:
                    assets_data.append({
                        'Assets': token,
                        'Sector': sector
                    })

        # Add unmapped assets to "Other" sector
        mapped_assets = {item['Assets'] for item in assets_data}
        for asset in self.assets:
            if asset not in mapped_assets:
                assets_data.append({
                    'Assets': asset,
                    'Sector': 'Other'
                })

        asset_classes = pd.DataFrame(assets_data).sort_values(by=['Assets'])

        # Create constraints with all required columns for riskfolio-lib
        constraints_data = []

        # Default sector constraints
        default_sector_constraints = {
            'Infrastructure': {'min': 0.10, 'max': 0.50},
            'DeFi': {'max': 0.30},
            'Oracle_Infrastructure': {'max': 0.25},
            'Web3_AI': {'max': 0.20},
            'Meme': {'max': 0.15}
        }

        # Apply sector constraints
        active_sectors = asset_classes['Sector'].unique()
        final_sector_constraints = sector_constraints or default_sector_constraints

        for sector in active_sectors:
            if sector in final_sector_constraints:
                limits = final_sector_constraints[sector]

                if isinstance(limits, dict):
                    if 'min' in limits:
                        constraints_data.append({
                            'Disabled': False,
                            'Type': 'Classes',
                            'Set': 'Sector',
                            'Position': sector,
                            'Sign': '>=',
                            'Weight': limits['min'],
                            'Type Relative': '',
                            'Relative Set': '',
                            'Relative': '',
                            'Factor': ''
                        })
                    if 'max' in limits:
                        constraints_data.append({
                            'Disabled': False,
                            'Type': 'Classes',
                            'Set': 'Sector',
                            'Position': sector,
                            'Sign': '<=',
                            'Weight': limits['max'],
                            'Type Relative': '',
                            'Relative Set': '',
                            'Relative': '',
                            'Factor': ''
                        })
                else:
                    constraints_data.append({
                        'Disabled': False,
                        'Type': 'Classes',
                        'Set': 'Sector',
                        'Position': sector,
                        'Sign': '<=',
                        'Weight': limits,
                        'Type Relative': '',
                        'Relative Set': '',
                        'Relative': '',
                        'Factor': ''
                    })

        # Apply asset constraints
        if asset_constraints:
            for asset, limits in asset_constraints.items():
                if asset in self.assets:
                    if 'min' in limits:
                        constraints_data.append({
                            'Disabled': False,
                            'Type': 'Assets',
                            'Set': '',
                            'Position': asset,
                            'Sign': '>=',
                            'Weight': limits['min'],
                            'Type Relative': '',
                            'Relative Set': '',
                            'Relative': '',
                            'Factor': ''
                        })
                    if 'max' in limits:
                        constraints_data.append({
                            'Disabled': False,
                            'Type': 'Assets',
                            'Set': '',
                            'Position': asset,
                            'Sign': '<=',
                            'Weight': limits['max'],
                            'Type Relative': '',
                            'Relative Set': '',
                            'Relative': '',
                            'Factor': ''
                        })

        # Default min/max weights for all assets
        for asset in self.assets:
            constraints_data.append({
                'Disabled': False,
                'Type': 'Assets',
                'Set': '',
                'Position': asset,
                'Sign': '>=',
                'Weight': self.config.min_weight,
                'Type Relative': '',
                'Relative Set': '',
                'Relative': '',
                'Factor': ''
            })
            constraints_data.append({
                'Disabled': False,
                'Type': 'Assets',
                'Set': '',
                'Position': asset,
                'Sign': '<=',
                'Weight': self.config.max_weight,
                'Type Relative': '',
                'Relative Set': '',
                'Relative': '',
                'Factor': ''
            })

        constraints = pd.DataFrame(constraints_data)

        # Format constraints based on optimization type
        if constraint_type == "hrp":
            # HRP constraints need exactly 6 columns
            required_columns = ['Disabled', 'Type', 'Set', 'Position', 'Sign', 'Weight']
            constraints = constraints[required_columns]
        else:
            # Mean-Variance constraints need exactly 10 columns
            required_columns = ['Disabled', 'Type', 'Set', 'Position', 'Sign', 'Weight',
                              'Type Relative', 'Relative Set', 'Relative', 'Factor']
            # Ensure all columns are present
            for col in required_columns:
                if col not in constraints.columns:
                    constraints[col] = ''

        return asset_classes, constraints

    async def optimize_hrp_herc(self,
                              method: str = "HRP",
                              sector_constraints: Optional[Dict] = None,
                              asset_constraints: Optional[Dict] = None) -> OptimizationResult:
        """
        Perform HRP or HERC optimization

        Args:
            method: "HRP" or "HERC"
            sector_constraints: Sector-level constraints
            asset_constraints: Asset-level constraints

        Returns:
            OptimizationResult with weights and metrics
        """
        start_time = datetime.now()

        try:
            # Create constraints for HRP
            asset_classes, constraints = self._create_sector_constraints(
                sector_constraints, asset_constraints, "hrp"
            )

            # Initialize HCPortfolio for hierarchical methods
            self.portfolio = rp.HCPortfolio(returns=self.returns_data)

            # Calculate weight bounds from constraints
            weight_max, weight_min = rp.hrp_constraints(constraints, asset_classes)
            self.portfolio.w_max = weight_max
            self.portfolio.w_min = weight_min

            # Optimize portfolio
            optimal_weights = self.portfolio.optimization(
                model=method,
                codependence=self.config.codependence,
                rm=self.config.risk_measure,
                rf=self.config.risk_free_rate,
                linkage=self.config.linkage_method,
                max_k=self.config.max_clusters,
                leaf_order=self.config.optimize_leaf_order
            )

            # Calculate performance metrics
            portfolio_return = (self.returns_data.mean() @ optimal_weights).iloc[0] * 252
            portfolio_vol = np.sqrt(optimal_weights.T @ self.returns_data.cov() @ optimal_weights).iloc[0] * np.sqrt(252)
            sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_vol

            # Calculate sector allocation
            sector_allocation = {}
            for sector in asset_classes['Sector'].unique():
                sector_assets = asset_classes[asset_classes['Sector'] == sector]['Assets'].tolist()
                sector_weight = optimal_weights.loc[optimal_weights.index.isin(sector_assets)].sum().iloc[0]
                sector_allocation[sector] = float(sector_weight)

            # Calculate additional risk metrics
            risk_metrics = await self.risk_analyzer.calculate_portfolio_metrics(
                portfolio_id=f"optimized_{method.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                lookback_days=30
            )

            optimization_time = (datetime.now() - start_time).total_seconds()

            result = OptimizationResult(
                weights=optimal_weights,
                expected_return=float(portfolio_return),
                volatility=float(portfolio_vol),
                sharpe_ratio=float(sharpe_ratio),
                method_used=method,
                objective_used=self.config.objective.value,
                risk_metrics=risk_metrics,
                sector_allocation=sector_allocation,
                optimization_time=optimization_time,
                timestamp=datetime.now()
            )

            # Store optimization result in database
            await self._store_optimization_result(result)

            logger.info(f"Completed {method} optimization in {optimization_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error in {method} optimization: {e}")
            raise

    async def optimize_mean_variance(self,
                                   objective: str = "Sharpe",
                                   sector_constraints: Optional[Dict] = None,
                                   asset_constraints: Optional[Dict] = None) -> OptimizationResult:
        """
        Perform classic Mean-Variance optimization

        Args:
            objective: Optimization objective (Sharpe, Utility, MinRisk, MaxRet)
            sector_constraints: Sector-level constraints
            asset_constraints: Asset-level constraints

        Returns:
            OptimizationResult with weights and metrics
        """
        start_time = datetime.now()

        try:
            # Create constraints for Mean-Variance
            asset_classes, constraints = self._create_sector_constraints(
                sector_constraints, asset_constraints, "mv"
            )

            # Initialize Portfolio for mean-variance optimization
            self.portfolio = rp.Portfolio(returns=self.returns_data)

            # Calculate optimal allocation matrix and vector
            A, b = rp.assets_constraints(constraints, asset_classes)
            self.portfolio.ainequality = A
            self.portfolio.binequality = b

            # Estimate input parameters
            self.portfolio.assets_stats(method_mu='hist', method_cov='hist')

            # Optimize portfolio
            optimal_weights = self.portfolio.optimization(
                model='Classic',
                rm=self.config.risk_measure,
                obj=objective,
                rf=self.config.risk_free_rate,
                l=self.config.risk_aversion,
                hist=True
            )

            # Calculate performance metrics
            portfolio_return = (self.portfolio.mu @ optimal_weights).iloc[0] * 252
            portfolio_vol = np.sqrt(optimal_weights.T @ self.portfolio.cov @ optimal_weights).iloc[0] * np.sqrt(252)
            sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_vol

            # Calculate sector allocation
            sector_allocation = {}
            for sector in asset_classes['Sector'].unique():
                sector_assets = asset_classes[asset_classes['Sector'] == sector]['Assets'].tolist()
                sector_weight = optimal_weights.loc[optimal_weights.index.isin(sector_assets)].sum().iloc[0]
                sector_allocation[sector] = float(sector_weight)

            # Calculate additional risk metrics
            risk_metrics = await self.risk_analyzer.calculate_portfolio_metrics(
                portfolio_id=f"optimized_mv_{objective.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                lookback_days=30
            )

            optimization_time = (datetime.now() - start_time).total_seconds()

            result = OptimizationResult(
                weights=optimal_weights,
                expected_return=float(portfolio_return),
                volatility=float(portfolio_vol),
                sharpe_ratio=float(sharpe_ratio),
                method_used='Mean-Variance',
                objective_used=objective,
                risk_metrics=risk_metrics,
                sector_allocation=sector_allocation,
                optimization_time=optimization_time,
                timestamp=datetime.now()
            )

            # Store optimization result in database
            await self._store_optimization_result(result)

            logger.info(f"Completed Mean-Variance optimization in {optimization_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error in Mean-Variance optimization: {e}")
            raise

    async def run_multi_method_optimization(self,
                                          portfolio_id: str,
                                          methods: List[str] = None,
                                          sector_constraints: Optional[Dict] = None,
                                          asset_constraints: Optional[Dict] = None) -> Dict[str, OptimizationResult]:
        """
        Run multiple optimization methods and compare results

        Args:
            portfolio_id: Portfolio identifier
            methods: List of methods to run ['HRP', 'HERC', 'Sharpe', 'MinRisk']
            sector_constraints: Sector-level constraints
            asset_constraints: Asset-level constraints

        Returns:
            Dictionary of optimization results by method
        """
        if methods is None:
            methods = ['HRP', 'HERC', 'Sharpe', 'MinRisk']

        # Prepare data
        await self.prepare_optimization_data(portfolio_id)

        results = {}

        # Run hierarchical methods
        if 'HRP' in methods:
            results['HRP'] = await self.optimize_hrp_herc(
                method='HRP',
                sector_constraints=sector_constraints,
                asset_constraints=asset_constraints
            )

        if 'HERC' in methods:
            results['HERC'] = await self.optimize_hrp_herc(
                method='HERC',
                sector_constraints=sector_constraints,
                asset_constraints=asset_constraints
            )

        # Run mean-variance methods
        mv_objectives = ['Sharpe', 'MinRisk', 'Utility', 'MaxRet']
        for obj in mv_objectives:
            if obj in methods:
                results[f'MV_{obj}'] = await self.optimize_mean_variance(
                    objective=obj,
                    sector_constraints=sector_constraints,
                    asset_constraints=asset_constraints
                )

        logger.info(f"Completed multi-method optimization with {len(results)} methods")
        return results

    def generate_efficient_frontier(self, points: int = 50) -> pd.DataFrame:
        """
        Generate efficient frontier for the current portfolio

        Args:
            points: Number of points on the frontier

        Returns:
            DataFrame with efficient frontier points
        """
        if self.portfolio is None:
            raise ValueError("Portfolio must be optimized before generating efficient frontier")

        try:
            frontier = self.portfolio.efficient_frontier(
                model='Classic',
                rm=self.config.risk_measure,
                points=points,
                rf=self.config.risk_free_rate,
                hist=True
            )

            return frontier

        except Exception as e:
            logger.error(f"Error generating efficient frontier: {e}")
            raise

    async def _store_optimization_result(self, result: OptimizationResult) -> str:
        """
        Store optimization result in database

        Args:
            result: OptimizationResult to store

        Returns:
            Document ID of stored result
        """
        try:
            # Convert weights to dictionary for storage
            weights_dict = result.weights.to_dict()

            doc = {
                'weights': weights_dict,
                'expected_return': result.expected_return,
                'volatility': result.volatility,
                'sharpe_ratio': result.sharpe_ratio,
                'method_used': result.method_used,
                'objective_used': result.objective_used,
                'risk_metrics': result.risk_metrics,
                'sector_allocation': result.sector_allocation,
                'optimization_time': result.optimization_time,
                'timestamp': result.timestamp,
                'config': {
                    'risk_measure': self.config.risk_measure,
                    'risk_free_rate': self.config.risk_free_rate,
                    'lookback_days': self.config.lookback_days
                }
            }

            insert_result = await self.db.optimization_results.insert_one(doc)
            logger.info(f"Stored optimization result with ID: {insert_result.inserted_id}")
            return str(insert_result.inserted_id)

        except Exception as e:
            logger.error(f"Error storing optimization result: {e}")
            return ""

    async def get_optimization_history(self,
                                     portfolio_id: Optional[str] = None,
                                     method: Optional[str] = None,
                                     limit: int = 10) -> List[Dict]:
        """
        Get historical optimization results

        Args:
            portfolio_id: Filter by portfolio ID
            method: Filter by optimization method
            limit: Maximum number of results

        Returns:
            List of optimization results
        """
        try:
            query = {}
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