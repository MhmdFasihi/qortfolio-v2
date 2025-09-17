# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Portfolio Optimization Service Layer
Interface between frontend and backend optimization models
Handles all business logic and data transformation
"""

from typing import Dict, List, Optional
import asyncio
import logging

from motor.motor_asyncio import AsyncIOMotorDatabase
from ..models.portfolio import (
    PortfolioOptimizationModel,
    OptimizationMethod,
    OptimizationObjective,
    OptimizationConfig,
    OptimizationResult
)
from ..core.database.connection import get_database_async

logger = logging.getLogger(__name__)

class PortfolioOptimizationService:
    """
    Service layer for portfolio optimization
    Handles all business logic between frontend and backend models
    """

    def __init__(self):
        self.db = None
        self.optimization_model = None

    async def initialize(self):
        """Initialize service with database connection"""
        if self.db is None:
            self.db = await get_database_async()
            self.optimization_model = PortfolioOptimizationModel(self.db)

    async def run_single_optimization(self,
                                    portfolio_id: str,
                                    assets: List[str],
                                    method: str,
                                    risk_free_rate: float = 0.05,
                                    risk_aversion: float = 2.0,
                                    lookback_days: int = 365,
                                    min_weight: float = 0.01,
                                    max_weight: float = 0.40,
                                    sector_constraints: Optional[Dict] = None,
                                    asset_constraints: Optional[Dict] = None) -> Dict:
        """
        Run single portfolio optimization

        Args:
            portfolio_id: Portfolio identifier
            assets: List of assets to optimize
            method: Optimization method (HRP, HERC, Sharpe, etc.)
            risk_free_rate: Risk-free rate
            risk_aversion: Risk aversion parameter
            lookback_days: Historical data lookback period
            min_weight: Minimum asset weight
            max_weight: Maximum asset weight
            sector_constraints: Sector-level constraints
            asset_constraints: Asset-level constraints

        Returns:
            Dictionary with optimization results formatted for UI
        """
        try:
            await self.initialize()

            # Create optimization configuration
            config = OptimizationConfig(
                risk_free_rate=risk_free_rate,
                risk_aversion=risk_aversion,
                lookback_days=lookback_days,
                min_weight=min_weight,
                max_weight=max_weight
            )

            # Run optimization based on method
            if method in ["HRP", "HERC"]:
                result = await self.optimization_model.optimize_hrp_herc(
                    portfolio_id=portfolio_id,
                    assets=assets,
                    config=config,
                    method=method,
                    sector_constraints=sector_constraints,
                    asset_constraints=asset_constraints
                )
            else:
                result = await self.optimization_model.optimize_mean_variance(
                    portfolio_id=portfolio_id,
                    assets=assets,
                    config=config,
                    objective=method,
                    sector_constraints=sector_constraints,
                    asset_constraints=asset_constraints
                )

            # Format result for UI
            return self._format_result_for_ui(result)

        except Exception as e:
            logger.error(f"Error in single optimization: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'method_used': method,
                'weights': {},
                'expected_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'sector_allocation': {},
                'optimization_time': 0.0
            }

    async def run_multi_method_comparison(self,
                                        portfolio_id: str,
                                        assets: List[str],
                                        methods: List[str] = None,
                                        risk_free_rate: float = 0.05,
                                        lookback_days: int = 365,
                                        sector_constraints: Optional[Dict] = None,
                                        asset_constraints: Optional[Dict] = None) -> List[Dict]:
        """
        Run multiple optimization methods and compare results

        Args:
            portfolio_id: Portfolio identifier
            assets: List of assets to optimize
            methods: List of methods to compare
            risk_free_rate: Risk-free rate
            lookback_days: Historical data lookback period
            sector_constraints: Sector-level constraints
            asset_constraints: Asset-level constraints

        Returns:
            List of optimization results formatted for UI comparison
        """
        try:
            await self.initialize()

            if methods is None:
                methods = ['HRP', 'HERC', 'Sharpe', 'MinRisk']

            config = OptimizationConfig(
                risk_free_rate=risk_free_rate,
                lookback_days=lookback_days
            )

            # Run multi-method comparison
            results = await self.optimization_model.run_multi_method_comparison(
                portfolio_id=portfolio_id,
                assets=assets,
                config=config,
                methods=methods,
                sector_constraints=sector_constraints,
                asset_constraints=asset_constraints
            )

            # Format results for UI comparison table
            comparison_data = []
            for method_name, result in results.items():
                comparison_data.append({
                    "method": result.method_used,
                    "expected_return": f"{result.expected_return:.2%}",
                    "volatility": f"{result.volatility:.2%}",
                    "sharpe_ratio": f"{result.sharpe_ratio:.3f}",
                    "optimization_time": f"{result.optimization_time:.2f}s",
                    "success": result.success
                })

            return comparison_data

        except Exception as e:
            logger.error(f"Error in multi-method comparison: {e}")
            return []

    async def get_portfolio_assets(self, portfolio_id: str) -> List[str]:
        """
        Get assets from existing portfolio

        Args:
            portfolio_id: Portfolio identifier

        Returns:
            List of asset symbols
        """
        try:
            await self.initialize()

            portfolio_data = await self.db.portfolio_data.find_one(
                {"portfolio_id": portfolio_id}
            )

            if portfolio_data and 'positions' in portfolio_data:
                assets = []
                for position in portfolio_data['positions']:
                    symbol = position.get('symbol', '')
                    if symbol:
                        # Ensure USD suffix for consistency
                        if not symbol.endswith('-USD'):
                            symbol += '-USD'
                        assets.append(symbol)
                return assets
            else:
                # Default crypto assets
                return ['BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'LINK-USD', 'AAVE-USD']

        except Exception as e:
            logger.error(f"Error getting portfolio assets: {e}")
            return ['BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'LINK-USD', 'AAVE-USD']

    async def generate_efficient_frontier(self,
                                        portfolio_id: str,
                                        assets: List[str],
                                        points: int = 20) -> List[Dict]:
        """
        Generate efficient frontier data

        Args:
            portfolio_id: Portfolio identifier
            assets: List of assets
            points: Number of frontier points

        Returns:
            List of risk-return points for UI chart
        """
        try:
            await self.initialize()

            config = OptimizationConfig(lookback_days=365)
            returns_data = await self.optimization_model.prepare_market_data(
                portfolio_id, assets, config.lookback_days
            )

            # Use riskfolio-lib to generate frontier
            import riskfolio as rp
            portfolio = rp.Portfolio(returns=returns_data)
            portfolio.assets_stats(method_mu='hist', method_cov='hist')

            frontier = portfolio.efficient_frontier(
                model='Classic',
                rm='MV',
                points=points,
                rf=config.risk_free_rate,
                hist=True
            )

            # Format for UI chart
            frontier_data = []
            for i in range(len(frontier)):
                ret = frontier.iloc[i]['Mean']
                vol = frontier.iloc[i]['Volatility']
                frontier_data.append({
                    "risk": float(vol),
                    "return": float(ret)
                })

            return frontier_data

        except Exception as e:
            logger.error(f"Error generating efficient frontier: {e}")
            return []

    async def get_optimization_history(self,
                                     portfolio_id: Optional[str] = None,
                                     method: Optional[str] = None,
                                     limit: int = 10) -> List[Dict]:
        """
        Get optimization history for UI display

        Args:
            portfolio_id: Optional portfolio filter
            method: Optional method filter
            limit: Maximum number of results

        Returns:
            List of historical optimization results
        """
        try:
            await self.initialize()
            return await self.optimization_model.get_optimization_history(
                portfolio_id, method, limit
            )

        except Exception as e:
            logger.error(f"Error getting optimization history: {e}")
            return []

    def _format_result_for_ui(self, result: OptimizationResult) -> Dict:
        """
        Format optimization result for UI consumption

        Args:
            result: OptimizationResult from backend model

        Returns:
            Dictionary formatted for UI state
        """
        # Convert weights to list format for UI table
        suggested_allocation = [
            {"asset": asset.replace("-USD", ""), "weight": float(weight) * 100}
            for asset, weight in result.weights.items()
            if float(weight) > 0.005  # Show only weights > 0.5%
        ]

        return {
            'success': result.success,
            'method_used': result.method_used,
            'objective_used': result.objective_used,
            'expected_return': result.expected_return,
            'volatility': result.volatility,
            'sharpe_ratio': result.sharpe_ratio,
            'sector_allocation': result.sector_allocation,
            'optimization_time': result.optimization_time,
            'timestamp': result.timestamp.isoformat() if result.timestamp else '',
            'suggested_allocation': suggested_allocation,
            'error_message': result.error_message
        }

# Global service instance
_optimization_service = None

async def get_optimization_service() -> PortfolioOptimizationService:
    """Get global optimization service instance"""
    global _optimization_service
    if _optimization_service is None:
        _optimization_service = PortfolioOptimizationService()
        await _optimization_service.initialize()
    return _optimization_service