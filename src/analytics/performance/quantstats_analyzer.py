# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Performance Analytics using QuantStats
Comprehensive performance analysis and reporting for cryptocurrency portfolios
"""

from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging
import io
import base64

import quantstats as qs
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

class QuantStatsAnalyzer:
    """
    Professional performance analysis using QuantStats
    Comprehensive performance metrics and reporting for crypto portfolios
    """

    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize QuantStats analyzer with database connection

        Args:
            db: MongoDB database connection
        """
        self.db = db

    async def generate_performance_report(
        self,
        portfolio_id: str,
        benchmark_symbol: str = "BTC",
        lookback_days: int = 252,
        include_tearsheet: bool = False
    ) -> Dict:
        """
        Generate comprehensive performance report using QuantStats

        Args:
            portfolio_id: Unique portfolio identifier
            benchmark_symbol: Benchmark asset symbol (default: BTC)
            lookback_days: Historical data lookback period
            include_tearsheet: Whether to include HTML tearsheet

        Returns:
            Dictionary containing comprehensive performance metrics
        """
        try:
            # Get portfolio returns
            portfolio_returns = await self._get_portfolio_returns(portfolio_id, lookback_days)
            if portfolio_returns.empty:
                raise ValueError(f"No returns data found for portfolio {portfolio_id}")

            # Get benchmark returns
            benchmark_returns = await self._get_benchmark_returns(benchmark_symbol, lookback_days)

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                portfolio_returns, benchmark_returns
            )

            # Add metadata
            performance_metrics.update({
                'portfolio_id': portfolio_id,
                'benchmark_symbol': benchmark_symbol,
                'analysis_date': datetime.utcnow(),
                'lookback_days': lookback_days,
                'data_points': len(portfolio_returns)
            })

            # Generate tearsheet if requested
            if include_tearsheet:
                tearsheet_html = await self._generate_tearsheet(
                    portfolio_returns, benchmark_returns, portfolio_id
                )
                performance_metrics['tearsheet_html'] = tearsheet_html

            # Store performance report in database
            await self._store_performance_report(portfolio_id, performance_metrics)

            logger.info(f"Performance report generated for portfolio {portfolio_id}")
            return performance_metrics

        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            raise

    def _calculate_performance_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict:
        """
        Calculate comprehensive performance metrics using QuantStats

        Args:
            portfolio_returns: Portfolio returns time series
            benchmark_returns: Benchmark returns time series

        Returns:
            Dictionary of performance metrics
        """
        try:
            # Basic performance metrics
            metrics = {
                # Returns
                'total_return': float(qs.stats.comp(portfolio_returns)),
                'annual_return': float(qs.stats.cagr(portfolio_returns)),
                'annual_volatility': float(qs.stats.volatility(portfolio_returns)),
                'annualized_return': float(qs.stats.cagr(portfolio_returns, periods=252)),

                # Risk metrics
                'sharpe_ratio': float(qs.stats.sharpe(portfolio_returns)),
                'sortino_ratio': float(qs.stats.sortino(portfolio_returns)),
                'calmar_ratio': float(qs.stats.calmar(portfolio_returns)),
                'omega_ratio': float(qs.stats.omega(portfolio_returns)),
                'value_at_risk': float(qs.stats.value_at_risk(portfolio_returns, cutoff=0.05)),
                'conditional_value_at_risk': float(qs.stats.conditional_value_at_risk(portfolio_returns, cutoff=0.05)),

                # Drawdown metrics
                'max_drawdown': float(qs.stats.max_drawdown(portfolio_returns)),
                'avg_drawdown': float(qs.stats.avg_drawdown(portfolio_returns)),
                'avg_drawdown_days': float(qs.stats.avg_drawdown_days(portfolio_returns)),
                'max_drawdown_days': float(qs.stats.max_drawdown_days(portfolio_returns)),

                # Distribution metrics
                'skewness': float(qs.stats.skew(portfolio_returns)),
                'kurtosis': float(qs.stats.kurtosis(portfolio_returns)),

                # Win/Loss metrics
                'win_rate': float(qs.stats.win_rate(portfolio_returns)),
                'avg_win': float(qs.stats.avg_win(portfolio_returns)),
                'avg_loss': float(qs.stats.avg_loss(portfolio_returns)),
                'profit_factor': float(qs.stats.profit_factor(portfolio_returns)),
                'profit_ratio': float(qs.stats.profit_ratio(portfolio_returns)),

                # Risk-adjusted returns
                'risk_return_ratio': float(qs.stats.risk_return_ratio(portfolio_returns)),
                'cpc_index': float(qs.stats.cpc_index(portfolio_returns)),
                'common_sense_ratio': float(qs.stats.common_sense_ratio(portfolio_returns)),
                'tail_ratio': float(qs.stats.tail_ratio(portfolio_returns)),

                # Recovery metrics
                'recovery_factor': float(qs.stats.recovery_factor(portfolio_returns)),
                'ulcer_index': float(qs.stats.ulcer_index(portfolio_returns)),
                'ulcer_performance_index': float(qs.stats.ulcer_performance_index(portfolio_returns)),
            }

            # Kelly Criterion
            try:
                kelly_pct = float(qs.stats.kelly_criterion(portfolio_returns))
                metrics['kelly_criterion'] = kelly_pct
            except:
                metrics['kelly_criterion'] = None

            # Expected returns
            try:
                expected_return = float(qs.stats.expected_return(portfolio_returns))
                metrics['expected_return'] = expected_return
            except:
                metrics['expected_return'] = None

            # Monthly and annual aggregations
            monthly_returns = portfolio_returns.resample('M').apply(qs.stats.comp)
            if len(monthly_returns) > 0:
                metrics.update({
                    'monthly_return_avg': float(monthly_returns.mean()),
                    'monthly_return_std': float(monthly_returns.std()),
                    'best_month': float(monthly_returns.max()),
                    'worst_month': float(monthly_returns.min()),
                    'positive_months_ratio': float((monthly_returns > 0).mean())
                })

            # Benchmark comparison (if available)
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                # Align returns to same dates
                common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
                if len(common_dates) > 10:  # Ensure sufficient data
                    portfolio_aligned = portfolio_returns.loc[common_dates]
                    benchmark_aligned = benchmark_returns.loc[common_dates]

                    try:
                        metrics.update({
                            'alpha': float(qs.stats.alpha(portfolio_aligned, benchmark_aligned)),
                            'beta': float(qs.stats.beta(portfolio_aligned, benchmark_aligned)),
                            'information_ratio': float(qs.stats.information_ratio(portfolio_aligned, benchmark_aligned)),
                            'treynor_ratio': float(qs.stats.treynor_ratio(portfolio_aligned, benchmark_aligned)),
                            'r_squared': float(qs.stats.r_squared(portfolio_aligned, benchmark_aligned)),
                            'tracking_error': float(qs.stats.tracking_error(portfolio_aligned, benchmark_aligned)),
                            'up_capture_ratio': float(qs.stats.capture_ratio(portfolio_aligned, benchmark_aligned, period='up')),
                            'down_capture_ratio': float(qs.stats.capture_ratio(portfolio_aligned, benchmark_aligned, period='down'))
                        })
                    except Exception as e:
                        logger.warning(f"Could not calculate benchmark metrics: {e}")

            return metrics

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}

    async def _get_portfolio_returns(self, portfolio_id: str, lookback_days: int) -> pd.Series:
        """
        Calculate portfolio returns from database

        Args:
            portfolio_id: Portfolio identifier
            lookback_days: Number of historical days

        Returns:
            Portfolio returns time series
        """
        try:
            # Get portfolio data
            portfolio_doc = await self.db.portfolio_data.find_one(
                {'portfolio_id': portfolio_id},
                sort=[('timestamp', -1)]
            )

            if not portfolio_doc:
                return pd.Series()

            assets = portfolio_doc.get('assets', [])
            weights = portfolio_doc.get('weights', {})

            # If no weights provided, use equal weighting
            if not weights:
                weights = {asset: 1.0 / len(assets) for asset in assets}

            # Get returns data for each asset
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=lookback_days + 30)

            asset_returns = {}
            for asset in assets:
                # Fetch price data
                price_cursor = self.db.price_data.find({
                    'symbol': asset,
                    'timestamp': {'$gte': start_date, '$lte': end_date}
                }).sort('timestamp', 1)

                price_data = await price_cursor.to_list(length=None)

                if len(price_data) < 2:
                    # Fallback: backfill from yfinance
                    try:
                        from src.data.collectors.crypto_collector import CryptoCollector
                        collector = CryptoCollector()
                        await collector.store_historical_data(symbol=asset, period="6mo", interval="1d")
                        # Re-query after backfill
                        price_cursor = self.db.price_data.find({
                            'symbol': asset,
                            'timestamp': {'$gte': start_date, '$lte': end_date}
                        }).sort('timestamp', 1)
                        price_data = await price_cursor.to_list(length=None)
                    except Exception as fe:
                        logger.warning(f"Failed to backfill {asset} data: {fe}")
                if len(price_data) < 2:
                    continue

                # Convert to DataFrame and calculate returns
                df = pd.DataFrame(price_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)

                price_col = 'price_usd' if 'price_usd' in df.columns else ('close' if 'close' in df.columns else None)
                if price_col is None:
                    continue
                returns = df[price_col].pct_change().dropna()
                if len(returns) > lookback_days:
                    returns = returns.tail(lookback_days)

                asset_returns[asset] = returns

            if not asset_returns:
                return pd.Series()

            # Combine into DataFrame and calculate portfolio returns
            returns_df = pd.DataFrame(asset_returns)
            returns_df = returns_df.dropna()

            # Calculate weighted portfolio returns
            portfolio_weights = np.array([weights.get(asset, 0) for asset in returns_df.columns])
            portfolio_weights = portfolio_weights / portfolio_weights.sum()  # Normalize

            portfolio_returns = returns_df.dot(portfolio_weights)
            portfolio_returns.name = f'Portfolio_{portfolio_id}'

            return portfolio_returns

        except Exception as e:
            logger.error(f"Error calculating portfolio returns: {e}")
            return pd.Series()

    async def _get_benchmark_returns(self, benchmark_symbol: str, lookback_days: int) -> pd.Series:
        """
        Get benchmark returns from database

        Args:
            benchmark_symbol: Benchmark asset symbol
            lookback_days: Number of historical days

        Returns:
            Benchmark returns time series
        """
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=lookback_days + 30)

            # Fetch benchmark price data
            price_cursor = self.db.price_data.find({
                'symbol': benchmark_symbol,
                'timestamp': {'$gte': start_date, '$lte': end_date}
            }).sort('timestamp', 1)

            price_data = await price_cursor.to_list(length=None)

            if len(price_data) < 2:
                # Attempt to backfill benchmark data
                try:
                    from src.data.collectors.crypto_collector import CryptoCollector
                    collector = CryptoCollector()
                    await collector.store_historical_data(symbol=benchmark_symbol, period="6mo", interval="1d")
                    price_cursor = self.db.price_data.find({
                        'symbol': benchmark_symbol,
                        'timestamp': {'$gte': start_date, '$lte': end_date}
                    }).sort('timestamp', 1)
                    price_data = await price_cursor.to_list(length=None)
                except Exception as fe:
                    logger.warning(f"Failed to backfill benchmark {benchmark_symbol}: {fe}")
            if len(price_data) < 2:
                return pd.Series()

            # Convert to DataFrame and calculate returns
            df = pd.DataFrame(price_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            price_col = 'price_usd' if 'price_usd' in df.columns else ('close' if 'close' in df.columns else None)
            if price_col is None:
                return pd.Series()
            returns = df[price_col].pct_change().dropna()
            if len(returns) > lookback_days:
                returns = returns.tail(lookback_days)

            returns.name = benchmark_symbol
            return returns

        except Exception as e:
            logger.error(f"Error fetching benchmark returns: {e}")
            return pd.Series()

    async def _generate_tearsheet(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series],
        portfolio_id: str
    ) -> str:
        """
        Generate HTML tearsheet using QuantStats

        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            portfolio_id: Portfolio identifier

        Returns:
            Base64 encoded HTML tearsheet
        """
        try:
            # Create HTML tearsheet
            html_buffer = io.StringIO()

            if benchmark_returns is not None and len(benchmark_returns) > 0:
                qs.reports.html(
                    portfolio_returns,
                    benchmark_returns,
                    title=f"Portfolio {portfolio_id} Performance Report",
                    output=html_buffer
                )
            else:
                qs.reports.html(
                    portfolio_returns,
                    title=f"Portfolio {portfolio_id} Performance Report",
                    output=html_buffer
                )

            html_content = html_buffer.getvalue()
            html_buffer.close()

            # Encode to base64 for storage
            html_b64 = base64.b64encode(html_content.encode()).decode()
            return html_b64

        except Exception as e:
            logger.error(f"Error generating tearsheet: {e}")
            return ""

    async def generate_comprehensive_tearsheet(
        self,
        portfolio_id: str,
        benchmark_symbol: str = "BTC",
        include_attribution: bool = True,
        include_sector_analysis: bool = True,
        lookback_days: int = 252
    ) -> Dict:
        """
        Generate comprehensive tearsheet with advanced analytics

        Args:
            portfolio_id: Portfolio identifier
            benchmark_symbol: Benchmark for comparison
            include_attribution: Include performance attribution analysis
            include_sector_analysis: Include sector breakdown
            lookback_days: Historical data period

        Returns:
            Comprehensive tearsheet data
        """
        try:
            # Generate base performance report
            performance_report = await self.generate_performance_report(
                portfolio_id, benchmark_symbol, lookback_days, include_tearsheet=True
            )

            comprehensive_data = {
                'portfolio_id': portfolio_id,
                'benchmark_symbol': benchmark_symbol,
                'base_metrics': performance_report,
                'generation_date': datetime.utcnow(),
                'lookback_days': lookback_days
            }

            # Add risk-adjusted metrics
            risk_adjusted = await self.generate_risk_adjusted_metrics(
                portfolio_id, lookback_days=lookback_days
            )
            comprehensive_data['risk_adjusted_metrics'] = risk_adjusted

            # Add performance attribution if requested
            if include_attribution:
                attribution_data = {}

                # Sector attribution
                sector_attribution = await self.generate_performance_attribution(
                    portfolio_id, "sector", lookback_days
                )
                attribution_data['sector'] = sector_attribution

                # Asset attribution
                asset_attribution = await self.generate_performance_attribution(
                    portfolio_id, "asset", lookback_days
                )
                attribution_data['asset'] = asset_attribution

                # Factor attribution
                factor_attribution = await self.generate_performance_attribution(
                    portfolio_id, "factor", lookback_days
                )
                attribution_data['factor'] = factor_attribution

                comprehensive_data['attribution_analysis'] = attribution_data

            # Add sector analysis if requested
            if include_sector_analysis:
                portfolio_data = await self._get_portfolio_data(portfolio_id)
                if portfolio_data:
                    from ...core.config.crypto_sectors import get_portfolio_sector_allocation

                    sector_allocation = get_portfolio_sector_allocation(
                        portfolio_data.get('weights', {})
                    )

                    sector_analysis = {
                        'allocation': sector_allocation,
                        'sector_metrics': {},
                        'concentration_risk': self._calculate_concentration_metrics(sector_allocation)
                    }

                    # Calculate metrics for each sector
                    for sector, weight in sector_allocation.items():
                        if weight > 0.01:  # Only analyze sectors with >1% allocation
                            sector_performance = await self._analyze_sector_performance(
                                portfolio_id, sector, lookback_days
                            )
                            sector_analysis['sector_metrics'][sector] = sector_performance

                    comprehensive_data['sector_analysis'] = sector_analysis

            # Store comprehensive tearsheet
            tearsheet_doc = {
                'portfolio_id': portfolio_id,
                'comprehensive_tearsheet': comprehensive_data,
                'timestamp': datetime.utcnow()
            }

            await self.db.comprehensive_tearsheets.insert_one(tearsheet_doc)

            logger.info(f"Comprehensive tearsheet generated for portfolio {portfolio_id}")
            return comprehensive_data

        except Exception as e:
            logger.error(f"Error generating comprehensive tearsheet: {e}")
            return {}

    def _calculate_concentration_metrics(self, sector_allocation: Dict[str, float]) -> Dict:
        """Calculate portfolio concentration metrics"""
        try:
            total_weight = sum(sector_allocation.values())
            if total_weight == 0:
                return {}

            # Normalize weights
            normalized_weights = {k: v/total_weight for k, v in sector_allocation.items()}

            # Herfindahl-Hirschman Index
            hhi = sum(weight**2 for weight in normalized_weights.values())

            # Effective number of sectors
            effective_sectors = 1 / hhi if hhi > 0 else 0

            # Maximum sector weight
            max_sector_weight = max(normalized_weights.values()) if normalized_weights else 0

            # Gini coefficient for inequality
            sorted_weights = sorted(normalized_weights.values())
            n = len(sorted_weights)
            gini = 0
            if n > 1:
                cumsum = sum((i + 1) * weight for i, weight in enumerate(sorted_weights))
                gini = (2 * cumsum) / (n * sum(sorted_weights)) - (n + 1) / n

            return {
                'herfindahl_index': float(hhi),
                'effective_sectors': float(effective_sectors),
                'max_sector_weight': float(max_sector_weight),
                'gini_coefficient': float(gini),
                'concentration_risk': 'High' if hhi > 0.25 else 'Medium' if hhi > 0.15 else 'Low'
            }

        except Exception as e:
            logger.error(f"Error calculating concentration metrics: {e}")
            return {}

    async def _analyze_sector_performance(
        self,
        portfolio_id: str,
        sector: str,
        lookback_days: int
    ) -> Dict:
        """Analyze performance metrics for a specific sector"""
        try:
            from ...core.config.crypto_sectors import get_sector_assets

            sector_assets = get_sector_assets(sector)
            if not sector_assets:
                return {}

            # Get portfolio data to determine weights
            portfolio_data = await self._get_portfolio_data(portfolio_id)
            if not portfolio_data:
                return {}

            # Find assets in portfolio that belong to this sector
            portfolio_sector_assets = [
                asset for asset in portfolio_data.get('assets', [])
                if asset in sector_assets
            ]

            if not portfolio_sector_assets:
                return {}

            # Calculate sector-specific returns
            sector_returns_dict = {}
            sector_weights = {}
            total_sector_weight = 0

            for asset in portfolio_sector_assets:
                asset_weight = portfolio_data.get('weights', {}).get(asset, 0)
                if asset_weight > 0:
                    asset_returns = await self._get_asset_returns(asset, lookback_days)
                    if not asset_returns.empty:
                        sector_returns_dict[asset] = asset_returns
                        sector_weights[asset] = asset_weight
                        total_sector_weight += asset_weight

            if not sector_returns_dict or total_sector_weight == 0:
                return {}

            # Normalize weights within sector
            normalized_weights = {
                asset: weight / total_sector_weight
                for asset, weight in sector_weights.items()
            }

            # Calculate weighted sector returns
            sector_returns_df = pd.DataFrame(sector_returns_dict)
            weighted_returns = sum(
                sector_returns_df[asset] * normalized_weights[asset]
                for asset in normalized_weights
            )

            # Calculate sector performance metrics
            sector_metrics = {
                'total_return': float(qs.stats.comp(weighted_returns)),
                'annual_return': float(qs.stats.cagr(weighted_returns)),
                'volatility': float(qs.stats.volatility(weighted_returns)),
                'sharpe_ratio': float(qs.stats.sharpe(weighted_returns)),
                'max_drawdown': float(qs.stats.max_drawdown(weighted_returns)),
                'win_rate': float(qs.stats.win_rate(weighted_returns)),
                'assets_count': len(portfolio_sector_assets),
                'sector_weight': float(total_sector_weight),
                'top_performer': None,
                'worst_performer': None
            }

            # Find best and worst performing assets in sector
            if len(sector_returns_dict) > 1:
                asset_returns = {
                    asset: float(qs.stats.comp(returns))
                    for asset, returns in sector_returns_dict.items()
                }
                sector_metrics['top_performer'] = max(asset_returns, key=asset_returns.get)
                sector_metrics['worst_performer'] = min(asset_returns, key=asset_returns.get)

            return sector_metrics

        except Exception as e:
            logger.error(f"Error analyzing sector {sector} performance: {e}")
            return {}

    async def _store_performance_report(self, portfolio_id: str, report: Dict) -> None:
        """Store performance report in MongoDB"""
        try:
            performance_doc = {
                'portfolio_id': portfolio_id,
                'performance_report': report,
                'timestamp': datetime.utcnow(),
                'generated_by': 'QuantStatsAnalyzer'
            }

            await self.db.performance_reports.insert_one(performance_doc)
            logger.info(f"Performance report stored for portfolio {portfolio_id}")

        except Exception as e:
            logger.error(f"Error storing performance report: {e}")
            raise

    async def get_latest_performance_report(self, portfolio_id: str) -> Optional[Dict]:
        """Retrieve latest performance report for a portfolio"""
        try:
            report_doc = await self.db.performance_reports.find_one(
                {'portfolio_id': portfolio_id},
                sort=[('timestamp', -1)]
            )

            if report_doc:
                return report_doc['performance_report']
            return None

        except Exception as e:
            logger.error(f"Error fetching latest performance report: {e}")
            return None

    async def compare_portfolios(
        self,
        portfolio_ids: List[str],
        lookback_days: int = 252
    ) -> Dict:
        """
        Compare performance metrics across multiple portfolios

        Args:
            portfolio_ids: List of portfolio identifiers
            lookback_days: Historical data lookback period

        Returns:
            Dictionary containing comparison metrics
        """
        try:
            portfolio_metrics = {}
            portfolio_returns_dict = {}

            # Generate metrics for each portfolio
            for portfolio_id in portfolio_ids:
                portfolio_returns = await self._get_portfolio_returns(portfolio_id, lookback_days)
                if not portfolio_returns.empty:
                    metrics = self._calculate_performance_metrics(portfolio_returns)
                    portfolio_metrics[portfolio_id] = metrics
                    portfolio_returns_dict[portfolio_id] = portfolio_returns

            if not portfolio_metrics:
                return {}

            # Calculate correlation matrix
            if len(portfolio_returns_dict) > 1:
                returns_df = pd.DataFrame(portfolio_returns_dict)
                correlation_matrix = returns_df.corr().to_dict()
            else:
                correlation_matrix = {}

            comparison_report = {
                'portfolio_metrics': portfolio_metrics,
                'correlation_matrix': correlation_matrix,
                'comparison_date': datetime.utcnow(),
                'portfolios_compared': len(portfolio_ids),
                'lookback_days': lookback_days
            }

            # Store comparison report
            await self.db.portfolio_comparisons.insert_one(comparison_report)

            logger.info(f"Portfolio comparison completed for {len(portfolio_ids)} portfolios")
            return comparison_report

        except Exception as e:
            logger.error(f"Error comparing portfolios: {e}")
            return {}

    async def generate_performance_attribution(
        self,
        portfolio_id: str,
        attribution_type: str = "sector",
        lookback_days: int = 252
    ) -> Dict:
        """
        Generate performance attribution analysis

        Args:
            portfolio_id: Portfolio identifier
            attribution_type: Type of attribution ('sector', 'asset', 'factor')
            lookback_days: Historical data lookback period

        Returns:
            Performance attribution breakdown
        """
        try:
            # Get portfolio data
            portfolio_data = await self._get_portfolio_data(portfolio_id)
            if not portfolio_data:
                return {}

            portfolio_returns = await self._get_portfolio_returns(portfolio_id, lookback_days)
            if portfolio_returns.empty:
                return {}

            attribution_results = {}

            if attribution_type == "sector":
                # Implement sector-based attribution
                attribution_results = await self._calculate_sector_attribution(
                    portfolio_data, portfolio_returns, lookback_days
                )
            elif attribution_type == "asset":
                # Implement asset-based attribution
                attribution_results = await self._calculate_asset_attribution(
                    portfolio_data, portfolio_returns, lookback_days
                )
            elif attribution_type == "factor":
                # Implement factor-based attribution
                attribution_results = await self._calculate_factor_attribution(
                    portfolio_data, portfolio_returns, lookback_days
                )

            # Store attribution analysis
            attribution_doc = {
                'portfolio_id': portfolio_id,
                'attribution_type': attribution_type,
                'attribution_results': attribution_results,
                'lookback_days': lookback_days,
                'timestamp': datetime.utcnow()
            }

            await self.db.performance_attribution.insert_one(attribution_doc)

            logger.info(f"Performance attribution calculated for portfolio {portfolio_id}")
            return attribution_results

        except Exception as e:
            logger.error(f"Error calculating performance attribution: {e}")
            return {}

    async def _calculate_sector_attribution(
        self,
        portfolio_data: Dict,
        portfolio_returns: pd.Series,
        lookback_days: int
    ) -> Dict:
        """Calculate sector-based performance attribution"""
        try:
            # Import crypto sectors configuration
            from ...core.config.crypto_sectors import CRYPTO_SECTORS

            sector_contributions = {}
            total_portfolio_return = portfolio_returns.sum()

            for sector_name, sector_config in CRYPTO_SECTORS.items():
                # Find portfolio assets in this sector
                sector_assets = [
                    asset for asset in portfolio_data.get('assets', [])
                    if asset in sector_config.get('tickers', [])
                ]

                if not sector_assets:
                    continue

                # Calculate sector weight in portfolio
                sector_weight = sum([
                    portfolio_data.get('weights', {}).get(asset, 0)
                    for asset in sector_assets
                ])

                if sector_weight == 0:
                    continue

                # Get sector returns
                sector_returns_dict = {}
                for asset in sector_assets:
                    asset_returns = await self._get_asset_returns(asset, lookback_days)
                    if not asset_returns.empty:
                        asset_weight = portfolio_data.get('weights', {}).get(asset, 0) / sector_weight
                        sector_returns_dict[asset] = asset_returns * asset_weight

                if sector_returns_dict:
                    sector_returns = pd.DataFrame(sector_returns_dict).sum(axis=1)
                    sector_total_return = sector_returns.sum()
                    sector_contribution = sector_weight * sector_total_return

                    sector_contributions[sector_name] = {
                        'weight': float(sector_weight),
                        'return': float(sector_total_return),
                        'contribution': float(sector_contribution),
                        'contribution_pct': float(sector_contribution / total_portfolio_return * 100) if total_portfolio_return != 0 else 0,
                        'assets': sector_assets,
                        'volatility': float(sector_returns.std() * np.sqrt(252)) if len(sector_returns) > 1 else 0,
                        'sharpe_ratio': float(qs.stats.sharpe(sector_returns)) if len(sector_returns) > 1 else 0
                    }

            return sector_contributions

        except Exception as e:
            logger.error(f"Error calculating sector attribution: {e}")
            return {}

    async def _calculate_asset_attribution(
        self,
        portfolio_data: Dict,
        portfolio_returns: pd.Series,
        lookback_days: int
    ) -> Dict:
        """Calculate asset-based performance attribution"""
        try:
            asset_contributions = {}
            total_portfolio_return = portfolio_returns.sum()

            for asset in portfolio_data.get('assets', []):
                asset_weight = portfolio_data.get('weights', {}).get(asset, 0)
                if asset_weight == 0:
                    continue

                # Get asset returns
                asset_returns = await self._get_asset_returns(asset, lookback_days)
                if asset_returns.empty:
                    continue

                asset_total_return = asset_returns.sum()
                asset_contribution = asset_weight * asset_total_return

                asset_contributions[asset] = {
                    'weight': float(asset_weight),
                    'return': float(asset_total_return),
                    'contribution': float(asset_contribution),
                    'contribution_pct': float(asset_contribution / total_portfolio_return * 100) if total_portfolio_return != 0 else 0,
                    'volatility': float(asset_returns.std() * np.sqrt(252)),
                    'sharpe_ratio': float(qs.stats.sharpe(asset_returns)),
                    'max_drawdown': float(qs.stats.max_drawdown(asset_returns)),
                    'beta': await self._calculate_asset_beta(asset, lookback_days)
                }

            return asset_contributions

        except Exception as e:
            logger.error(f"Error calculating asset attribution: {e}")
            return {}

    async def _calculate_factor_attribution(
        self,
        portfolio_data: Dict,
        portfolio_returns: pd.Series,
        lookback_days: int
    ) -> Dict:
        """Calculate factor-based performance attribution"""
        try:
            # Implement factor attribution using common crypto factors
            factors = {
                'market': 'BTC',  # Market factor
                'size': 'ETH',    # Size factor (large vs small cap)
                'momentum': None,  # Momentum factor (to be calculated)
                'volatility': None # Volatility factor (to be calculated)
            }

            factor_contributions = {}

            # Market factor (BTC)
            btc_returns = await self._get_asset_returns('BTC', lookback_days)
            if not btc_returns.empty:
                market_beta = self._calculate_beta(portfolio_returns, btc_returns)
                market_contribution = market_beta * btc_returns.sum()

                factor_contributions['market'] = {
                    'beta': float(market_beta),
                    'factor_return': float(btc_returns.sum()),
                    'contribution': float(market_contribution),
                    'r_squared': float(self._calculate_r_squared(portfolio_returns, btc_returns))
                }

            # Size factor (ETH relative to BTC)
            eth_returns = await self._get_asset_returns('ETH', lookback_days)
            if not eth_returns.empty and not btc_returns.empty:
                size_factor_returns = eth_returns - btc_returns
                size_beta = self._calculate_beta(portfolio_returns, size_factor_returns)
                size_contribution = size_beta * size_factor_returns.sum()

                factor_contributions['size'] = {
                    'beta': float(size_beta),
                    'factor_return': float(size_factor_returns.sum()),
                    'contribution': float(size_contribution),
                    'r_squared': float(self._calculate_r_squared(portfolio_returns, size_factor_returns))
                }

            return factor_contributions

        except Exception as e:
            logger.error(f"Error calculating factor attribution: {e}")
            return {}

    async def _get_asset_returns(self, asset: str, lookback_days: int) -> pd.Series:
        """Get returns for a specific asset"""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=lookback_days + 30)

            # Fetch price data
            price_cursor = self.db.price_data.find({
                'symbol': asset,
                'timestamp': {'$gte': start_date, '$lte': end_date}
            }).sort('timestamp', 1)

            price_data = await price_cursor.to_list(length=None)

            if len(price_data) < 2:
                return pd.Series()

            # Convert to DataFrame and calculate returns
            df = pd.DataFrame(price_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            returns = df['price_usd'].pct_change().dropna()
            if len(returns) > lookback_days:
                returns = returns.tail(lookback_days)

            returns.name = asset
            return returns

        except Exception as e:
            logger.error(f"Error fetching asset returns for {asset}: {e}")
            return pd.Series()

    async def _calculate_asset_beta(self, asset: str, lookback_days: int) -> float:
        """Calculate beta for an asset relative to BTC"""
        try:
            asset_returns = await self._get_asset_returns(asset, lookback_days)
            btc_returns = await self._get_asset_returns('BTC', lookback_days)

            if asset_returns.empty or btc_returns.empty:
                return 1.0

            return self._calculate_beta(asset_returns, btc_returns)

        except Exception as e:
            logger.error(f"Error calculating beta for {asset}: {e}")
            return 1.0

    def _calculate_beta(self, returns1: pd.Series, returns2: pd.Series) -> float:
        """Calculate beta between two return series"""
        try:
            # Align the series
            common_dates = returns1.index.intersection(returns2.index)
            if len(common_dates) < 10:
                return 1.0

            aligned_returns1 = returns1.loc[common_dates]
            aligned_returns2 = returns2.loc[common_dates]

            # Calculate beta using covariance
            covariance = np.cov(aligned_returns1, aligned_returns2)[0, 1]
            variance = np.var(aligned_returns2)

            if variance == 0:
                return 1.0

            return covariance / variance

        except Exception:
            return 1.0

    def _calculate_r_squared(self, returns1: pd.Series, returns2: pd.Series) -> float:
        """Calculate R-squared between two return series"""
        try:
            # Align the series
            common_dates = returns1.index.intersection(returns2.index)
            if len(common_dates) < 10:
                return 0.0

            aligned_returns1 = returns1.loc[common_dates]
            aligned_returns2 = returns2.loc[common_dates]

            # Calculate correlation and square it
            correlation = np.corrcoef(aligned_returns1, aligned_returns2)[0, 1]
            return correlation ** 2 if not np.isnan(correlation) else 0.0

        except Exception:
            return 0.0

    async def _get_portfolio_data(self, portfolio_id: str) -> Optional[Dict]:
        """Fetch portfolio configuration data from MongoDB"""
        try:
            portfolio_doc = await self.db.portfolio_data.find_one(
                {'portfolio_id': portfolio_id},
                sort=[('timestamp', -1)]
            )
            return portfolio_doc
        except Exception as e:
            logger.error(f"Error fetching portfolio data: {e}")
            return None

    async def generate_risk_adjusted_metrics(
        self,
        portfolio_id: str,
        risk_free_rate: float = 0.05,
        lookback_days: int = 252
    ) -> Dict:
        """
        Generate comprehensive risk-adjusted performance metrics

        Args:
            portfolio_id: Portfolio identifier
            risk_free_rate: Risk-free rate for calculations
            lookback_days: Historical data lookback period

        Returns:
            Risk-adjusted performance metrics
        """
        try:
            portfolio_returns = await self._get_portfolio_returns(portfolio_id, lookback_days)
            if portfolio_returns.empty:
                return {}

            # Calculate risk-adjusted metrics
            risk_adjusted_metrics = {
                # Risk-adjusted returns
                'excess_returns': float(qs.stats.comp(portfolio_returns) - risk_free_rate),
                'information_ratio': float(qs.stats.information_ratio(portfolio_returns)),
                'treynor_ratio': 0.0,  # Will calculate if benchmark available
                'jensen_alpha': 0.0,   # Will calculate if benchmark available

                # Downside risk metrics
                'downside_deviation': float(qs.stats.volatility(portfolio_returns[portfolio_returns < 0]) * np.sqrt(252)),
                'upside_deviation': float(qs.stats.volatility(portfolio_returns[portfolio_returns > 0]) * np.sqrt(252)),
                'capture_ratio': 0.0,  # Will calculate if benchmark available

                # Advanced risk metrics
                'var_modified': float(qs.stats.value_at_risk(portfolio_returns, cutoff=0.05, method="modified")),
                'cvar_modified': float(qs.stats.conditional_value_at_risk(portfolio_returns, cutoff=0.05, method="modified")),
                'max_drawdown_duration': float(qs.stats.max_drawdown_days(portfolio_returns)),

                # Consistency metrics
                'win_loss_ratio': float(qs.stats.avg_win(portfolio_returns) / abs(qs.stats.avg_loss(portfolio_returns))) if qs.stats.avg_loss(portfolio_returns) != 0 else 0,
                'gain_to_pain_ratio': float(qs.stats.gain_to_pain_ratio(portfolio_returns)),
                'lake_ratio': float(qs.stats.lake_ratio(portfolio_returns)),

                # Tail risk metrics
                'tail_ratio': float(qs.stats.tail_ratio(portfolio_returns)),
                'common_sense_ratio': float(qs.stats.common_sense_ratio(portfolio_returns)),

                # Efficiency metrics
                'efficiency_ratio': float(qs.stats.risk_return_ratio(portfolio_returns)),
                'smart_sharpe': float(qs.stats.smart_sharpe(portfolio_returns)),
                'smart_sortino': float(qs.stats.smart_sortino(portfolio_returns))
            }

            # Add benchmark comparison if BTC data available
            btc_returns = await self._get_asset_returns('BTC', lookback_days)
            if not btc_returns.empty:
                # Align returns
                common_dates = portfolio_returns.index.intersection(btc_returns.index)
                if len(common_dates) > 10:
                    portfolio_aligned = portfolio_returns.loc[common_dates]
                    btc_aligned = btc_returns.loc[common_dates]

                    try:
                        risk_adjusted_metrics.update({
                            'treynor_ratio': float(qs.stats.treynor_ratio(portfolio_aligned, btc_aligned)),
                            'jensen_alpha': float(qs.stats.alpha(portfolio_aligned, btc_aligned)),
                            'up_capture_ratio': float(qs.stats.capture_ratio(portfolio_aligned, btc_aligned, period='up')),
                            'down_capture_ratio': float(qs.stats.capture_ratio(portfolio_aligned, btc_aligned, period='down')),
                            'capture_ratio': float(qs.stats.capture_ratio(portfolio_aligned, btc_aligned))
                        })
                    except Exception as e:
                        logger.warning(f"Could not calculate benchmark metrics: {e}")

            # Store risk-adjusted metrics
            metrics_doc = {
                'portfolio_id': portfolio_id,
                'risk_adjusted_metrics': risk_adjusted_metrics,
                'risk_free_rate': risk_free_rate,
                'lookback_days': lookback_days,
                'timestamp': datetime.utcnow()
            }

            await self.db.risk_adjusted_metrics.insert_one(metrics_doc)

            logger.info(f"Risk-adjusted metrics calculated for portfolio {portfolio_id}")
            return risk_adjusted_metrics

        except Exception as e:
            logger.error(f"Error calculating risk-adjusted metrics: {e}")
            return {}
