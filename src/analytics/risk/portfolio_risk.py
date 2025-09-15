# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Portfolio Risk Analytics using Riskfolio-lib and QuantStats
Advanced risk management for cryptocurrency portfolios with professional-grade metrics
"""

from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging

import riskfolio as rp
import quantstats as qs
from motor.motor_asyncio import AsyncIOMotorDatabase

try:
    from ...core.database.models import RiskMetrics, PortfolioData
    from ...core.utils.time_utils import TimeUtils
except ImportError:
    # Fallback imports for direct execution
    try:
        from src.core.database.models import RiskMetrics, PortfolioData
        from src.core.utils.time_utils import TimeUtils
    except ImportError:
        # Create minimal fallback if modules don't exist
        class TimeUtils:
            pass
        class RiskMetrics:
            pass
        class PortfolioData:
            pass

# Import advanced risk analytics
from .monte_carlo import MonteCarloEngine, SimulationConfig, run_comprehensive_risk_analysis
from .correlation_analysis import CorrelationAnalyzer
from .backtesting import RiskModelBacktester, BacktestConfig, VaRModels
from .risk_monitoring import RiskMonitor, MonitoringConfig, create_monitoring_system

logger = logging.getLogger(__name__)

class PortfolioRiskAnalyzer:
    """
    Professional portfolio risk analysis using riskfolio-lib
    Implements comprehensive risk metrics calculation and storage
    """

    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize portfolio risk analyzer with database connection

        Args:
            db: MongoDB database connection
        """
        self.db = db
        self.time_utils = TimeUtils()

    async def calculate_portfolio_metrics(
        self,
        portfolio_id: str,
        lookback_days: int = 252,
        confidence_levels: List[float] = [0.05, 0.01]
    ) -> Dict:
        """
        Calculate comprehensive risk metrics using riskfolio-lib

        Args:
            portfolio_id: Unique portfolio identifier
            lookback_days: Historical data lookback period
            confidence_levels: VaR/CVaR confidence levels

        Returns:
            Dictionary containing all calculated risk metrics
        """
        try:
            # Fetch portfolio data from database
            portfolio_data = await self._get_portfolio_data(portfolio_id)
            if not portfolio_data:
                raise ValueError(f"No portfolio data found for ID: {portfolio_id}")

            # Get returns data for portfolio assets
            returns_data = await self._get_returns_data(
                portfolio_data['assets'],
                lookback_days
            )

            if returns_data.empty:
                raise ValueError("No returns data available for portfolio assets")

            # Create riskfolio Portfolio object
            port = rp.Portfolio(returns=returns_data)

            # Calculate assets statistics
            port.assets_stats(method_mu='hist', method_cov='hist')

            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(
                port, returns_data, confidence_levels
            )

            # Add portfolio metadata
            risk_metrics.update({
                'portfolio_id': portfolio_id,
                'calculation_date': datetime.utcnow(),
                'lookback_days': lookback_days,
                'assets_count': len(portfolio_data['assets']),
                'confidence_levels': confidence_levels
            })

            # Store metrics in database
            await self._store_risk_metrics(portfolio_id, risk_metrics)

            logger.info(f"Risk metrics calculated for portfolio {portfolio_id}")
            return risk_metrics

        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            raise

    async def _calculate_risk_metrics(
        self,
        port: rp.Portfolio,
        returns_data: pd.DataFrame,
        confidence_levels: List[float]
    ) -> Dict:
        """
        Calculate comprehensive risk metrics using riskfolio and quantstats

        Args:
            port: Riskfolio Portfolio object
            returns_data: Historical returns data
            confidence_levels: VaR/CVaR confidence levels

        Returns:
            Dictionary of calculated risk metrics
        """
        # Portfolio returns (equal weighted for now)
        weights = np.ones(len(returns_data.columns)) / len(returns_data.columns)
        portfolio_returns = returns_data.dot(weights)

        # Basic performance metrics using QuantStats
        performance_metrics = {
            'total_return': float(qs.stats.comp(portfolio_returns)),
            'annual_return': float(qs.stats.cagr(portfolio_returns)),
            'annual_volatility': float(qs.stats.volatility(portfolio_returns)),
            'sharpe_ratio': float(qs.stats.sharpe(portfolio_returns)),
            'sortino_ratio': float(qs.stats.sortino(portfolio_returns)),
            'calmar_ratio': float(qs.stats.calmar(portfolio_returns)),
            'max_drawdown': float(qs.stats.max_drawdown(portfolio_returns)),
            'omega_ratio': float(qs.stats.omega(portfolio_returns)),
            'skewness': float(qs.stats.skew(portfolio_returns)),
            'kurtosis': float(qs.stats.kurtosis(portfolio_returns))
        }

        # Advanced risk metrics using Riskfolio
        risk_metrics = {}

        # Calculate VaR and CVaR for each confidence level
        for alpha in confidence_levels:
            alpha_pct = int(alpha * 100)

            try:
                # Historical VaR and CVaR
                var_hist = float(port.var_hist(returns_data.dot(weights), alpha=alpha))
                cvar_hist = float(port.cvar_hist(returns_data.dot(weights), alpha=alpha))

                risk_metrics.update({
                    f'var_{alpha_pct}': var_hist,
                    f'cvar_{alpha_pct}': cvar_hist,
                    f'expected_shortfall_{alpha_pct}': cvar_hist  # ES = CVaR
                })
            except Exception as e:
                logger.warning(f"Could not calculate VaR/CVaR for alpha={alpha}: {e}")
                risk_metrics.update({
                    f'var_{alpha_pct}': None,
                    f'cvar_{alpha_pct}': None,
                    f'expected_shortfall_{alpha_pct}': None
                })

        # Portfolio optimization metrics
        try:
            # Mean-Variance optimization
            w_mv = port.optimization(model='Classic', rm='MV', obj='Sharpe')
            if w_mv is not None:
                risk_metrics['optimal_sharpe_weights'] = w_mv.to_dict()['weights'] if hasattr(w_mv, 'to_dict') else None
        except Exception as e:
            logger.warning(f"Could not calculate optimal weights: {e}")
            risk_metrics['optimal_sharpe_weights'] = None

        # Combine all metrics
        all_metrics = {**performance_metrics, **risk_metrics}

        # Additional portfolio-level metrics
        all_metrics.update({
            'returns_correlation_mean': float(returns_data.corr().values[np.triu_indices_from(returns_data.corr().values, k=1)].mean()),
            'portfolio_concentration': float(self._calculate_concentration_ratio(weights)),
            'diversification_ratio': float(self._calculate_diversification_ratio(returns_data, weights))
        })

        return all_metrics

    def _calculate_concentration_ratio(self, weights: np.ndarray) -> float:
        """Calculate portfolio concentration using Herfindahl-Hirschman Index"""
        return np.sum(weights ** 2)

    def _calculate_diversification_ratio(self, returns_data: pd.DataFrame, weights: np.ndarray) -> float:
        """Calculate diversification ratio"""
        try:
            portfolio_vol = np.sqrt(weights.T @ returns_data.cov().values @ weights)
            weighted_avg_vol = weights @ returns_data.std().values
            return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        except:
            return 1.0

    async def _get_portfolio_data(self, portfolio_id: str) -> Optional[Dict]:
        """Fetch portfolio data from MongoDB"""
        try:
            portfolio_doc = await self.db.portfolio_data.find_one(
                {'portfolio_id': portfolio_id},
                sort=[('timestamp', -1)]
            )
            return portfolio_doc
        except Exception as e:
            logger.error(f"Error fetching portfolio data: {e}")
            return None

    async def _get_returns_data(
        self,
        assets: List[str],
        lookback_days: int
    ) -> pd.DataFrame:
        """
        Fetch historical returns data for portfolio assets

        Args:
            assets: List of asset symbols
            lookback_days: Number of historical days to fetch

        Returns:
            DataFrame with returns data
        """
        try:
            returns_dict = {}

            for asset in assets:
                # FIXED: Use sort + limit approach instead of date range query
                # MongoDB date range queries were failing, so get recent data by sorting

                # Get recent price data (with buffer for calculations)
                price_cursor = self.db.price_data.find({
                    'symbol': asset
                }).sort('timestamp', -1).limit(lookback_days + 50)  # Extra buffer

                price_data = await price_cursor.to_list(length=None)

                if len(price_data) < 2:
                    # Fallback: fetch and store historical data from yfinance
                    try:
                        from src.data.collectors.crypto_collector import CryptoCollector
                        collector = CryptoCollector()
                        # Fetch ~6 months of daily data
                        await collector.store_historical_data(symbol=asset, period="6mo", interval="1d")
                    except Exception as fe:
                        logger.warning(f"Failed to backfill {asset} historical prices: {fe}")
                    # Retry fetching after backfill
                    price_cursor = self.db.price_data.find({'symbol': asset}).sort('timestamp', -1).limit(lookback_days + 50)
                    price_data = await price_cursor.to_list(length=None)
                    if len(price_data) < 2:
                        logger.warning(f"Insufficient price data for {asset}")
                        continue

                # Convert to DataFrame and calculate returns
                df = pd.DataFrame(price_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)  # Sort chronologically (oldest first)

                # Determine price column: prefer 'price_usd', fallback to 'close'
                price_col = 'price_usd' if 'price_usd' in df.columns else ('close' if 'close' in df.columns else None)
                if price_col is None:
                    logger.warning(f"No usable price column for {asset}")
                    continue

                # Calculate daily returns
                returns = df[price_col].pct_change().dropna()

                # Take only the required lookback period
                if len(returns) > lookback_days:
                    returns = returns.tail(lookback_days)

                returns_dict[asset] = returns

            if not returns_dict:
                return pd.DataFrame()

            # Combine returns into single DataFrame with proper alignment
            returns_df = pd.DataFrame(returns_dict)

            # Instead of dropping all NaN rows, use forward fill and then drop
            # This handles cases where assets have slightly different timestamps
            returns_df = returns_df.ffill().bfill()
            returns_df = returns_df.dropna()

            # If still empty, try a more lenient approach
            if len(returns_df) == 0 and returns_dict:
                logger.warning("Strict alignment failed, using individual asset data")
                # Use the longest available series as base
                longest_asset = max(returns_dict.keys(), key=lambda x: len(returns_dict[x]))
                returns_df = pd.DataFrame({longest_asset: returns_dict[longest_asset]})

                # Add other assets where timestamps align
                for asset, returns_series in returns_dict.items():
                    if asset != longest_asset:
                        returns_df[asset] = returns_series

                # Fill NaN and clean up
                returns_df = returns_df.ffill().fillna(0)

            logger.info(f"Loaded returns data: {len(returns_df)} days, {len(returns_df.columns)} assets")
            return returns_df

        except Exception as e:
            logger.error(f"Error loading returns data: {e}")
            return pd.DataFrame()

    async def _store_risk_metrics(self, portfolio_id: str, metrics: Dict) -> None:
        """Store calculated risk metrics in MongoDB"""
        try:
            risk_doc = {
                'portfolio_id': portfolio_id,
                'metrics': metrics,
                'timestamp': datetime.utcnow(),
                'calculated_by': 'PortfolioRiskAnalyzer'
            }

            await self.db.risk_metrics.insert_one(risk_doc)
            logger.info(f"Risk metrics stored for portfolio {portfolio_id}")

        except Exception as e:
            logger.error(f"Error storing risk metrics: {e}")
            raise

    async def get_latest_risk_metrics(self, portfolio_id: str) -> Optional[Dict]:
        """Retrieve latest risk metrics for a portfolio"""
        try:
            risk_doc = await self.db.risk_metrics.find_one(
                {'portfolio_id': portfolio_id},
                sort=[('timestamp', -1)]
            )

            if risk_doc:
                return risk_doc['metrics']
            return None

        except Exception as e:
            logger.error(f"Error fetching latest risk metrics: {e}")
            return None

    async def calculate_sector_allocation_risk(
        self,
        portfolio_id: str,
        crypto_sectors: Dict[str, List[str]]
    ) -> Dict:
        """
        Calculate risk metrics by crypto sector allocation

        Args:
            portfolio_id: Portfolio identifier
            crypto_sectors: Mapping of sectors to asset lists

        Returns:
            Risk metrics grouped by crypto sector
        """
        try:
            portfolio_data = await self._get_portfolio_data(portfolio_id)
            if not portfolio_data:
                return {}

            sector_metrics = {}

            for sector_name, sector_assets in crypto_sectors.items():
                # Filter portfolio assets that belong to this sector
                portfolio_sector_assets = [
                    asset for asset in portfolio_data['assets']
                    if asset in sector_assets
                ]

                if not portfolio_sector_assets:
                    continue

                # Calculate returns for sector assets
                sector_returns = await self._get_returns_data(portfolio_sector_assets, 252)

                if sector_returns.empty:
                    continue

                # Create sector portfolio
                sector_port = rp.Portfolio(returns=sector_returns)
                sector_port.assets_stats(method_mu='hist', method_cov='hist')

                # Calculate sector risk metrics
                sector_risk = await self._calculate_risk_metrics(
                    sector_port, sector_returns, [0.05, 0.01]
                )

                sector_metrics[sector_name] = {
                    'assets': portfolio_sector_assets,
                    'asset_count': len(portfolio_sector_assets),
                    'risk_metrics': sector_risk
                }

            # Store sector allocation risk
            sector_doc = {
                'portfolio_id': portfolio_id,
                'sector_allocation_risk': sector_metrics,
                'timestamp': datetime.utcnow()
            }

            await self.db.sector_risk_metrics.insert_one(sector_doc)

            logger.info(f"Sector allocation risk calculated for portfolio {portfolio_id}")
            return sector_metrics

        except Exception as e:
            logger.error(f"Error calculating sector allocation risk: {e}")
            return {}

    # === ADVANCED RISK ANALYTICS METHODS ===

    async def run_monte_carlo_analysis(
        self,
        portfolio_id: str,
        simulation_config: SimulationConfig = None,
        lookback_days: int = 252
    ) -> Dict:
        """
        Run comprehensive Monte Carlo risk analysis

        Args:
            portfolio_id: Portfolio identifier
            simulation_config: Monte Carlo simulation configuration
            lookback_days: Historical data lookback period

        Returns:
            Comprehensive Monte Carlo analysis results
        """
        try:
            logger.info(f"Starting Monte Carlo analysis for portfolio {portfolio_id}")

            # Get portfolio data and returns
            portfolio_data = await self._get_portfolio_data(portfolio_id)
            if not portfolio_data:
                return {}

            returns_data = await self._get_returns_data(portfolio_data['assets'], lookback_days)
            if returns_data.empty:
                return {}

            # Portfolio weights
            weights = np.array([portfolio_data['weights'].get(asset, 0) for asset in returns_data.columns])
            weights = weights / np.sum(weights)  # Normalize

            # Run comprehensive analysis
            results = await run_comprehensive_risk_analysis(
                weights, returns_data, simulation_config
            )

            # Store results in database
            analysis_doc = {
                'portfolio_id': portfolio_id,
                'analysis_type': 'monte_carlo',
                'results': {
                    'base_case_var_95': float(results['base_case'].var_estimates.get(0.95, 0)),
                    'base_case_cvar_95': float(results['base_case'].cvar_estimates.get(0.95, 0)),
                    'base_case_statistics': results['base_case'].statistics,
                    'stress_test_results': {
                        scenario: {
                            'var_95': float(result.var_estimates.get(0.95, 0)),
                            'cvar_95': float(result.cvar_estimates.get(0.95, 0)),
                            'statistics': result.statistics
                        }
                        for scenario, result in results['stress_tests'].items()
                    },
                    'historical_scenarios': results['historical_scenarios']
                },
                'metadata': results['analysis_metadata'],
                'timestamp': datetime.utcnow()
            }

            await self.db.monte_carlo_analysis.insert_one(analysis_doc)

            logger.info(f"Monte Carlo analysis completed for portfolio {portfolio_id}")
            return results

        except Exception as e:
            logger.error(f"Error in Monte Carlo analysis: {e}")
            return {}

    async def analyze_correlation_structure(
        self,
        portfolio_id: str,
        estimation_method: str = "ledoit_wolf",
        lookback_days: int = 252
    ) -> Dict:
        """
        Perform comprehensive correlation structure analysis

        Args:
            portfolio_id: Portfolio identifier
            estimation_method: Correlation estimation method
            lookback_days: Historical data lookback period

        Returns:
            Correlation analysis results
        """
        try:
            logger.info(f"Analyzing correlation structure for portfolio {portfolio_id}")

            # Get portfolio data and returns
            portfolio_data = await self._get_portfolio_data(portfolio_id)
            if not portfolio_data:
                return {}

            returns_data = await self._get_returns_data(portfolio_data['assets'], lookback_days)
            if returns_data.empty:
                return {}

            # Initialize correlation analyzer
            analyzer = CorrelationAnalyzer(estimation_method=estimation_method)
            analyzer.load_data(returns_data)

            # Analyze correlation structure
            corr_metrics = analyzer.analyze_correlation_structure()

            # Risk decomposition
            weights = np.array([portfolio_data['weights'].get(asset, 0) for asset in returns_data.columns])
            weights = weights / np.sum(weights)  # Normalize
            risk_decomp = analyzer.risk_decomposition(weights)

            # Hierarchical clustering
            clustering = analyzer.hierarchical_clustering()

            # Factor analysis
            factor_analysis = analyzer.factor_analysis()

            # Sector correlation analysis
            try:
                from ...core.config.crypto_sectors import get_asset_sector
                sector_mapping = {asset: get_asset_sector(asset) for asset in returns_data.columns}
            except ImportError:
                # Fallback for import issues
                from src.core.config.crypto_sectors import get_asset_sector
                sector_mapping = {asset: get_asset_sector(asset) for asset in returns_data.columns}
            sector_correlations = analyzer.correlation_breakdown_by_sector(sector_mapping)

            # Compile results
            analysis_results = {
                'correlation_metrics': {
                    'avg_correlation': float(corr_metrics.concentration_metrics['avg_correlation']),
                    'max_correlation': float(corr_metrics.concentration_metrics['max_correlation']),
                    'effective_assets': float(corr_metrics.concentration_metrics['effective_assets']),
                    'diversification_ratio': float(corr_metrics.diversification_ratio),
                    'condition_number': float(corr_metrics.condition_number)
                },
                'risk_decomposition': {
                    'total_risk': float(risk_decomp.total_risk),
                    'component_contributions': {k: float(v) for k, v in risk_decomp.component_contributions.items()},
                    'percentage_contributions': {k: float(v) for k, v in risk_decomp.percentage_contributions.items()}
                },
                'factor_analysis': {
                    'n_factors': factor_analysis['n_factors'],
                    'variance_explained': factor_analysis['variance_explained'].tolist(),
                    'cumulative_variance_explained': factor_analysis['cumulative_variance_explained'].tolist()
                },
                'sector_correlations': {k: {ks: float(vs) for ks, vs in v.items()} for k, v in sector_correlations.items()},
                'clustering': {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in clustering['clusters'].items()}
            }

            # Store results in database
            analysis_doc = {
                'portfolio_id': portfolio_id,
                'analysis_type': 'correlation_structure',
                'results': analysis_results,
                'estimation_method': estimation_method,
                'timestamp': datetime.utcnow()
            }

            await self.db.correlation_analysis.insert_one(analysis_doc)

            logger.info(f"Correlation analysis completed for portfolio {portfolio_id}")
            return analysis_results

        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return {}

    async def backtest_risk_models(
        self,
        portfolio_id: str,
        backtest_config: BacktestConfig = None,
        lookback_days: int = 500
    ) -> Dict:
        """
        Backtest VaR models for portfolio

        Args:
            portfolio_id: Portfolio identifier
            backtest_config: Backtesting configuration
            lookback_days: Historical data lookback period

        Returns:
            Backtesting results for multiple VaR models
        """
        try:
            logger.info(f"Starting risk model backtesting for portfolio {portfolio_id}")

            if backtest_config is None:
                backtest_config = BacktestConfig()

            # Get portfolio data and returns
            portfolio_data = await self._get_portfolio_data(portfolio_id)
            if not portfolio_data:
                return {}

            returns_data = await self._get_returns_data(portfolio_data['assets'], lookback_days)
            if returns_data.empty or len(returns_data) < 200:
                logger.warning("Insufficient data for backtesting")
                return {}

            # Portfolio weights
            weights = np.array([portfolio_data['weights'].get(asset, 0) for asset in returns_data.columns])
            weights = weights / np.sum(weights)  # Normalize

            # Initialize backtester
            backtester = RiskModelBacktester(backtest_config)

            # Test different VaR models
            var_models = {
                'historical_var': VaRModels.historical_var,
                'parametric_var': VaRModels.parametric_var,
                'modified_var': VaRModels.modified_var
            }

            model_results = {}

            for model_name, var_model in var_models.items():
                logger.info(f"Backtesting {model_name}")
                try:
                    results = backtester.backtest_var_model(
                        returns_data,
                        var_model,
                        weights,
                        confidence_level=backtest_config.confidence_level
                    )
                    model_results[model_name] = results
                except Exception as e:
                    logger.error(f"Error backtesting {model_name}: {e}")
                    continue

            if not model_results:
                return {}

            # Compare models
            comparison = backtester.compare_models(model_results)

            # Compile results for storage
            backtest_results = {
                'model_results': {
                    model_name: {
                        'violation_rate': float(results.violation_rate),
                        'expected_violations': int(results.expected_violations),
                        'actual_violations': int(results.actual_violations),
                        'test_statistics': {k: float(v) for k, v in results.test_statistics.items()},
                        'test_p_values': {k: float(v) for k, v in results.test_p_values.items()},
                        'test_results': results.test_results,
                        'coverage_statistics': {k: float(v) for k, v in results.coverage_statistics.items()}
                    }
                    for model_name, results in model_results.items()
                },
                'model_comparison': {
                    'overall_ranking': comparison['overall_ranking'],
                    'rankings': comparison['rankings'],
                    'summary': {k: {ks: float(vs) if isinstance(vs, (int, float)) else vs for ks, vs in v.items()}
                               for k, v in comparison['summary'].items()}
                }
            }

            # Store results in database
            backtest_doc = {
                'portfolio_id': portfolio_id,
                'analysis_type': 'var_backtesting',
                'results': backtest_results,
                'config': {
                    'confidence_level': backtest_config.confidence_level,
                    'lookback_window': backtest_config.lookback_window,
                    'test_types': backtest_config.test_types
                },
                'timestamp': datetime.utcnow()
            }

            await self.db.backtest_results.insert_one(backtest_doc)

            logger.info(f"Risk model backtesting completed for portfolio {portfolio_id}")
            return backtest_results

        except Exception as e:
            logger.error(f"Error in risk model backtesting: {e}")
            return {}

    async def setup_risk_monitoring(
        self,
        portfolio_id: str,
        monitoring_config: MonitoringConfig = None,
        webhook_url: str = None
    ) -> RiskMonitor:
        """
        Setup risk monitoring system for portfolio

        Args:
            portfolio_id: Portfolio identifier
            monitoring_config: Monitoring configuration
            webhook_url: Optional webhook URL for alerts

        Returns:
            Configured RiskMonitor instance
        """
        try:
            logger.info(f"Setting up risk monitoring for portfolio {portfolio_id}")

            # Create monitoring system
            monitor = await create_monitoring_system(webhook_url)

            if monitoring_config:
                monitor.config = monitoring_config

            # Customize thresholds based on portfolio characteristics
            portfolio_data = await self._get_portfolio_data(portfolio_id)
            if portfolio_data:
                # Adjust thresholds based on portfolio size and composition
                portfolio_value = portfolio_data.get('total_value', 0)
                asset_count = len(portfolio_data.get('assets', []))

                # More conservative thresholds for larger portfolios
                if portfolio_value > 1000000:  # $1M+
                    monitor.config.thresholds['portfolio_var_95'].warning_threshold = 0.03
                    monitor.config.thresholds['portfolio_var_95'].critical_threshold = 0.06

                # Adjust concentration thresholds for diversified portfolios
                if asset_count > 10:
                    monitor.config.thresholds['concentration_hhi'].warning_threshold = 0.25
                    monitor.config.thresholds['concentration_hhi'].critical_threshold = 0.40

            logger.info(f"Risk monitoring system configured for portfolio {portfolio_id}")
            return monitor

        except Exception as e:
            logger.error(f"Error setting up risk monitoring: {e}")
            return None

    async def generate_comprehensive_risk_report(
        self,
        portfolio_id: str,
        include_monte_carlo: bool = True,
        include_backtesting: bool = True,
        include_correlation_analysis: bool = True,
        lookback_days: int = 252
    ) -> Dict:
        """
        Generate comprehensive risk analysis report

        Args:
            portfolio_id: Portfolio identifier
            include_monte_carlo: Include Monte Carlo analysis
            include_backtesting: Include VaR model backtesting
            include_correlation_analysis: Include correlation analysis
            lookback_days: Historical data lookback period

        Returns:
            Comprehensive risk report
        """
        try:
            logger.info(f"Generating comprehensive risk report for portfolio {portfolio_id}")

            report = {
                'portfolio_id': portfolio_id,
                'report_date': datetime.utcnow(),
                'analysis_period_days': lookback_days
            }

            # Basic risk metrics
            basic_metrics = await self.calculate_portfolio_metrics(portfolio_id, lookback_days)
            report['basic_risk_metrics'] = basic_metrics

            # Monte Carlo analysis
            if include_monte_carlo:
                logger.info("Including Monte Carlo analysis")
                mc_config = SimulationConfig(n_simulations=10000, confidence_levels=[0.95, 0.99])
                mc_results = await self.run_monte_carlo_analysis(portfolio_id, mc_config, lookback_days)
                report['monte_carlo_analysis'] = mc_results

            # Correlation analysis
            if include_correlation_analysis:
                logger.info("Including correlation analysis")
                corr_results = await self.analyze_correlation_structure(portfolio_id, "ledoit_wolf", lookback_days)
                report['correlation_analysis'] = corr_results

            # Backtesting (requires more data)
            if include_backtesting and lookback_days >= 400:
                logger.info("Including VaR model backtesting")
                backtest_config = BacktestConfig(confidence_level=0.95, lookback_window=100)
                backtest_results = await self.backtest_risk_models(portfolio_id, backtest_config, lookback_days)
                report['backtesting_results'] = backtest_results

            # Sector analysis
            sector_metrics = await self.calculate_sector_allocation_risk(portfolio_id)
            report['sector_analysis'] = sector_metrics

            # Store comprehensive report
            report_doc = {
                'portfolio_id': portfolio_id,
                'report_type': 'comprehensive_risk_analysis',
                'report': report,
                'timestamp': datetime.utcnow()
            }

            await self.db.comprehensive_risk_reports.insert_one(report_doc)

            logger.info(f"Comprehensive risk report generated for portfolio {portfolio_id}")
            return report

        except Exception as e:
            logger.error(f"Error generating comprehensive risk report: {e}")
            return {}
