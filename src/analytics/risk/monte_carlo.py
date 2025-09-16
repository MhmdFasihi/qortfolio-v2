# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Monte Carlo simulation engine for portfolio risk analysis.
Implements various simulation techniques for VaR, stress testing, and scenario analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.stats import t, norm
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulations"""
    n_simulations: int = 10000
    time_horizon: int = 1  # days
    confidence_levels: List[float] = None
    random_seed: Optional[int] = None
    distribution_type: str = "normal"  # normal, t, skew_normal
    include_fat_tails: bool = True
    correlation_model: str = "historical"  # historical, shrinkage, factor_model

    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.95, 0.99, 0.999]

@dataclass
class SimulationResults:
    """Results from Monte Carlo simulation"""
    portfolio_returns: np.ndarray
    var_estimates: Dict[float, float]
    cvar_estimates: Dict[float, float]
    max_drawdown_estimates: Dict[str, float]
    percentiles: Dict[int, float]
    statistics: Dict[str, float]
    simulation_metadata: Dict[str, any]

class MonteCarloEngine:
    """Advanced Monte Carlo simulation engine for risk analysis"""

    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.historical_data: Optional[pd.DataFrame] = None
        self.correlation_matrix: Optional[np.ndarray] = None
        self.volatilities: Optional[np.ndarray] = None
        self.expected_returns: Optional[np.ndarray] = None

        if self.config.random_seed:
            np.random.seed(self.config.random_seed)

    def load_historical_data(self, returns_data: pd.DataFrame) -> None:
        """
        Load historical returns data for simulation calibration

        Args:
            returns_data: DataFrame with asset returns, assets as columns
        """
        self.historical_data = returns_data.copy()

        # Calculate key statistics
        self.expected_returns = returns_data.mean().values
        self.volatilities = returns_data.std().values
        self.correlation_matrix = returns_data.corr().values

        logger.info(f"Loaded historical data: {len(returns_data.columns)} assets, {len(returns_data)} observations")

    def estimate_parameters(self, method: str = "mle") -> Dict[str, any]:
        """
        Estimate distribution parameters from historical data

        Args:
            method: Parameter estimation method (mle, mom, robust)

        Returns:
            Dictionary of estimated parameters
        """
        if self.historical_data is None:
            raise ValueError("Historical data must be loaded first")

        params = {}

        for asset in self.historical_data.columns:
            returns = self.historical_data[asset].dropna()

            if self.config.distribution_type == "normal":
                params[asset] = {
                    'mu': returns.mean(),
                    'sigma': returns.std(),
                    'distribution': 'normal'
                }

            elif self.config.distribution_type == "t":
                # Fit t-distribution
                df, loc, scale = t.fit(returns)
                params[asset] = {
                    'mu': loc,
                    'sigma': scale,
                    'df': df,
                    'distribution': 't'
                }

            elif self.config.distribution_type == "skew_normal":
                # Fit skewed normal distribution
                from scipy.stats import skewnorm
                a, loc, scale = skewnorm.fit(returns)
                params[asset] = {
                    'mu': loc,
                    'sigma': scale,
                    'skewness': a,
                    'distribution': 'skew_normal'
                }

        logger.info(f"Estimated parameters using {method} method")
        return params

    def generate_scenarios(self,
                         portfolio_weights: np.ndarray,
                         custom_correlation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate Monte Carlo scenarios for portfolio returns

        Args:
            portfolio_weights: Array of portfolio weights
            custom_correlation: Optional custom correlation matrix

        Returns:
            Array of simulated portfolio returns
        """
        if self.historical_data is None:
            raise ValueError("Historical data must be loaded first")

        n_assets = len(portfolio_weights)
        correlation_matrix = custom_correlation if custom_correlation is not None else self.correlation_matrix

        # Ensure correlation matrix is positive definite
        correlation_matrix = self._ensure_positive_definite(correlation_matrix)

        # Generate correlated random variables
        if self.config.distribution_type == "normal":
            # Multivariate normal simulation
            cov_matrix = np.outer(self.volatilities, self.volatilities) * correlation_matrix
            scenarios = np.random.multivariate_normal(
                self.expected_returns * self.config.time_horizon,
                cov_matrix * self.config.time_horizon,
                self.config.n_simulations
            )

        elif self.config.distribution_type == "t":
            # Multivariate t-distribution simulation
            scenarios = self._simulate_multivariate_t(
                correlation_matrix,
                degrees_freedom=5.0  # Can be estimated from data
            )

        else:
            # Fall back to normal distribution
            cov_matrix = np.outer(self.volatilities, self.volatilities) * correlation_matrix
            scenarios = np.random.multivariate_normal(
                self.expected_returns * self.config.time_horizon,
                cov_matrix * self.config.time_horizon,
                self.config.n_simulations
            )

        # Calculate portfolio returns
        portfolio_returns = np.dot(scenarios, portfolio_weights)

        logger.info(f"Generated {self.config.n_simulations} scenarios for {n_assets} assets")
        return portfolio_returns

    def _simulate_multivariate_t(self,
                                correlation_matrix: np.ndarray,
                                degrees_freedom: float) -> np.ndarray:
        """Simulate from multivariate t-distribution"""
        n_assets = len(correlation_matrix)

        # Generate from multivariate normal
        normal_samples = np.random.multivariate_normal(
            np.zeros(n_assets),
            correlation_matrix,
            self.config.n_simulations
        )

        # Generate chi-squared samples
        chi2_samples = np.random.chisquare(degrees_freedom, self.config.n_simulations)

        # Transform to t-distribution
        t_samples = normal_samples * np.sqrt(degrees_freedom / chi2_samples[:, np.newaxis])

        # Scale and shift
        scenarios = np.zeros_like(t_samples)
        for i in range(n_assets):
            scenarios[:, i] = (self.expected_returns[i] * self.config.time_horizon +
                             self.volatilities[i] * np.sqrt(self.config.time_horizon) * t_samples[:, i])

        return scenarios

    def _ensure_positive_definite(self, matrix: np.ndarray, min_eigenvalue: float = 1e-8) -> np.ndarray:
        """Ensure correlation matrix is positive definite"""
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        eigenvals = np.maximum(eigenvals, min_eigenvalue)
        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    def calculate_risk_metrics(self, portfolio_returns: np.ndarray) -> SimulationResults:
        """
        Calculate comprehensive risk metrics from simulation results

        Args:
            portfolio_returns: Array of simulated portfolio returns

        Returns:
            SimulationResults object with all metrics
        """
        # Sort returns for VaR calculation
        sorted_returns = np.sort(portfolio_returns)

        # Calculate VaR at different confidence levels
        var_estimates = {}
        cvar_estimates = {}

        for confidence in self.config.confidence_levels:
            var_index = int((1 - confidence) * len(sorted_returns))
            var_estimates[confidence] = -sorted_returns[var_index]  # Negative for loss

            # CVaR (Expected Shortfall) - average of tail losses
            tail_losses = sorted_returns[:var_index]
            cvar_estimates[confidence] = -np.mean(tail_losses) if len(tail_losses) > 0 else 0

        # Calculate percentiles
        percentiles = {}
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            percentiles[p] = np.percentile(portfolio_returns, p)

        # Calculate max drawdown estimates
        max_drawdown_estimates = self._calculate_drawdown_metrics(portfolio_returns)

        # Basic statistics
        statistics = {
            'mean': np.mean(portfolio_returns),
            'std': np.std(portfolio_returns),
            'skewness': stats.skew(portfolio_returns),
            'kurtosis': stats.kurtosis(portfolio_returns),
            'min': np.min(portfolio_returns),
            'max': np.max(portfolio_returns),
            'sharpe_ratio': np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0
        }

        # Simulation metadata
        metadata = {
            'n_simulations': self.config.n_simulations,
            'time_horizon': self.config.time_horizon,
            'distribution_type': self.config.distribution_type,
            'simulation_date': datetime.now(),
            'random_seed': self.config.random_seed
        }

        return SimulationResults(
            portfolio_returns=portfolio_returns,
            var_estimates=var_estimates,
            cvar_estimates=cvar_estimates,
            max_drawdown_estimates=max_drawdown_estimates,
            percentiles=percentiles,
            statistics=statistics,
            simulation_metadata=metadata
        )

    def _calculate_drawdown_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate drawdown metrics from returns"""
        # Simulate cumulative returns path
        cumulative_returns = np.cumsum(returns.reshape(-1, 1))
        running_max = np.maximum.accumulate(cumulative_returns.flatten())
        drawdowns = (cumulative_returns.flatten() - running_max) / (running_max + 1e-8)

        return {
            'max_drawdown': np.min(drawdowns),
            'avg_drawdown': np.mean(drawdowns[drawdowns < 0]),
            'drawdown_99': np.percentile(drawdowns, 1)
        }

    def stress_test(self,
                   portfolio_weights: np.ndarray,
                   stress_scenarios: Dict[str, Dict[str, float]]) -> Dict[str, SimulationResults]:
        """
        Run stress tests with predefined scenarios

        Args:
            portfolio_weights: Portfolio allocation weights
            stress_scenarios: Dictionary of stress scenarios

        Returns:
            Dictionary of simulation results for each scenario
        """
        results = {}

        for scenario_name, scenario_params in stress_scenarios.items():
            logger.info(f"Running stress test: {scenario_name}")

            # Modify parameters for stress scenario
            stressed_returns = self.expected_returns.copy()
            stressed_volatilities = self.volatilities.copy()
            stressed_correlation = self.correlation_matrix.copy()

            # Apply stress parameters
            if 'return_shock' in scenario_params:
                stressed_returns *= (1 + scenario_params['return_shock'])

            if 'volatility_shock' in scenario_params:
                stressed_volatilities *= (1 + scenario_params['volatility_shock'])

            if 'correlation_shock' in scenario_params:
                # Increase correlations during stress
                correlation_shock = scenario_params['correlation_shock']
                stressed_correlation = (stressed_correlation * (1 - correlation_shock) +
                                      correlation_shock * np.ones_like(stressed_correlation))
                np.fill_diagonal(stressed_correlation, 1.0)

            # Temporarily update engine parameters
            original_returns = self.expected_returns.copy()
            original_volatilities = self.volatilities.copy()
            original_correlation = self.correlation_matrix.copy()

            self.expected_returns = stressed_returns
            self.volatilities = stressed_volatilities
            self.correlation_matrix = stressed_correlation

            try:
                # Generate stressed scenarios
                stressed_portfolio_returns = self.generate_scenarios(portfolio_weights)
                results[scenario_name] = self.calculate_risk_metrics(stressed_portfolio_returns)

            finally:
                # Restore original parameters
                self.expected_returns = original_returns
                self.volatilities = original_volatilities
                self.correlation_matrix = original_correlation

        return results

    def scenario_analysis(self,
                         portfolio_weights: np.ndarray,
                         custom_scenarios: List[np.ndarray]) -> Dict[str, float]:
        """
        Analyze portfolio performance under specific return scenarios

        Args:
            portfolio_weights: Portfolio allocation weights
            custom_scenarios: List of return scenario vectors

        Returns:
            Dictionary of scenario results
        """
        results = {}

        for i, scenario in enumerate(custom_scenarios):
            portfolio_return = np.dot(scenario, portfolio_weights)
            results[f"scenario_{i+1}"] = portfolio_return

        return results

class StressTesting:
    """Comprehensive stress testing framework"""

    @staticmethod
    def get_predefined_stress_scenarios() -> Dict[str, Dict[str, float]]:
        """Get collection of predefined stress scenarios"""
        return {
            "market_crash": {
                "return_shock": -0.30,  # 30% market decline
                "volatility_shock": 2.0,  # Double volatility
                "correlation_shock": 0.8   # High correlation
            },
            "crypto_winter": {
                "return_shock": -0.80,  # Severe crypto decline
                "volatility_shock": 3.0,  # Triple volatility
                "correlation_shock": 0.9   # Very high correlation
            },
            "liquidity_crisis": {
                "return_shock": -0.15,
                "volatility_shock": 1.5,
                "correlation_shock": 0.7
            },
            "black_swan": {
                "return_shock": -0.50,
                "volatility_shock": 4.0,
                "correlation_shock": 0.95
            },
            "inflation_shock": {
                "return_shock": -0.20,
                "volatility_shock": 1.8,
                "correlation_shock": 0.6
            },
            "regulatory_shock": {
                "return_shock": -0.25,
                "volatility_shock": 2.5,
                "correlation_shock": 0.75
            }
        }

    @staticmethod
    def generate_historical_scenarios(returns_data: pd.DataFrame,
                                    lookback_days: int = 365) -> List[np.ndarray]:
        """Generate scenarios based on historical worst periods"""
        scenarios = []

        # Find worst performing periods
        rolling_returns = returns_data.rolling(window=lookback_days).sum()
        worst_periods = rolling_returns.min(axis=1).nsmallest(10)

        for date in worst_periods.index:
            if date in rolling_returns.index:
                scenario = rolling_returns.loc[date].values
                scenarios.append(scenario)

        return scenarios

async def run_comprehensive_risk_analysis(portfolio_weights: np.ndarray,
                                        returns_data: pd.DataFrame,
                                        config: SimulationConfig = None) -> Dict[str, any]:
    """
    Run comprehensive Monte Carlo risk analysis

    Args:
        portfolio_weights: Portfolio allocation weights
        returns_data: Historical returns data
        config: Simulation configuration

    Returns:
        Dictionary with comprehensive risk analysis results
    """
    if config is None:
        config = SimulationConfig()

    # Initialize Monte Carlo engine
    mc_engine = MonteCarloEngine(config)
    mc_engine.load_historical_data(returns_data)

    # Base case simulation
    logger.info("Running base case Monte Carlo simulation")
    base_returns = mc_engine.generate_scenarios(portfolio_weights)
    base_results = mc_engine.calculate_risk_metrics(base_returns)

    # Stress testing
    logger.info("Running stress tests")
    stress_scenarios = StressTesting.get_predefined_stress_scenarios()
    stress_results = mc_engine.stress_test(portfolio_weights, stress_scenarios)

    # Historical scenario analysis
    logger.info("Running historical scenario analysis")
    historical_scenarios = StressTesting.generate_historical_scenarios(returns_data)
    scenario_results = mc_engine.scenario_analysis(portfolio_weights, historical_scenarios)

    # Compile comprehensive results
    comprehensive_results = {
        "base_case": base_results,
        "stress_tests": stress_results,
        "historical_scenarios": scenario_results,
        "analysis_metadata": {
            "analysis_date": datetime.now(),
            "portfolio_size": len(portfolio_weights),
            "data_period": f"{returns_data.index[0]} to {returns_data.index[-1]}",
            "simulation_config": config.__dict__
        }
    }

    logger.info("Comprehensive risk analysis completed")
    return comprehensive_results

if __name__ == "__main__":
    # Example usage
    import asyncio

    async def main():
        # Generate sample data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
        assets = ['BTC', 'ETH', 'SOL', 'AVAX']

        # Simulate correlated returns
        returns_data = pd.DataFrame(
            np.random.multivariate_normal(
                [0.001, 0.0008, 0.0012, 0.0015],  # Expected daily returns
                [[0.04, 0.03, 0.025, 0.02],        # Correlation matrix
                 [0.03, 0.035, 0.028, 0.022],
                 [0.025, 0.028, 0.045, 0.025],
                 [0.02, 0.022, 0.025, 0.05]],
                365
            ),
            index=dates,
            columns=assets
        )

        # Portfolio weights
        portfolio_weights = np.array([0.4, 0.3, 0.2, 0.1])

        # Run comprehensive analysis
        config = SimulationConfig(n_simulations=5000, confidence_levels=[0.95, 0.99])
        results = await run_comprehensive_risk_analysis(portfolio_weights, returns_data, config)

        # Display results
        print("=== MONTE CARLO RISK ANALYSIS ===")
        print(f"Base Case VaR (95%): {results['base_case'].var_estimates[0.95]:.4f}")
        print(f"Base Case CVaR (95%): {results['base_case'].cvar_estimates[0.95]:.4f}")
        print(f"Expected Return: {results['base_case'].statistics['mean']:.4f}")
        print(f"Volatility: {results['base_case'].statistics['std']:.4f}")
        print(f"Sharpe Ratio: {results['base_case'].statistics['sharpe_ratio']:.4f}")

        print("\n=== STRESS TEST RESULTS ===")
        for scenario, result in results['stress_tests'].items():
            print(f"{scenario.upper()}: VaR(95%) = {result.var_estimates[0.95]:.4f}")

    asyncio.run(main())