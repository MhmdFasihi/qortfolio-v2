# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Backtesting framework for risk models and VaR validation.
Implements various backtesting methodologies and statistical tests.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.stats import chi2, binom
import warnings

logger = logging.getLogger(__name__)

@dataclass
class BacktestResults:
    """Results from risk model backtesting"""
    var_forecasts: np.ndarray
    actual_returns: np.ndarray
    violations: np.ndarray
    violation_rate: float
    expected_violations: int
    actual_violations: int
    test_statistics: Dict[str, float]
    test_p_values: Dict[str, float]
    test_results: Dict[str, bool]
    coverage_statistics: Dict[str, float]

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    confidence_level: float = 0.95
    lookback_window: int = 365
    rebalance_frequency: int = 1  # days
    min_observations: int = 100
    test_types: List[str] = None

    def __post_init__(self):
        if self.test_types is None:
            self.test_types = ['kupiec', 'christoffersen', 'dynamic_quantile']

class RiskModelBacktester:
    """Comprehensive backtesting framework for risk models"""

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

    def backtest_var_model(self,
                          returns_data: pd.DataFrame,
                          var_model: Callable,
                          portfolio_weights: Union[np.ndarray, pd.DataFrame],
                          **model_kwargs) -> BacktestResults:
        """
        Backtest a VaR model using rolling window approach

        Args:
            returns_data: Historical returns data
            var_model: Function that estimates VaR given returns data
            portfolio_weights: Portfolio weights (static or time-varying)
            **model_kwargs: Additional arguments for VaR model

        Returns:
            BacktestResults with comprehensive test statistics
        """
        logger.info(f"Starting VaR backtesting with {self.config.confidence_level:.1%} confidence level")

        # Prepare data
        clean_data = returns_data.dropna()
        n_obs = len(clean_data)

        # Handle portfolio weights
        if isinstance(portfolio_weights, np.ndarray):
            # Static weights
            weights_series = pd.DataFrame(
                np.tile(portfolio_weights, (n_obs, 1)),
                index=clean_data.index,
                columns=clean_data.columns
            )
        else:
            # Time-varying weights
            weights_series = portfolio_weights.reindex(clean_data.index).fillna(method='ffill')

        # Rolling window backtest
        var_forecasts = []
        actual_returns = []
        test_dates = []

        start_idx = self.config.lookback_window
        end_idx = n_obs

        for i in range(start_idx, end_idx, self.config.rebalance_frequency):
            # Historical window
            hist_returns = clean_data.iloc[i-self.config.lookback_window:i]
            current_weights = weights_series.iloc[i].values

            # Calculate portfolio returns for historical window
            portfolio_returns = (hist_returns * current_weights).sum(axis=1)

            # Estimate VaR
            try:
                var_forecast = var_model(portfolio_returns, **model_kwargs)
                if isinstance(var_forecast, dict):
                    var_forecast = var_forecast.get(self.config.confidence_level, 0)
            except Exception as e:
                logger.warning(f"VaR estimation failed at index {i}: {e}")
                var_forecast = np.nan

            # Actual return for next period
            if i < len(clean_data):
                next_weights = current_weights
                actual_return = (clean_data.iloc[i] * next_weights).sum()
            else:
                actual_return = np.nan

            var_forecasts.append(var_forecast)
            actual_returns.append(actual_return)
            test_dates.append(clean_data.index[i])

        # Convert to arrays
        var_forecasts = np.array(var_forecasts)
        actual_returns = np.array(actual_returns)

        # Remove NaN values
        valid_mask = ~(np.isnan(var_forecasts) | np.isnan(actual_returns))
        var_forecasts = var_forecasts[valid_mask]
        actual_returns = actual_returns[valid_mask]
        valid_dates = np.array(test_dates)[valid_mask]

        # Calculate violations (VaR is positive, return is negative for loss)
        violations = actual_returns < -var_forecasts
        violation_rate = np.mean(violations)
        expected_violation_rate = 1 - self.config.confidence_level

        # Run statistical tests
        test_results = self._run_backtest_statistics(
            var_forecasts, actual_returns, violations, expected_violation_rate
        )

        # Coverage statistics
        coverage_stats = self._calculate_coverage_statistics(
            var_forecasts, actual_returns, violations
        )

        logger.info(f"Backtesting completed: {len(var_forecasts)} observations, "
                   f"{np.sum(violations)} violations ({violation_rate:.2%})")

        return BacktestResults(
            var_forecasts=var_forecasts,
            actual_returns=actual_returns,
            violations=violations,
            violation_rate=violation_rate,
            expected_violations=int(expected_violation_rate * len(violations)),
            actual_violations=int(np.sum(violations)),
            test_statistics=test_results['statistics'],
            test_p_values=test_results['p_values'],
            test_results=test_results['results'],
            coverage_statistics=coverage_stats
        )

    def _run_backtest_statistics(self,
                               var_forecasts: np.ndarray,
                               actual_returns: np.ndarray,
                               violations: np.ndarray,
                               expected_rate: float) -> Dict[str, Dict[str, float]]:
        """Run comprehensive backtesting statistical tests"""

        test_stats = {}
        p_values = {}
        test_results = {}
        n_obs = len(violations)
        n_violations = np.sum(violations)

        # 1. Kupiec Test (Unconditional Coverage)
        if 'kupiec' in self.config.test_types:
            kupiec_stat, kupiec_p = self._kupiec_test(n_violations, n_obs, expected_rate)
            test_stats['kupiec'] = kupiec_stat
            p_values['kupiec'] = kupiec_p
            test_results['kupiec'] = kupiec_p > 0.05  # Null: correct coverage

        # 2. Christoffersen Test (Independence)
        if 'christoffersen' in self.config.test_types:
            chris_stat, chris_p = self._christoffersen_test(violations)
            test_stats['christoffersen'] = chris_stat
            p_values['christoffersen'] = chris_p
            test_results['christoffersen'] = chris_p > 0.05  # Null: independence

        # 3. Combined Christoffersen Test (Coverage + Independence)
        if 'combined_christoffersen' in self.config.test_types:
            combined_stat = test_stats.get('kupiec', 0) + test_stats.get('christoffersen', 0)
            combined_p = 1 - chi2.cdf(combined_stat, df=2)
            test_stats['combined_christoffersen'] = combined_stat
            p_values['combined_christoffersen'] = combined_p
            test_results['combined_christoffersen'] = combined_p > 0.05

        # 4. Dynamic Quantile Test
        if 'dynamic_quantile' in self.config.test_types:
            dq_stat, dq_p = self._dynamic_quantile_test(var_forecasts, actual_returns, expected_rate)
            test_stats['dynamic_quantile'] = dq_stat
            p_values['dynamic_quantile'] = dq_p
            test_results['dynamic_quantile'] = dq_p > 0.05

        # 5. Traffic Light Test (Basel)
        if 'traffic_light' in self.config.test_types:
            tl_zone, tl_multiplier = self._traffic_light_test(n_violations, n_obs)
            test_stats['traffic_light_zone'] = tl_zone
            test_stats['traffic_light_multiplier'] = tl_multiplier
            test_results['traffic_light'] = tl_zone <= 1  # Green or Yellow zone

        return {
            'statistics': test_stats,
            'p_values': p_values,
            'results': test_results
        }

    def _kupiec_test(self, violations: int, observations: int, expected_rate: float) -> Tuple[float, float]:
        """
        Kupiec unconditional coverage test

        H0: Violation rate equals expected rate
        """
        if violations == 0:
            return 0.0, 1.0

        violation_rate = violations / observations

        # Likelihood ratio test statistic
        if violation_rate == 0:
            lr_stat = 2 * observations * np.log(1 - expected_rate)
        elif violation_rate == 1:
            lr_stat = 2 * observations * np.log(expected_rate)
        else:
            lr_stat = 2 * (violations * np.log(violation_rate / expected_rate) +
                          (observations - violations) * np.log((1 - violation_rate) / (1 - expected_rate)))

        # Chi-squared test with 1 degree of freedom
        p_value = 1 - chi2.cdf(lr_stat, df=1)

        return lr_stat, p_value

    def _christoffersen_test(self, violations: np.ndarray) -> Tuple[float, float]:
        """
        Christoffersen independence test

        H0: Violations are independent (no clustering)
        """
        # Count transitions
        n00 = n01 = n10 = n11 = 0

        for i in range(1, len(violations)):
            if not violations[i-1] and not violations[i]:
                n00 += 1
            elif not violations[i-1] and violations[i]:
                n01 += 1
            elif violations[i-1] and not violations[i]:
                n10 += 1
            elif violations[i-1] and violations[i]:
                n11 += 1

        # Avoid division by zero
        n0 = n00 + n01
        n1 = n10 + n11

        if n0 == 0 or n1 == 0 or (n01 + n11) == 0:
            return 0.0, 1.0

        # Transition probabilities
        pi_01 = n01 / n0 if n0 > 0 else 0
        pi_11 = n11 / n1 if n1 > 0 else 0
        pi = (n01 + n11) / (n0 + n1)

        # Likelihood ratio test statistic
        if pi_01 == 0 or pi_11 == 0 or pi == 0:
            lr_stat = 0.0
        else:
            lr_stat = 2 * (n01 * np.log(pi_01 / pi) + n11 * np.log(pi_11 / pi))

        # Chi-squared test with 1 degree of freedom
        p_value = 1 - chi2.cdf(lr_stat, df=1)

        return lr_stat, p_value

    def _dynamic_quantile_test(self,
                             var_forecasts: np.ndarray,
                             actual_returns: np.ndarray,
                             expected_rate: float) -> Tuple[float, float]:
        """
        Dynamic Quantile test (Engle and Manganelli, 2004)

        Tests if VaR forecasts are optimal given information set
        """
        # Create hit sequence
        hits = (actual_returns < -var_forecasts).astype(float) - expected_rate

        # Simple DQ test with lagged hits and VaR forecasts
        n = len(hits)

        if n < 10:  # Need sufficient observations
            return 0.0, 1.0

        # Design matrix: constant, lagged hit, VaR forecast
        X = np.ones((n-1, 3))
        X[1:, 1] = hits[:-2]  # Lagged hit
        X[:, 2] = var_forecasts[1:]  # VaR forecast

        y = hits[1:]  # Current hit (centered)

        try:
            # OLS regression
            beta = np.linalg.solve(X.T @ X, X.T @ y)
            residuals = y - X @ beta

            # DQ test statistic
            dq_stat = (n-1) * (1 - np.var(residuals) / np.var(y))
            p_value = 1 - chi2.cdf(dq_stat, df=2)

        except (np.linalg.LinAlgError, ZeroDivisionError):
            dq_stat = 0.0
            p_value = 1.0

        return dq_stat, p_value

    def _traffic_light_test(self, violations: int, observations: int) -> Tuple[int, float]:
        """
        Basel Traffic Light Test

        Returns zone (0=Green, 1=Yellow, 2=Red) and multiplier
        """
        # Basel zones based on number of violations
        if violations <= 4:
            zone = 0  # Green
            multiplier = 0.0
        elif violations <= 9:
            zone = 1  # Yellow
            multiplier = 0.4 + 0.1 * (violations - 4)
        else:
            zone = 2  # Red
            multiplier = 1.0

        return zone, multiplier

    def _calculate_coverage_statistics(self,
                                     var_forecasts: np.ndarray,
                                     actual_returns: np.ndarray,
                                     violations: np.ndarray) -> Dict[str, float]:
        """Calculate additional coverage statistics"""

        stats = {}

        # Violation clustering
        violation_runs = self._calculate_violation_runs(violations)
        stats['avg_violation_run_length'] = np.mean(violation_runs) if violation_runs else 0
        stats['max_violation_run_length'] = np.max(violation_runs) if violation_runs else 0

        # Loss severity (conditional on violation)
        if np.any(violations):
            violation_losses = actual_returns[violations]
            violation_vars = var_forecasts[violations]
            excess_losses = -violation_losses - violation_vars

            stats['avg_excess_loss'] = np.mean(excess_losses)
            stats['max_excess_loss'] = np.max(excess_losses)
            stats['avg_loss_ratio'] = np.mean(-violation_losses / violation_vars)
        else:
            stats['avg_excess_loss'] = 0.0
            stats['max_excess_loss'] = 0.0
            stats['avg_loss_ratio'] = 0.0

        # Time between violations
        violation_indices = np.where(violations)[0]
        if len(violation_indices) > 1:
            time_between = np.diff(violation_indices)
            stats['avg_time_between_violations'] = np.mean(time_between)
            stats['min_time_between_violations'] = np.min(time_between)
        else:
            stats['avg_time_between_violations'] = len(violations)
            stats['min_time_between_violations'] = len(violations)

        return stats

    def _calculate_violation_runs(self, violations: np.ndarray) -> List[int]:
        """Calculate lengths of consecutive violation runs"""
        if not np.any(violations):
            return []

        runs = []
        current_run = 0

        for violation in violations:
            if violation:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                    current_run = 0

        # Add final run if it ends with violations
        if current_run > 0:
            runs.append(current_run)

        return runs

    def generate_backtest_report(self, results: BacktestResults) -> str:
        """Generate comprehensive backtesting report"""

        report = []
        report.append("=" * 60)
        report.append("VaR MODEL BACKTESTING REPORT")
        report.append("=" * 60)

        # Basic statistics
        report.append(f"\nBASIC STATISTICS:")
        report.append(f"Confidence Level: {self.config.confidence_level:.1%}")
        report.append(f"Total Observations: {len(results.actual_returns)}")
        report.append(f"Expected Violations: {results.expected_violations}")
        report.append(f"Actual Violations: {results.actual_violations}")
        report.append(f"Violation Rate: {results.violation_rate:.2%}")
        report.append(f"Expected Rate: {1-self.config.confidence_level:.2%}")

        # Test results
        report.append(f"\nSTATISTICAL TESTS:")
        for test_name, passed in results.test_results.items():
            status = "PASS" if passed else "FAIL"
            p_val = results.test_p_values.get(test_name, 0)
            stat = results.test_statistics.get(test_name, 0)
            report.append(f"{test_name.upper()}: {status} (stat={stat:.3f}, p={p_val:.3f})")

        # Coverage statistics
        report.append(f"\nCOVERAGE STATISTICS:")
        for stat_name, value in results.coverage_statistics.items():
            report.append(f"{stat_name.replace('_', ' ').title()}: {value:.3f}")

        # Overall assessment
        passed_tests = sum(results.test_results.values())
        total_tests = len(results.test_results)

        report.append(f"\nOVERALL ASSESSMENT:")
        report.append(f"Tests Passed: {passed_tests}/{total_tests}")

        if passed_tests == total_tests:
            assessment = "MODEL VALIDATION: ACCEPTABLE"
        elif passed_tests >= total_tests * 0.7:
            assessment = "MODEL VALIDATION: MARGINAL"
        else:
            assessment = "MODEL VALIDATION: INADEQUATE"

        report.append(f"{assessment}")
        report.append("=" * 60)

        return "\n".join(report)

    def compare_models(self,
                      model_results: Dict[str, BacktestResults]) -> Dict[str, any]:
        """
        Compare multiple VaR models based on backtesting results

        Args:
            model_results: Dictionary of {model_name: BacktestResults}

        Returns:
            Model comparison results
        """
        comparison = {
            'summary': {},
            'rankings': {},
            'detailed_comparison': {}
        }

        # Extract key metrics for comparison
        model_metrics = {}

        for model_name, results in model_results.items():
            metrics = {
                'violation_rate': results.violation_rate,
                'kupiec_p_value': results.test_p_values.get('kupiec', 0),
                'christoffersen_p_value': results.test_p_values.get('christoffersen', 0),
                'avg_excess_loss': results.coverage_statistics.get('avg_excess_loss', 0),
                'max_violation_run': results.coverage_statistics.get('max_violation_run_length', 0),
                'tests_passed': sum(results.test_results.values()),
                'total_tests': len(results.test_results)
            }
            model_metrics[model_name] = metrics

        # Rank models
        rankings = {}

        # By violation rate closeness to expected
        expected_rate = 1 - self.config.confidence_level
        rate_errors = {name: abs(metrics['violation_rate'] - expected_rate)
                      for name, metrics in model_metrics.items()}
        rankings['violation_rate'] = sorted(rate_errors, key=rate_errors.get)

        # By number of tests passed
        test_scores = {name: metrics['tests_passed'] / metrics['total_tests']
                      for name, metrics in model_metrics.items()}
        rankings['test_success'] = sorted(test_scores, key=test_scores.get, reverse=True)

        # By excess loss (lower is better)
        excess_losses = {name: metrics['avg_excess_loss']
                        for name, metrics in model_metrics.items()}
        rankings['excess_loss'] = sorted(excess_losses, key=excess_losses.get)

        comparison['summary'] = model_metrics
        comparison['rankings'] = rankings

        # Overall ranking (simple average of ranks)
        overall_scores = {}
        for model_name in model_metrics:
            rank_sum = (rankings['violation_rate'].index(model_name) +
                       rankings['test_success'].index(model_name) +
                       rankings['excess_loss'].index(model_name))
            overall_scores[model_name] = rank_sum

        comparison['overall_ranking'] = sorted(overall_scores, key=overall_scores.get)

        return comparison

# Example VaR models for backtesting
class VaRModels:
    """Collection of VaR models for backtesting"""

    @staticmethod
    def historical_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Historical simulation VaR"""
        return np.percentile(returns, (1 - confidence_level) * 100)

    @staticmethod
    def parametric_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Parametric VaR assuming normal distribution"""
        return stats.norm.ppf(1 - confidence_level, returns.mean(), returns.std())

    @staticmethod
    def modified_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Modified VaR accounting for skewness and kurtosis"""
        mean = returns.mean()
        std = returns.std()
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)

        z = stats.norm.ppf(confidence_level)
        modified_z = (z +
                     (z**2 - 1) * skew / 6 +
                     (z**3 - 3*z) * kurt / 24 -
                     (2*z**3 - 5*z) * skew**2 / 36)

        return mean + modified_z * std

if __name__ == "__main__":
    # Example backtesting
    np.random.seed(42)

    # Generate sample returns data
    dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
    returns_data = pd.DataFrame(
        np.random.normal(0.001, 0.02, (500, 4)),
        index=dates,
        columns=['BTC', 'ETH', 'SOL', 'AVAX']
    )

    # Portfolio weights
    weights = np.array([0.4, 0.3, 0.2, 0.1])

    # Backtest configuration
    config = BacktestConfig(
        confidence_level=0.95,
        lookback_window=100,
        test_types=['kupiec', 'christoffersen', 'combined_christoffersen']
    )

    # Initialize backtester
    backtester = RiskModelBacktester(config)

    # Test historical VaR model
    results = backtester.backtest_var_model(
        returns_data,
        VaRModels.historical_var,
        weights,
        confidence_level=0.95
    )

    # Generate report
    report = backtester.generate_backtest_report(results)
    print(report)