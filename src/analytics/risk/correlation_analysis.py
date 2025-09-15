# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Correlation analysis and risk decomposition for portfolio risk management.
Advanced correlation modeling, regime detection, and risk attribution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.mixture import GaussianMixture
import warnings

logger = logging.getLogger(__name__)

@dataclass
class CorrelationMetrics:
    """Container for correlation analysis results"""
    correlation_matrix: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    condition_number: float
    effective_rank: float
    diversification_ratio: float
    concentration_metrics: Dict[str, float]

@dataclass
class RiskDecomposition:
    """Risk decomposition results"""
    total_risk: float
    component_contributions: Dict[str, float]
    marginal_contributions: Dict[str, float]
    percentage_contributions: Dict[str, float]
    interaction_effects: np.ndarray

class CorrelationAnalyzer:
    """Advanced correlation analysis and risk decomposition"""

    def __init__(self, estimation_method: str = "ledoit_wolf"):
        """
        Initialize correlation analyzer

        Args:
            estimation_method: Method for covariance estimation
                             ('sample', 'ledoit_wolf', 'oas', 'shrunk')
        """
        self.estimation_method = estimation_method
        self.returns_data: Optional[pd.DataFrame] = None
        self.correlation_matrix: Optional[np.ndarray] = None
        self.covariance_matrix: Optional[np.ndarray] = None

    def load_data(self, returns_data: pd.DataFrame) -> None:
        """Load returns data for analysis"""
        self.returns_data = returns_data.copy()
        logger.info(f"Loaded data: {returns_data.shape[1]} assets, {returns_data.shape[0]} observations")

    def estimate_correlation_matrix(self, **kwargs) -> np.ndarray:
        """
        Estimate correlation matrix using specified method

        Returns:
            Correlation matrix
        """
        if self.returns_data is None:
            raise ValueError("Data must be loaded first")

        # Clean data
        clean_data = self.returns_data.dropna()

        if self.estimation_method == "sample":
            self.correlation_matrix = clean_data.corr().values
            self.covariance_matrix = clean_data.cov().values

        elif self.estimation_method == "ledoit_wolf":
            lw = LedoitWolf()
            self.covariance_matrix = lw.fit(clean_data).covariance_
            # Convert to correlation
            diag_sqrt = np.sqrt(np.diag(self.covariance_matrix))
            self.correlation_matrix = (self.covariance_matrix /
                                     np.outer(diag_sqrt, diag_sqrt))

        elif self.estimation_method == "oas":
            oas = OAS()
            self.covariance_matrix = oas.fit(clean_data).covariance_
            # Convert to correlation
            diag_sqrt = np.sqrt(np.diag(self.covariance_matrix))
            self.correlation_matrix = (self.covariance_matrix /
                                     np.outer(diag_sqrt, diag_sqrt))

        elif self.estimation_method == "shrunk":
            shrinkage = kwargs.get('shrinkage', 0.1)
            shrunk_cov = ShrunkCovariance(shrinkage=shrinkage)
            self.covariance_matrix = shrunk_cov.fit(clean_data).covariance_
            # Convert to correlation
            diag_sqrt = np.sqrt(np.diag(self.covariance_matrix))
            self.correlation_matrix = (self.covariance_matrix /
                                     np.outer(diag_sqrt, diag_sqrt))

        else:
            raise ValueError(f"Unknown estimation method: {self.estimation_method}")

        # Ensure diagonal is exactly 1
        np.fill_diagonal(self.correlation_matrix, 1.0)

        logger.info(f"Estimated correlation matrix using {self.estimation_method} method")
        return self.correlation_matrix

    def analyze_correlation_structure(self) -> CorrelationMetrics:
        """
        Comprehensive correlation structure analysis

        Returns:
            CorrelationMetrics with analysis results
        """
        if self.correlation_matrix is None:
            self.estimate_correlation_matrix()

        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(self.correlation_matrix)
        eigenvals = eigenvals[::-1]  # Sort descending
        eigenvecs = eigenvecs[:, ::-1]

        # Matrix condition and effective rank
        condition_number = eigenvals[0] / eigenvals[-1] if eigenvals[-1] > 0 else np.inf
        effective_rank = np.sum(eigenvals)**2 / np.sum(eigenvals**2)

        # Diversification ratio calculation
        portfolio_weights = np.ones(len(self.correlation_matrix)) / len(self.correlation_matrix)
        weighted_avg_vol = np.sqrt(np.dot(portfolio_weights, np.diag(self.covariance_matrix)))
        portfolio_vol = np.sqrt(np.dot(portfolio_weights,
                                     np.dot(self.covariance_matrix, portfolio_weights)))
        diversification_ratio = weighted_avg_vol / portfolio_vol

        # Concentration metrics
        concentration_metrics = self._calculate_concentration_metrics()

        return CorrelationMetrics(
            correlation_matrix=self.correlation_matrix,
            eigenvalues=eigenvals,
            eigenvectors=eigenvecs,
            condition_number=condition_number,
            effective_rank=effective_rank,
            diversification_ratio=diversification_ratio,
            concentration_metrics=concentration_metrics
        )

    def _calculate_concentration_metrics(self) -> Dict[str, float]:
        """Calculate correlation concentration metrics"""
        if self.correlation_matrix is None:
            return {}

        # Average correlation
        n = len(self.correlation_matrix)
        off_diagonal = self.correlation_matrix[np.triu_indices(n, k=1)]
        avg_correlation = np.mean(off_diagonal)

        # Maximum correlation
        max_correlation = np.max(off_diagonal)

        # Correlation dispersion
        correlation_std = np.std(off_diagonal)

        # Effective number of assets (based on correlation)
        eigenvals, _ = np.linalg.eigh(self.correlation_matrix)
        effective_assets = (np.sum(eigenvals)**2) / np.sum(eigenvals**2)

        # Absorption ratio (fraction of variance explained by first eigenvalue)
        absorption_ratio = eigenvals[-1] / np.sum(eigenvals)

        return {
            "avg_correlation": avg_correlation,
            "max_correlation": max_correlation,
            "correlation_std": correlation_std,
            "effective_assets": effective_assets,
            "absorption_ratio": absorption_ratio,
            "condition_number": eigenvals[-1] / eigenvals[0] if eigenvals[0] > 0 else np.inf
        }

    def detect_correlation_regimes(self,
                                 window_size: int = 252,
                                 n_regimes: int = 3) -> Dict[str, any]:
        """
        Detect correlation regime changes over time

        Args:
            window_size: Rolling window size for correlation estimation
            n_regimes: Number of correlation regimes to identify

        Returns:
            Dictionary with regime analysis results
        """
        if self.returns_data is None:
            raise ValueError("Data must be loaded first")

        # Calculate rolling correlations
        rolling_corr_series = []
        dates = []

        for i in range(window_size, len(self.returns_data)):
            window_data = self.returns_data.iloc[i-window_size:i]
            corr_matrix = window_data.corr().values

            # Extract upper triangular correlations
            upper_tri = corr_matrix[np.triu_indices(len(corr_matrix), k=1)]
            rolling_corr_series.append(upper_tri)
            dates.append(self.returns_data.index[i])

        rolling_corr_array = np.array(rolling_corr_series)

        # Fit Gaussian Mixture Model to identify regimes
        gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        regime_labels = gmm.fit_predict(rolling_corr_array)

        # Calculate regime statistics
        regime_stats = {}
        for regime in range(n_regimes):
            regime_mask = regime_labels == regime
            regime_correlations = rolling_corr_array[regime_mask]

            regime_stats[f"regime_{regime}"] = {
                "avg_correlation": np.mean(regime_correlations),
                "correlation_std": np.std(regime_correlations),
                "frequency": np.sum(regime_mask) / len(regime_labels),
                "periods": np.array(dates)[regime_mask]
            }

        return {
            "regime_labels": regime_labels,
            "regime_dates": dates,
            "regime_stats": regime_stats,
            "gmm_model": gmm,
            "n_regimes": n_regimes
        }

    def hierarchical_clustering(self, method: str = "ward") -> Dict[str, any]:
        """
        Perform hierarchical clustering on correlation matrix

        Args:
            method: Clustering linkage method

        Returns:
            Clustering results
        """
        if self.correlation_matrix is None:
            self.estimate_correlation_matrix()

        # Convert correlation to distance
        distance_matrix = np.sqrt(2 * (1 - self.correlation_matrix))

        # Perform hierarchical clustering
        condensed_distances = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_distances, method=method)

        # Extract clusters at different levels
        cluster_results = {}
        for n_clusters in [2, 3, 4, 5]:
            clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            cluster_results[f"{n_clusters}_clusters"] = clusters

        return {
            "linkage_matrix": linkage_matrix,
            "distance_matrix": distance_matrix,
            "clusters": cluster_results,
            "asset_names": self.returns_data.columns.tolist() if self.returns_data is not None else None
        }

    def factor_analysis(self, n_factors: int = None) -> Dict[str, any]:
        """
        Perform factor analysis on returns data

        Args:
            n_factors: Number of factors to extract (auto-determined if None)

        Returns:
            Factor analysis results
        """
        if self.returns_data is None:
            raise ValueError("Data must be loaded first")

        clean_data = self.returns_data.dropna()

        # Determine optimal number of factors using eigenvalue > 1 rule
        if n_factors is None:
            eigenvals, _ = np.linalg.eigh(np.corrcoef(clean_data.T))
            n_factors = np.sum(eigenvals > 1)
            n_factors = max(1, min(n_factors, clean_data.shape[1] - 1))

        # Perform factor analysis
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        factors = fa.fit_transform(clean_data)

        # Create factor loadings DataFrame
        loadings = pd.DataFrame(
            fa.components_.T,
            columns=[f"Factor_{i+1}" for i in range(n_factors)],
            index=clean_data.columns
        )

        # Calculate factor contributions to variance
        factor_variance = np.var(factors, axis=0)
        total_variance = np.sum(factor_variance)
        variance_explained = factor_variance / total_variance

        return {
            "loadings": loadings,
            "factors": factors,
            "factor_variance": factor_variance,
            "variance_explained": variance_explained,
            "cumulative_variance_explained": np.cumsum(variance_explained),
            "n_factors": n_factors,
            "log_likelihood": fa.score(clean_data)
        }

    def risk_decomposition(self, portfolio_weights: np.ndarray) -> RiskDecomposition:
        """
        Decompose portfolio risk into component contributions

        Args:
            portfolio_weights: Portfolio allocation weights

        Returns:
            RiskDecomposition results
        """
        if self.covariance_matrix is None:
            self.estimate_correlation_matrix()

        weights = np.array(portfolio_weights)
        cov_matrix = self.covariance_matrix

        # Total portfolio risk
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_risk = np.sqrt(portfolio_variance)

        # Marginal contributions to risk
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_risk

        # Component contributions to risk
        component_contrib = weights * marginal_contrib

        # Percentage contributions
        percentage_contrib = component_contrib / portfolio_risk

        # Interaction effects matrix
        interaction_effects = np.outer(weights, weights) * cov_matrix / portfolio_variance

        # Create dictionaries with asset names if available
        asset_names = (self.returns_data.columns.tolist()
                      if self.returns_data is not None
                      else [f"Asset_{i}" for i in range(len(weights))])

        component_dict = dict(zip(asset_names, component_contrib))
        marginal_dict = dict(zip(asset_names, marginal_contrib))
        percentage_dict = dict(zip(asset_names, percentage_contrib))

        return RiskDecomposition(
            total_risk=portfolio_risk,
            component_contributions=component_dict,
            marginal_contributions=marginal_dict,
            percentage_contributions=percentage_dict,
            interaction_effects=interaction_effects
        )

    def calculate_rolling_correlations(self,
                                     asset1: str,
                                     asset2: str,
                                     window: int = 252) -> pd.Series:
        """
        Calculate rolling correlation between two assets

        Args:
            asset1, asset2: Asset names
            window: Rolling window size

        Returns:
            Rolling correlation series
        """
        if self.returns_data is None:
            raise ValueError("Data must be loaded first")

        return self.returns_data[asset1].rolling(window).corr(self.returns_data[asset2])

    def correlation_breakdown_by_sector(self,
                                       sector_mapping: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """
        Break down correlations by sector

        Args:
            sector_mapping: Dictionary mapping assets to sectors

        Returns:
            Dictionary of correlation statistics by sector pair
        """
        if self.correlation_matrix is None:
            self.estimate_correlation_matrix()

        asset_names = (self.returns_data.columns.tolist()
                      if self.returns_data is not None
                      else [f"Asset_{i}" for i in range(len(self.correlation_matrix))])

        # Group assets by sector
        sector_groups = {}
        for asset in asset_names:
            sector = sector_mapping.get(asset, "Unknown")
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(asset)

        # Calculate within and between sector correlations
        sector_correlations = {}

        sectors = list(sector_groups.keys())
        for i, sector1 in enumerate(sectors):
            for j, sector2 in enumerate(sectors):
                if i <= j:  # Only calculate upper triangular
                    pair_key = f"{sector1}_{sector2}" if i < j else sector1

                    # Get indices for assets in these sectors
                    indices1 = [asset_names.index(asset) for asset in sector_groups[sector1]]
                    indices2 = [asset_names.index(asset) for asset in sector_groups[sector2]]

                    # Extract correlations
                    if i == j:  # Within sector
                        correlations = []
                        for idx1 in indices1:
                            for idx2 in indices1:
                                if idx1 < idx2:
                                    correlations.append(self.correlation_matrix[idx1, idx2])
                    else:  # Between sectors
                        correlations = []
                        for idx1 in indices1:
                            for idx2 in indices2:
                                correlations.append(self.correlation_matrix[idx1, idx2])

                    if correlations:
                        sector_correlations[pair_key] = {
                            "mean": np.mean(correlations),
                            "std": np.std(correlations),
                            "min": np.min(correlations),
                            "max": np.max(correlations),
                            "n_pairs": len(correlations)
                        }

        return sector_correlations

class DynamicCorrelationModels:
    """Dynamic correlation modeling (DCC, CCC, etc.)"""

    @staticmethod
    def estimate_dcc_garch(returns_data: pd.DataFrame,
                          max_iterations: int = 1000) -> Dict[str, any]:
        """
        Estimate Dynamic Conditional Correlation GARCH model

        Note: This is a simplified implementation. For production use,
        consider using specialized libraries like arch or statsmodels.
        """
        logger.warning("DCC-GARCH implementation is simplified. Consider using arch library for production.")

        # Clean data
        clean_data = returns_data.dropna()
        n_assets = clean_data.shape[1]
        n_obs = clean_data.shape[0]

        # Initialize parameters
        alpha = 0.05  # GARCH parameter
        beta = 0.90   # GARCH parameter

        # Estimate univariate GARCH(1,1) for each asset
        conditional_vol = np.zeros((n_obs, n_assets))

        for i in range(n_assets):
            returns = clean_data.iloc[:, i].values

            # Simple GARCH(1,1) estimation
            omega = np.var(returns) * (1 - alpha - beta)
            sigma2 = np.zeros(n_obs)
            sigma2[0] = np.var(returns)

            for t in range(1, n_obs):
                sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]

            conditional_vol[:, i] = np.sqrt(sigma2)

        # Standardized residuals
        standardized_residuals = clean_data.values / conditional_vol

        # Dynamic correlation estimation (simplified)
        # In practice, this would involve MLE estimation
        static_corr = np.corrcoef(standardized_residuals.T)

        # Simulate dynamic correlations (placeholder)
        dynamic_correlations = np.tile(static_corr, (n_obs, 1, 1))

        return {
            "conditional_volatilities": conditional_vol,
            "dynamic_correlations": dynamic_correlations,
            "standardized_residuals": standardized_residuals,
            "unconditional_correlation": static_corr,
            "parameters": {"alpha": alpha, "beta": beta}
        }

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate sample data
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    assets = ['BTC', 'ETH', 'SOL', 'AVAX']

    # Create correlated returns
    true_corr = np.array([[1.0, 0.7, 0.5, 0.3],
                         [0.7, 1.0, 0.6, 0.4],
                         [0.5, 0.6, 1.0, 0.5],
                         [0.3, 0.4, 0.5, 1.0]])

    returns_data = pd.DataFrame(
        np.random.multivariate_normal(np.zeros(4), true_corr * 0.04, 252),
        index=dates,
        columns=assets
    )

    # Initialize analyzer
    analyzer = CorrelationAnalyzer(estimation_method="ledoit_wolf")
    analyzer.load_data(returns_data)

    # Analyze correlation structure
    corr_metrics = analyzer.analyze_correlation_structure()
    print("=== CORRELATION ANALYSIS ===")
    print(f"Average correlation: {corr_metrics.concentration_metrics['avg_correlation']:.3f}")
    print(f"Effective assets: {corr_metrics.concentration_metrics['effective_assets']:.1f}")
    print(f"Diversification ratio: {corr_metrics.diversification_ratio:.3f}")

    # Risk decomposition
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    risk_decomp = analyzer.risk_decomposition(weights)
    print("\n=== RISK DECOMPOSITION ===")
    print(f"Total portfolio risk: {risk_decomp.total_risk:.4f}")
    for asset, contrib in risk_decomp.percentage_contributions.items():
        print(f"{asset}: {contrib:.1%} contribution")

    # Hierarchical clustering
    clustering = analyzer.hierarchical_clustering()
    print(f"\n=== CLUSTERING ===")
    print(f"3-cluster grouping: {clustering['clusters']['3_clusters']}")