# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Risk monitoring and alerting system for real-time portfolio risk management.
Implements threshold-based alerts, regime change detection, and automated notifications.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertType(Enum):
    """Types of risk alerts"""
    VAR_BREACH = "var_breach"
    CONCENTRATION_RISK = "concentration_risk"
    CORRELATION_SPIKE = "correlation_spike"
    VOLATILITY_REGIME = "volatility_regime"
    DRAWDOWN_LIMIT = "drawdown_limit"
    LIQUIDITY_RISK = "liquidity_risk"
    SECTOR_EXPOSURE = "sector_exposure"
    LEVERAGE_LIMIT = "leverage_limit"

@dataclass
class RiskAlert:
    """Risk alert data structure"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    portfolio_id: str
    timestamp: datetime
    message: str
    current_value: float
    threshold_value: float
    metadata: Dict = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class RiskThreshold:
    """Risk threshold configuration"""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    lookback_days: int = 1
    comparison_type: str = "absolute"  # absolute, percentage, percentile
    enabled: bool = True

@dataclass
class MonitoringConfig:
    """Risk monitoring configuration"""
    update_frequency: int = 300  # seconds
    alert_cooldown: int = 3600  # seconds between same alert types
    max_alerts_per_hour: int = 10
    enable_email_alerts: bool = False
    enable_webhook_alerts: bool = True
    webhook_url: Optional[str] = None
    thresholds: Dict[str, RiskThreshold] = field(default_factory=dict)

class AlertChannel(ABC):
    """Abstract base class for alert channels"""

    @abstractmethod
    async def send_alert(self, alert: RiskAlert) -> bool:
        """Send alert through this channel"""
        pass

class WebhookAlertChannel(AlertChannel):
    """Webhook-based alert channel"""

    def __init__(self, webhook_url: str, timeout: int = 30):
        self.webhook_url = webhook_url
        self.timeout = timeout

    async def send_alert(self, alert: RiskAlert) -> bool:
        """Send alert via webhook"""
        try:
            import aiohttp

            payload = {
                "alert_id": alert.alert_id,
                "type": alert.alert_type.value,
                "severity": alert.severity.value,
                "portfolio_id": alert.portfolio_id,
                "timestamp": alert.timestamp.isoformat(),
                "message": alert.message,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "metadata": alert.metadata
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Alert {alert.alert_id} sent successfully via webhook")
                        return True
                    else:
                        logger.error(f"Webhook alert failed with status {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")
            return False

class ConsoleAlertChannel(AlertChannel):
    """Console-based alert channel for testing"""

    async def send_alert(self, alert: RiskAlert) -> bool:
        """Send alert to console"""
        severity_icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
        icon = severity_icon.get(alert.severity.value, "")

        print(f"\n{icon} RISK ALERT [{alert.severity.value.upper()}] {icon}")
        print(f"Portfolio: {alert.portfolio_id}")
        print(f"Type: {alert.alert_type.value}")
        print(f"Time: {alert.timestamp}")
        print(f"Message: {alert.message}")
        print(f"Current: {alert.current_value:.4f}, Threshold: {alert.threshold_value:.4f}")
        print("-" * 50)

        return True

class RiskMonitor:
    """Real-time risk monitoring system"""

    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        self.alert_channels: List[AlertChannel] = []
        self.active_alerts: Dict[str, RiskAlert] = {}
        self.alert_history: List[RiskAlert] = []
        self.last_alert_times: Dict[str, datetime] = {}
        self.monitoring_active = False

        # Default risk thresholds
        self._setup_default_thresholds()

    def _setup_default_thresholds(self):
        """Setup default risk thresholds"""
        default_thresholds = {
            "portfolio_var_95": RiskThreshold("VaR 95%", 0.05, 0.10, 1, "absolute"),
            "portfolio_cvar_95": RiskThreshold("CVaR 95%", 0.08, 0.15, 1, "absolute"),
            "max_drawdown": RiskThreshold("Max Drawdown", 0.10, 0.20, 1, "absolute"),
            "concentration_hhi": RiskThreshold("Concentration (HHI)", 0.30, 0.50, 1, "absolute"),
            "avg_correlation": RiskThreshold("Average Correlation", 0.70, 0.85, 5, "absolute"),
            "portfolio_volatility": RiskThreshold("Portfolio Volatility", 0.30, 0.50, 5, "absolute"),
            "leverage_ratio": RiskThreshold("Leverage Ratio", 2.0, 3.0, 1, "absolute"),
        }

        for name, threshold in default_thresholds.items():
            if name not in self.config.thresholds:
                self.config.thresholds[name] = threshold

    def add_alert_channel(self, channel: AlertChannel):
        """Add an alert channel"""
        self.alert_channels.append(channel)
        logger.info(f"Added alert channel: {type(channel).__name__}")

    def update_threshold(self, metric_name: str, threshold: RiskThreshold):
        """Update a risk threshold"""
        self.config.thresholds[metric_name] = threshold
        logger.info(f"Updated threshold for {metric_name}")

    async def check_risk_metrics(self,
                                portfolio_id: str,
                                risk_metrics: Dict[str, float]) -> List[RiskAlert]:
        """
        Check risk metrics against thresholds and generate alerts

        Args:
            portfolio_id: Portfolio identifier
            risk_metrics: Dictionary of current risk metrics

        Returns:
            List of new alerts generated
        """
        new_alerts = []
        current_time = datetime.now()

        for metric_name, current_value in risk_metrics.items():
            if metric_name not in self.config.thresholds:
                continue

            threshold = self.config.thresholds[metric_name]
            if not threshold.enabled:
                continue

            # Check for threshold breaches
            alert_type = self._get_alert_type_for_metric(metric_name)
            severity = None

            if current_value >= threshold.critical_threshold:
                severity = AlertSeverity.CRITICAL
            elif current_value >= threshold.warning_threshold:
                severity = AlertSeverity.WARNING

            if severity:
                # Check cooldown period
                alert_key = f"{portfolio_id}_{alert_type.value}_{severity.value}"

                if alert_key in self.last_alert_times:
                    time_since_last = (current_time - self.last_alert_times[alert_key]).total_seconds()
                    if time_since_last < self.config.alert_cooldown:
                        continue  # Skip due to cooldown

                # Create alert
                alert = RiskAlert(
                    alert_id=f"{alert_key}_{int(current_time.timestamp())}",
                    alert_type=alert_type,
                    severity=severity,
                    portfolio_id=portfolio_id,
                    timestamp=current_time,
                    message=self._generate_alert_message(metric_name, current_value, threshold, severity),
                    current_value=current_value,
                    threshold_value=threshold.warning_threshold if severity == AlertSeverity.WARNING else threshold.critical_threshold,
                    metadata={
                        "metric_name": metric_name,
                        "lookback_days": threshold.lookback_days,
                        "comparison_type": threshold.comparison_type
                    }
                )

                new_alerts.append(alert)
                self.active_alerts[alert.alert_id] = alert
                self.last_alert_times[alert_key] = current_time

        # Send alerts
        for alert in new_alerts:
            await self._send_alert(alert)

        return new_alerts

    def _get_alert_type_for_metric(self, metric_name: str) -> AlertType:
        """Map metric names to alert types"""
        mapping = {
            "portfolio_var_95": AlertType.VAR_BREACH,
            "portfolio_cvar_95": AlertType.VAR_BREACH,
            "max_drawdown": AlertType.DRAWDOWN_LIMIT,
            "concentration_hhi": AlertType.CONCENTRATION_RISK,
            "avg_correlation": AlertType.CORRELATION_SPIKE,
            "portfolio_volatility": AlertType.VOLATILITY_REGIME,
            "leverage_ratio": AlertType.LEVERAGE_LIMIT,
        }

        return mapping.get(metric_name, AlertType.VAR_BREACH)

    def _generate_alert_message(self,
                              metric_name: str,
                              current_value: float,
                              threshold: RiskThreshold,
                              severity: AlertSeverity) -> str:
        """Generate human-readable alert message"""
        severity_text = "WARNING" if severity == AlertSeverity.WARNING else "CRITICAL"
        threshold_val = threshold.warning_threshold if severity == AlertSeverity.WARNING else threshold.critical_threshold

        return (f"{severity_text}: {metric_name} is {current_value:.4f}, "
                f"exceeding {severity_text.lower()} threshold of {threshold_val:.4f}")

    async def _send_alert(self, alert: RiskAlert):
        """Send alert through all configured channels"""
        self.alert_history.append(alert)

        for channel in self.alert_channels:
            try:
                success = await channel.send_alert(alert)
                if not success:
                    logger.error(f"Failed to send alert {alert.alert_id} via {type(channel).__name__}")
            except Exception as e:
                logger.error(f"Error sending alert via {type(channel).__name__}: {e}")

    async def start_monitoring(self):
        """Start the risk monitoring loop"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        logger.info("Starting risk monitoring")

        # This would typically integrate with your portfolio data source
        while self.monitoring_active:
            try:
                await asyncio.sleep(self.config.update_frequency)
                # In practice, you would fetch current portfolio data here
                # and call check_risk_metrics()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    def stop_monitoring(self):
        """Stop the risk monitoring"""
        self.monitoring_active = False
        logger.info("Stopping risk monitoring")

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an active alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert {alert_id} acknowledged")
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            del self.active_alerts[alert_id]
            logger.info(f"Alert {alert_id} resolved")
            return True
        return False

    def get_active_alerts(self, portfolio_id: Optional[str] = None) -> List[RiskAlert]:
        """Get list of active alerts"""
        alerts = list(self.active_alerts.values())

        if portfolio_id:
            alerts = [alert for alert in alerts if alert.portfolio_id == portfolio_id]

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)

    def get_alert_history(self,
                         portfolio_id: Optional[str] = None,
                         hours_back: int = 24) -> List[RiskAlert]:
        """Get alert history"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        alerts = [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]

        if portfolio_id:
            alerts = [alert for alert in alerts if alert.portfolio_id == portfolio_id]

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)

    def get_monitoring_stats(self) -> Dict[str, any]:
        """Get monitoring system statistics"""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)

        recent_alerts = [alert for alert in self.alert_history if alert.timestamp >= last_24h]

        severity_counts = {}
        type_counts = {}

        for alert in recent_alerts:
            severity = alert.severity.value
            alert_type = alert.alert_type.value

            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1

        return {
            "monitoring_active": self.monitoring_active,
            "active_alerts_count": len(self.active_alerts),
            "total_alerts_24h": len(recent_alerts),
            "alerts_by_severity": severity_counts,
            "alerts_by_type": type_counts,
            "configured_thresholds": len(self.config.thresholds),
            "alert_channels": len(self.alert_channels),
            "last_update": now.isoformat()
        }

class RegimeDetector:
    """Detects regime changes in market conditions"""

    def __init__(self, lookback_window: int = 365):
        self.lookback_window = lookback_window

    def detect_volatility_regime_change(self,
                                      returns: pd.Series,
                                      threshold_multiplier: float = 2.0) -> Dict[str, any]:
        """
        Detect changes in volatility regime

        Args:
            returns: Time series of returns
            threshold_multiplier: Threshold for regime change detection

        Returns:
            Dictionary with regime change information
        """
        if len(returns) < self.lookback_window:
            return {"regime_change": False, "reason": "Insufficient data"}

        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=21).std() * np.sqrt(365)  # Annualized

        # Historical volatility statistics
        hist_vol_mean = rolling_vol.mean()
        hist_vol_std = rolling_vol.std()

        # Current volatility
        current_vol = rolling_vol.iloc[-1]

        # Z-score
        z_score = (current_vol - hist_vol_mean) / hist_vol_std

        regime_change = abs(z_score) > threshold_multiplier

        return {
            "regime_change": regime_change,
            "current_volatility": current_vol,
            "historical_mean": hist_vol_mean,
            "z_score": z_score,
            "regime_type": "high" if z_score > threshold_multiplier else "low" if z_score < -threshold_multiplier else "normal"
        }

    def detect_correlation_regime_change(self,
                                       returns_data: pd.DataFrame,
                                       threshold: float = 0.15) -> Dict[str, any]:
        """
        Detect changes in correlation regime

        Args:
            returns_data: DataFrame of asset returns
            threshold: Threshold for significant correlation change

        Returns:
            Dictionary with correlation regime change information
        """
        if len(returns_data) < self.lookback_window:
            return {"regime_change": False, "reason": "Insufficient data"}

        # Rolling correlation (average pairwise)
        rolling_corr = []
        window = 63  # Quarterly

        for i in range(window, len(returns_data)):
            window_data = returns_data.iloc[i-window:i]
            corr_matrix = window_data.corr()

            # Average correlation (upper triangular)
            n = len(corr_matrix)
            upper_tri = corr_matrix.values[np.triu_indices(n, k=1)]
            avg_corr = np.mean(upper_tri)
            rolling_corr.append(avg_corr)

        if len(rolling_corr) < 2:
            return {"regime_change": False, "reason": "Insufficient correlation history"}

        # Check for regime change
        recent_corr = np.mean(rolling_corr[-5:])  # Last week average
        historical_corr = np.mean(rolling_corr[:-5])

        corr_change = abs(recent_corr - historical_corr)
        regime_change = corr_change > threshold

        return {
            "regime_change": regime_change,
            "correlation_change": corr_change,
            "recent_correlation": recent_corr,
            "historical_correlation": historical_corr,
            "regime_type": "high" if recent_corr > historical_corr + threshold else "low" if recent_corr < historical_corr - threshold else "normal"
        }

async def create_monitoring_system(webhook_url: Optional[str] = None) -> RiskMonitor:
    """
    Factory function to create a configured risk monitoring system

    Args:
        webhook_url: Optional webhook URL for alerts

    Returns:
        Configured RiskMonitor instance
    """
    config = MonitoringConfig(
        update_frequency=300,  # 5 minutes
        alert_cooldown=1800,   # 30 minutes
        webhook_url=webhook_url
    )

    monitor = RiskMonitor(config)

    # Add default alert channels
    monitor.add_alert_channel(ConsoleAlertChannel())

    if webhook_url:
        monitor.add_alert_channel(WebhookAlertChannel(webhook_url))

    logger.info("Risk monitoring system created")
    return monitor

if __name__ == "__main__":
    # Example usage
    import asyncio

    async def main():
        # Create monitoring system
        monitor = await create_monitoring_system()

        # Example risk metrics
        test_metrics = {
            "portfolio_var_95": 0.12,  # Exceeds critical threshold (0.10)
            "concentration_hhi": 0.35,  # Exceeds warning threshold (0.30)
            "avg_correlation": 0.65,    # Within normal range
        }

        # Check metrics and generate alerts
        alerts = await monitor.check_risk_metrics("test_portfolio", test_metrics)

        print(f"\nGenerated {len(alerts)} alerts:")
        for alert in alerts:
            print(f"- {alert.alert_type.value}: {alert.message}")

        # Get monitoring stats
        stats = monitor.get_monitoring_stats()
        print(f"\nMonitoring Stats:")
        print(f"Active alerts: {stats['active_alerts_count']}")
        print(f"Configured thresholds: {stats['configured_thresholds']}")

    asyncio.run(main())