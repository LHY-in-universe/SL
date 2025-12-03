"""
Monitoring Configuration

Centralized configuration for monitoring components
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MonitoringConfig:
    """
    Configuration for monitoring components

    Attributes:
        max_history_size: Maximum number of metrics data points to retain
        max_log_size: Maximum number of log entries to retain
        ui_refresh_interval: UI auto-refresh interval in seconds
        enable_detailed_metrics: Whether to collect detailed performance metrics
        enable_bandwidth_tracking: Whether to track bandwidth usage
        ui_port: Default port for monitoring UI
        ui_share: Whether to create public Gradio share link
        log_level: Default log level filter for UI
    """

    # Data retention
    max_history_size: int = 1000
    max_log_size: int = 1000

    # UI configuration
    ui_refresh_interval: int = 2  # seconds
    ui_port: int = 7861  # default for server
    ui_share: bool = False
    log_level: str = "INFO"

    # Feature flags
    enable_detailed_metrics: bool = True
    enable_bandwidth_tracking: bool = False  # Not implemented yet

    # Performance
    metrics_collection_overhead_threshold: float = 0.05  # 5% max overhead

    def validate(self) -> bool:
        """
        Validate configuration values

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If any configuration value is invalid
        """
        if self.max_history_size < 100:
            raise ValueError("max_history_size must be at least 100")

        if self.max_log_size < 100:
            raise ValueError("max_log_size must be at least 100")

        if self.ui_refresh_interval < 1:
            raise ValueError("ui_refresh_interval must be at least 1 second")

        if not (1024 <= self.ui_port <= 65535):
            raise ValueError("ui_port must be between 1024 and 65535")

        if self.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'ALL']:
            raise ValueError(
                "log_level must be one of: DEBUG, INFO, WARNING, ERROR, ALL"
            )

        return True


# Default configuration instance
DEFAULT_CONFIG = MonitoringConfig()
