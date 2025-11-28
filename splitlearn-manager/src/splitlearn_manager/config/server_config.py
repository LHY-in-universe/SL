"""
Server configuration management.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import yaml


@dataclass
class ServerConfig:
    """
    Configuration for managed server.

    Attributes:
        host: Server host address
        port: Server port
        max_workers: Maximum number of worker threads
        max_models: Maximum number of models to load simultaneously
        metrics_port: Port for Prometheus metrics
        health_check_interval: Health check interval in seconds
        enable_monitoring: Whether to enable Prometheus monitoring
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        config: Additional server configuration
    """

    host: str = "0.0.0.0"
    port: int = 50051
    max_workers: int = 10
    max_models: int = 5
    metrics_port: int = 8000
    health_check_interval: float = 30.0
    enable_monitoring: bool = True
    log_level: str = "INFO"
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ServerConfig":
        """
        Load server configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            ServerConfig instance
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, yaml_path: str) -> None:
        """
        Save server configuration to YAML file.

        Args:
            yaml_path: Path to save YAML configuration
        """
        data = {
            "host": self.host,
            "port": self.port,
            "max_workers": self.max_workers,
            "max_models": self.max_models,
            "metrics_port": self.metrics_port,
            "health_check_interval": self.health_check_interval,
            "enable_monitoring": self.enable_monitoring,
            "log_level": self.log_level,
            "config": self.config,
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "max_workers": self.max_workers,
            "max_models": self.max_models,
            "metrics_port": self.metrics_port,
            "health_check_interval": self.health_check_interval,
            "enable_monitoring": self.enable_monitoring,
            "log_level": self.log_level,
            "config": self.config,
        }

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Invalid port: {self.port}")

        if self.metrics_port < 1 or self.metrics_port > 65535:
            raise ValueError(f"Invalid metrics_port: {self.metrics_port}")

        if self.max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got {self.max_workers}")

        if self.max_models <= 0:
            raise ValueError(f"max_models must be positive, got {self.max_models}")

        if self.health_check_interval <= 0:
            raise ValueError(
                f"health_check_interval must be positive, got {self.health_check_interval}"
            )

        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"Invalid log_level: {self.log_level}")

        return True
