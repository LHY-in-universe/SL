"""
Health checking functionality.
"""

import logging
from typing import Dict, List
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthChecker:
    """
    Performs health checks on managed models and system resources.
    """

    def __init__(self, resource_manager=None, model_manager=None):
        """
        Initialize health checker.

        Args:
            resource_manager: ResourceManager instance
            model_manager: ModelManager instance
        """
        self.resource_manager = resource_manager
        self.model_manager = model_manager
        self.last_check_time = None
        self.last_status = HealthStatus.HEALTHY

    def check_health(self) -> Dict:
        """
        Perform comprehensive health check.

        Returns:
            Health check result dictionary
        """
        self.last_check_time = datetime.now()
        checks = []
        overall_status = HealthStatus.HEALTHY

        # Check system resources
        resource_check = self._check_resources()
        checks.append(resource_check)
        if resource_check["status"] != HealthStatus.HEALTHY.value:
            overall_status = HealthStatus.DEGRADED

        # Check models if manager available
        if self.model_manager:
            model_check = self._check_models()
            checks.append(model_check)
            if model_check["status"] != HealthStatus.HEALTHY.value:
                overall_status = HealthStatus.DEGRADED

        self.last_status = overall_status

        return {
            "status": overall_status.value,
            "timestamp": self.last_check_time.isoformat(),
            "checks": checks
        }

    def _check_resources(self) -> Dict:
        """Check system resource health."""
        if not self.resource_manager:
            return {
                "name": "resources",
                "status": HealthStatus.HEALTHY.value,
                "message": "Resource manager not configured"
            }

        try:
            usage = self.resource_manager.get_current_usage()

            # Determine status based on usage
            status = HealthStatus.HEALTHY

            warnings = []

            # CPU check
            if usage.cpu_percent > 90:
                status = HealthStatus.DEGRADED
                warnings.append(f"High CPU usage: {usage.cpu_percent:.1f}%")

            # Memory check
            if usage.memory_percent > 85:
                status = HealthStatus.DEGRADED
                warnings.append(f"High memory usage: {usage.memory_percent:.1f}%")
            elif usage.memory_percent > 95:
                status = HealthStatus.UNHEALTHY
                warnings.append(f"Critical memory usage: {usage.memory_percent:.1f}%")

            # GPU memory check
            if usage.gpu_memory_mb:
                for gpu_id, mem_mb in usage.gpu_memory_mb.items():
                    # Get total GPU memory
                    import torch
                    if torch.cuda.is_available():
                        total_mb = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 2)
                        usage_percent = (mem_mb / total_mb) * 100

                        if usage_percent > 90:
                            status = HealthStatus.DEGRADED
                            warnings.append(f"High GPU {gpu_id} memory: {usage_percent:.1f}%")

            message = "; ".join(warnings) if warnings else "Resources healthy"

            return {
                "name": "resources",
                "status": status.value,
                "message": message,
                "details": {
                    "cpu_percent": usage.cpu_percent,
                    "memory_percent": usage.memory_percent,
                    "memory_mb": usage.memory_mb,
                }
            }

        except Exception as e:
            logger.error(f"Resource health check failed: {e}")
            return {
                "name": "resources",
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Health check error: {e}"
            }

    def _check_models(self) -> Dict:
        """Check loaded models health."""
        try:
            models_info = self.model_manager.list_models()

            # Count models by status
            status_counts = {}
            for model_info in models_info:
                status = model_info.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1

            # Determine overall model health
            if status_counts.get("failed", 0) > 0:
                status = HealthStatus.DEGRADED
                message = f"{status_counts['failed']} failed model(s)"
            else:
                status = HealthStatus.HEALTHY
                message = f"{len(models_info)} model(s) loaded"

            return {
                "name": "models",
                "status": status.value,
                "message": message,
                "details": {
                    "total_models": len(models_info),
                    "status_counts": status_counts
                }
            }

        except Exception as e:
            logger.error(f"Model health check failed: {e}")
            return {
                "name": "models",
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Health check error: {e}"
            }

    def is_healthy(self) -> bool:
        """
        Quick health check.

        Returns:
            True if system is healthy
        """
        result = self.check_health()
        return result["status"] == HealthStatus.HEALTHY.value

    def get_last_check(self) -> Dict:
        """Get result of last health check."""
        if self.last_check_time is None:
            return self.check_health()

        return {
            "status": self.last_status.value,
            "timestamp": self.last_check_time.isoformat()
        }
