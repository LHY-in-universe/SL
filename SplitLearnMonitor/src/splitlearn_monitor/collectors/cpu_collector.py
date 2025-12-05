"""
CPU resource collector using psutil
"""
import psutil
from typing import Optional


class CPUCollector:
    """
    Collects CPU utilization metrics

    Uses psutil to measure CPU usage. Provides both instantaneous
    and interval-based measurements.

    Example:
        >>> collector = CPUCollector()
        >>> cpu_percent = collector.get_cpu_percent()
        >>> print(f"CPU: {cpu_percent}%")
    """

    def __init__(self):
        """Initialize CPU collector"""
        self._process = psutil.Process()
        # Initialize the CPU percent measurement
        self._process.cpu_percent(interval=None)

    def get_cpu_percent(self, interval: Optional[float] = None) -> float:
        """
        Get current process CPU utilization percentage

        Args:
            interval: Time interval for measurement in seconds.
                     If None, returns instantaneous value.

        Returns:
            CPU utilization percentage (0-100 per core, can exceed 100 on multi-core)
        """
        try:
            return self._process.cpu_percent(interval=interval)
        except Exception:
            return 0.0

    def get_system_cpu_percent(self, interval: Optional[float] = None) -> float:
        """
        Get system-wide CPU utilization percentage

        Args:
            interval: Time interval for measurement in seconds.
                     If None, returns instantaneous value.

        Returns:
            System CPU utilization percentage (0-100)
        """
        try:
            return psutil.cpu_percent(interval=interval)
        except Exception:
            return 0.0

    def get_cpu_count(self) -> int:
        """
        Get number of CPU cores

        Returns:
            Number of logical CPU cores
        """
        try:
            return psutil.cpu_count(logical=True) or 1
        except Exception:
            return 1

    def get_cpu_frequency(self) -> Optional[float]:
        """
        Get current CPU frequency in MHz

        Returns:
            Current CPU frequency in MHz, or None if unavailable
        """
        try:
            freq = psutil.cpu_freq()
            return freq.current if freq else None
        except Exception:
            return None

    def get_cpu_stats(self) -> dict:
        """
        Get comprehensive CPU statistics

        Returns:
            Dictionary with CPU stats including percent, count, frequency
        """
        return {
            "cpu_percent": self.get_cpu_percent(),
            "system_cpu_percent": self.get_system_cpu_percent(),
            "cpu_count": self.get_cpu_count(),
            "cpu_frequency_mhz": self.get_cpu_frequency(),
        }
