"""
Memory resource collector using psutil
"""
import psutil
from typing import Tuple


class MemoryCollector:
    """
    Collects memory usage metrics

    Uses psutil to measure process and system memory usage.
    Returns memory in megabytes for easier interpretation.

    Example:
        >>> collector = MemoryCollector()
        >>> memory_mb, memory_percent = collector.get_process_memory()
        >>> print(f"Memory: {memory_mb:.1f} MB ({memory_percent:.1f}%)")
    """

    def __init__(self):
        """Initialize memory collector"""
        self._process = psutil.Process()

    def get_process_memory(self) -> Tuple[float, float]:
        """
        Get current process memory usage

        Returns:
            Tuple of (memory_mb, memory_percent):
            - memory_mb: Memory usage in megabytes
            - memory_percent: Percentage of system memory used
        """
        try:
            mem_info = self._process.memory_info()
            mem_percent = self._process.memory_percent()
            memory_mb = mem_info.rss / (1024 * 1024)  # Convert bytes to MB
            return memory_mb, mem_percent
        except Exception:
            return 0.0, 0.0

    def get_system_memory(self) -> Tuple[float, float, float, float]:
        """
        Get system-wide memory statistics

        Returns:
            Tuple of (total_mb, available_mb, used_mb, percent):
            - total_mb: Total system memory in MB
            - available_mb: Available system memory in MB
            - used_mb: Used system memory in MB
            - percent: Percentage of memory used
        """
        try:
            mem = psutil.virtual_memory()
            total_mb = mem.total / (1024 * 1024)
            available_mb = mem.available / (1024 * 1024)
            used_mb = mem.used / (1024 * 1024)
            percent = mem.percent
            return total_mb, available_mb, used_mb, percent
        except Exception:
            return 0.0, 0.0, 0.0, 0.0

    def get_process_memory_details(self) -> dict:
        """
        Get detailed process memory information

        Returns:
            Dictionary with detailed memory metrics:
            - rss: Resident Set Size in MB
            - vms: Virtual Memory Size in MB
            - percent: Percentage of system memory
            - peak_rss: Peak RSS (if available) in MB
        """
        try:
            mem_info = self._process.memory_info()
            result = {
                "rss_mb": mem_info.rss / (1024 * 1024),
                "vms_mb": mem_info.vms / (1024 * 1024),
                "percent": self._process.memory_percent(),
            }

            # Peak RSS is available on some platforms (Windows)
            if hasattr(mem_info, 'peak_wset'):
                result["peak_rss_mb"] = mem_info.peak_wset / (1024 * 1024)

            return result
        except Exception:
            return {
                "rss_mb": 0.0,
                "vms_mb": 0.0,
                "percent": 0.0,
            }

    def get_memory_stats(self) -> dict:
        """
        Get comprehensive memory statistics

        Returns:
            Dictionary with both process and system memory stats
        """
        process_mb, process_percent = self.get_process_memory()
        sys_total, sys_available, sys_used, sys_percent = self.get_system_memory()

        return {
            "process_memory_mb": process_mb,
            "process_memory_percent": process_percent,
            "system_total_mb": sys_total,
            "system_available_mb": sys_available,
            "system_used_mb": sys_used,
            "system_memory_percent": sys_percent,
        }
