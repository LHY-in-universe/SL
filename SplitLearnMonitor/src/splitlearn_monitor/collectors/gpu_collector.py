"""
GPU resource collector using pynvml

Gracefully handles systems without GPU or without pynvml installed.
"""
from typing import Optional, Tuple, Dict

# Try to import pynvml
GPU_AVAILABLE = False
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except (ImportError, Exception):
    # pynvml not installed or NVIDIA driver not available
    GPU_AVAILABLE = False


class GPUCollector:
    """
    Collects GPU utilization and memory metrics

    Uses pynvml (NVIDIA Management Library) to access GPU metrics.
    Gracefully degrades if GPU or pynvml is not available.

    Example:
        >>> collector = GPUCollector(device_id=0)
        >>> if collector.is_available():
        ...     util, mem_used, mem_total = collector.get_gpu_usage()
        ...     print(f"GPU: {util}%, Memory: {mem_used}/{mem_total} MB")
    """

    def __init__(self, device_id: int = 0):
        """
        Initialize GPU collector

        Args:
            device_id: GPU device ID to monitor (default: 0)
        """
        self.device_id = device_id
        self._handle = None
        self._available = GPU_AVAILABLE

        if self._available:
            try:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            except Exception:
                self._available = False

    def is_available(self) -> bool:
        """
        Check if GPU monitoring is available

        Returns:
            True if GPU monitoring is available, False otherwise
        """
        return self._available

    def get_gpu_usage(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Get GPU utilization and memory usage

        Returns:
            Tuple of (utilization_percent, memory_used_mb, memory_total_mb):
            - utilization_percent: GPU utilization (0-100)
            - memory_used_mb: GPU memory used in MB
            - memory_total_mb: Total GPU memory in MB
            Returns (None, None, None) if GPU is not available
        """
        if not self._available or self._handle is None:
            return None, None, None

        try:
            # Get GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            utilization_percent = float(util.gpu)

            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            memory_used_mb = mem_info.used / (1024 * 1024)
            memory_total_mb = mem_info.total / (1024 * 1024)

            return utilization_percent, memory_used_mb, memory_total_mb
        except Exception:
            return None, None, None

    def get_gpu_temperature(self) -> Optional[float]:
        """
        Get GPU temperature in Celsius

        Returns:
            GPU temperature in Celsius, or None if unavailable
        """
        if not self._available or self._handle is None:
            return None

        try:
            temp = pynvml.nvmlDeviceGetTemperature(
                self._handle,
                pynvml.NVML_TEMPERATURE_GPU
            )
            return float(temp)
        except Exception:
            return None

    def get_gpu_power_usage(self) -> Optional[float]:
        """
        Get GPU power usage in watts

        Returns:
            GPU power usage in watts, or None if unavailable
        """
        if not self._available or self._handle is None:
            return None

        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            power_w = power_mw / 1000.0
            return power_w
        except Exception:
            return None

    def get_gpu_name(self) -> Optional[str]:
        """
        Get GPU device name

        Returns:
            GPU device name, or None if unavailable
        """
        if not self._available or self._handle is None:
            return None

        try:
            name = pynvml.nvmlDeviceGetName(self._handle)
            # pynvml returns bytes in some versions
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            return name
        except Exception:
            return None

    def get_gpu_stats(self) -> Dict:
        """
        Get comprehensive GPU statistics

        Returns:
            Dictionary with all available GPU metrics
        """
        if not self._available:
            return {
                "available": False,
                "utilization_percent": None,
                "memory_used_mb": None,
                "memory_total_mb": None,
                "temperature_celsius": None,
                "power_usage_watts": None,
                "device_name": None,
            }

        util, mem_used, mem_total = self.get_gpu_usage()
        temp = self.get_gpu_temperature()
        power = self.get_gpu_power_usage()
        name = self.get_gpu_name()

        return {
            "available": True,
            "utilization_percent": util,
            "memory_used_mb": mem_used,
            "memory_total_mb": mem_total,
            "memory_percent": (mem_used / mem_total * 100) if (mem_used and mem_total) else None,
            "temperature_celsius": temp,
            "power_usage_watts": power,
            "device_name": name,
        }

    def __del__(self):
        """Cleanup on deletion"""
        # nvmlShutdown is global and should only be called once
        # We don't call it here to avoid issues with multiple collectors
        pass


def get_gpu_count() -> int:
    """
    Get the number of available GPUs

    Returns:
        Number of GPUs, or 0 if none available
    """
    if not GPU_AVAILABLE:
        return 0

    try:
        return pynvml.nvmlDeviceGetCount()
    except Exception:
        return 0
