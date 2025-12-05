"""
Utility functions for SplitLearnMonitor
"""
import time
from typing import List, Optional
import numpy as np


def calculate_percentile(values: List[float], percentile: float) -> float:
    """
    Calculate the given percentile of a list of values

    Args:
        values: List of numeric values
        percentile: Percentile to calculate (0-100)

    Returns:
        The percentile value
    """
    if not values:
        return 0.0
    return float(np.percentile(values, percentile))


def calculate_statistics(values: List[float]) -> dict:
    """
    Calculate comprehensive statistics for a list of values

    Args:
        values: List of numeric values

    Returns:
        Dictionary containing mean, median, std, min, max, p50, p95, p99
    """
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "count": 0,
        }

    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "count": len(values),
    }


def format_bytes(bytes_value: float, precision: int = 2) -> str:
    """
    Format bytes into human-readable string

    Args:
        bytes_value: Number of bytes
        precision: Number of decimal places

    Returns:
        Formatted string (e.g., "1.50 GB")
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    value = float(bytes_value)

    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1

    return f"{value:.{precision}f} {units[unit_index]}"


def format_duration(seconds: float, precision: int = 2) -> str:
    """
    Format duration into human-readable string

    Args:
        seconds: Duration in seconds
        precision: Number of decimal places

    Returns:
        Formatted string (e.g., "1h 23m 45.67s")
    """
    if seconds < 60:
        return f"{seconds:.{precision}f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.{precision}f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {minutes}m {remaining_seconds:.{precision}f}s"


def get_timestamp_ms() -> float:
    """
    Get current timestamp in milliseconds

    Returns:
        Current time in milliseconds since epoch
    """
    return time.time() * 1000


def get_timestamp() -> float:
    """
    Get current timestamp in seconds

    Returns:
        Current time in seconds since epoch
    """
    return time.time()


class TimerContext:
    """
    Context manager for timing code blocks

    Example:
        >>> with TimerContext() as timer:
        ...     # do some work
        ...     pass
        >>> print(f"Elapsed: {timer.elapsed_ms}ms")
    """

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if self.start_time is not None:
            self.elapsed_ms = (self.end_time - self.start_time) * 1000
        return False


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Value to return if denominator is zero

    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator
