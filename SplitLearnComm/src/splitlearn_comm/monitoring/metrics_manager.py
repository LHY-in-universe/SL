"""
Metrics Manager for tracking and analyzing performance metrics

Provides centralized management for:
- Latency tracking (request/response times)
- Throughput monitoring
- Statistical analysis (P50, P95, P99, etc.)
- Historical data retention
"""

import time
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np


class MetricsManager:
    """
    Thread-safe metrics manager for tracking performance data

    Features:
    - Latency tracking with statistical analysis
    - Throughput monitoring
    - Error rate tracking
    - Historical data retention with configurable size

    Example:
        >>> manager = MetricsManager(max_history_size=1000)
        >>> manager.record_latency(0.045)  # 45ms
        >>> stats = manager.get_latency_stats()
        >>> print(f"P95: {stats['p95']:.2f}ms")
    """

    def __init__(self, max_history_size: int = 1000):
        """
        Initialize the metrics manager

        Args:
            max_history_size: Maximum number of data points to retain
        """
        self.max_history_size = max_history_size

        # Latency tracking
        self._latencies = deque(maxlen=max_history_size)
        self._timestamps = deque(maxlen=max_history_size)

        # Throughput tracking (requests per second)
        self._throughput_history = deque(maxlen=max_history_size)
        self._throughput_timestamps = deque(maxlen=max_history_size)

        # Request counting
        self._total_requests = 0
        self._last_throughput_calculation = time.time()
        self._requests_since_last_calc = 0

        # Thread safety
        self._lock = threading.Lock()

        # Start time
        self._start_time = time.time()

    def record_latency(self, latency_seconds: float):
        """
        Record a latency measurement

        Args:
            latency_seconds: Latency in seconds
        """
        with self._lock:
            current_time = time.time()
            self._latencies.append(latency_seconds * 1000)  # Convert to ms
            self._timestamps.append(current_time)
            self._total_requests += 1
            self._requests_since_last_calc += 1

            # Update throughput every second
            if current_time - self._last_throughput_calculation >= 1.0:
                throughput = self._requests_since_last_calc / (
                    current_time - self._last_throughput_calculation
                )
                self._throughput_history.append(throughput)
                self._throughput_timestamps.append(current_time)
                self._last_throughput_calculation = current_time
                self._requests_since_last_calc = 0

    def get_latency_stats(self) -> Dict:
        """
        Get comprehensive latency statistics

        Returns:
            Dictionary containing:
            - count: Number of measurements
            - mean: Average latency (ms)
            - median: Median latency (ms)
            - min: Minimum latency (ms)
            - max: Maximum latency (ms)
            - p50, p95, p99: Percentile values (ms)
            - std: Standard deviation (ms)
        """
        with self._lock:
            if not self._latencies:
                return {
                    'count': 0,
                    'mean': 0,
                    'median': 0,
                    'min': 0,
                    'max': 0,
                    'p50': 0,
                    'p95': 0,
                    'p99': 0,
                    'std': 0
                }

            latencies_array = np.array(list(self._latencies))

            return {
                'count': len(latencies_array),
                'mean': float(np.mean(latencies_array)),
                'median': float(np.median(latencies_array)),
                'min': float(np.min(latencies_array)),
                'max': float(np.max(latencies_array)),
                'p50': float(np.percentile(latencies_array, 50)),
                'p95': float(np.percentile(latencies_array, 95)),
                'p99': float(np.percentile(latencies_array, 99)),
                'std': float(np.std(latencies_array))
            }

    def get_latency_distribution(self, num_bins: int = 20) -> Tuple[List[float], List[int]]:
        """
        Get latency distribution for histogram plotting

        Args:
            num_bins: Number of histogram bins

        Returns:
            Tuple of (bin_edges, counts)
        """
        with self._lock:
            if not self._latencies:
                return ([], [])

            latencies_array = np.array(list(self._latencies))
            counts, bin_edges = np.histogram(latencies_array, bins=num_bins)

            return (bin_edges.tolist(), counts.tolist())

    def get_latency_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get latency history with timestamps

        Args:
            limit: Maximum number of recent measurements to return

        Returns:
            List of dictionaries with 'timestamp' and 'latency_ms' keys
        """
        with self._lock:
            if not self._latencies:
                return []

            latencies = list(self._latencies)
            timestamps = list(self._timestamps)

            if limit:
                latencies = latencies[-limit:]
                timestamps = timestamps[-limit:]

            return [
                {
                    'timestamp': datetime.fromtimestamp(ts),
                    'latency_ms': lat
                }
                for ts, lat in zip(timestamps, latencies)
            ]

    def get_throughput_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get throughput history (requests per second)

        Args:
            limit: Maximum number of recent measurements to return

        Returns:
            List of dictionaries with 'timestamp' and 'rps' (requests per second) keys
        """
        with self._lock:
            if not self._throughput_history:
                return []

            throughput = list(self._throughput_history)
            timestamps = list(self._throughput_timestamps)

            if limit:
                throughput = throughput[-limit:]
                timestamps = timestamps[-limit:]

            return [
                {
                    'timestamp': datetime.fromtimestamp(ts),
                    'rps': rps
                }
                for ts, rps in zip(timestamps, throughput)
            ]

    def get_current_throughput(self) -> float:
        """
        Get current throughput (requests per second)

        Returns:
            Current RPS, or 0 if no data
        """
        with self._lock:
            if not self._throughput_history:
                return 0.0
            return self._throughput_history[-1]

    def get_total_requests(self) -> int:
        """Get total number of requests recorded"""
        with self._lock:
            return self._total_requests

    def get_uptime_seconds(self) -> float:
        """Get uptime in seconds since manager initialization"""
        return time.time() - self._start_time

    def reset(self):
        """Reset all metrics"""
        with self._lock:
            self._latencies.clear()
            self._timestamps.clear()
            self._throughput_history.clear()
            self._throughput_timestamps.clear()
            self._total_requests = 0
            self._requests_since_last_calc = 0
            self._start_time = time.time()
            self._last_throughput_calculation = time.time()
