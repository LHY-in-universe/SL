"""
Performance Tracker - Track timing for different phases
"""
import time
import threading
from contextlib import contextmanager
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np

from ..config import PhaseStatistics
from ..utils import calculate_statistics, safe_divide


class PerformanceTracker:
    """
    Track performance timing for different phases

    Supports context manager for automatic timing and nested phase tracking.
    Computes statistical analysis including P50, P95, P99.

    Example:
        >>> tracker = PerformanceTracker()
        >>> with tracker.track_phase("bottom_model"):
        ...     # model inference
        ...     time.sleep(0.1)
        >>> stats = tracker.get_phase_statistics("bottom_model")
        >>> print(f"Average: {stats.mean_ms:.1f}ms")
    """

    def __init__(self):
        """Initialize performance tracker"""
        # Storage for phase timings (phase_name -> list of durations in ms)
        self._phase_timings: Dict[str, List[float]] = defaultdict(list)

        # Storage for phase metadata
        self._phase_metadata: Dict[str, List[Dict]] = defaultdict(list)

        # Thread safety
        self._lock = threading.Lock()

        # Track total time across all phases
        self._total_time_ms = 0.0

    @contextmanager
    def track_phase(self, phase_name: str, metadata: Optional[Dict] = None):
        """
        Context manager for tracking a phase

        Args:
            phase_name: Name of the phase (e.g., "bottom_model", "trunk_remote")
            metadata: Optional metadata to associate with this measurement

        Example:
            >>> with tracker.track_phase("computation"):
            ...     result = expensive_computation()
        """
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            self.record_phase(phase_name, duration_ms, metadata)

    def record_phase(
        self,
        phase_name: str,
        duration_ms: float,
        metadata: Optional[Dict] = None
    ):
        """
        Manually record a phase timing

        Args:
            phase_name: Name of the phase
            duration_ms: Duration in milliseconds
            metadata: Optional metadata to associate with this measurement
        """
        with self._lock:
            self._phase_timings[phase_name].append(duration_ms)
            if metadata:
                self._phase_metadata[phase_name].append(metadata)
            self._total_time_ms += duration_ms

    def get_phase_statistics(self, phase_name: str) -> Optional[PhaseStatistics]:
        """
        Get statistics for a specific phase

        Args:
            phase_name: Name of the phase

        Returns:
            PhaseStatistics for the phase, or None if phase not found
        """
        with self._lock:
            if phase_name not in self._phase_timings:
                return None

            timings = self._phase_timings[phase_name]
            if not timings:
                return None

            # Calculate statistics
            stats = calculate_statistics(timings)
            total = sum(timings)

            # Calculate percentage of total time
            percentage = safe_divide(total, self._total_time_ms, 0.0) * 100

            return PhaseStatistics(
                phase_name=phase_name,
                count=stats["count"],
                total_time_ms=total,
                mean_ms=stats["mean"],
                median_ms=stats["median"],
                min_ms=stats["min"],
                max_ms=stats["max"],
                p50_ms=stats["p50"],
                p95_ms=stats["p95"],
                p99_ms=stats["p99"],
                std_ms=stats["std"],
                percentage_of_total=percentage,
            )

    def get_all_statistics(self) -> Dict[str, PhaseStatistics]:
        """
        Get statistics for all phases

        Returns:
            Dictionary mapping phase names to PhaseStatistics
        """
        with self._lock:
            phase_names = list(self._phase_timings.keys())

        result = {}
        for phase_name in phase_names:
            stats = self.get_phase_statistics(phase_name)
            if stats:
                result[phase_name] = stats

        return result

    def get_phase_breakdown(self) -> Dict[str, float]:
        """
        Get percentage breakdown of time spent in each phase

        Returns:
            Dictionary mapping phase names to percentage of total time
        """
        with self._lock:
            if self._total_time_ms == 0:
                return {}

            breakdown = {}
            for phase_name, timings in self._phase_timings.items():
                total = sum(timings)
                percentage = (total / self._total_time_ms) * 100
                breakdown[phase_name] = percentage

            return breakdown

    def get_phase_timings(self, phase_name: str) -> List[float]:
        """
        Get raw timing measurements for a phase

        Args:
            phase_name: Name of the phase

        Returns:
            List of timing measurements in milliseconds
        """
        with self._lock:
            return list(self._phase_timings.get(phase_name, []))

    def get_total_time(self) -> float:
        """
        Get total time across all phases

        Returns:
            Total time in milliseconds
        """
        with self._lock:
            return self._total_time_ms

    def get_phase_count(self, phase_name: str) -> int:
        """
        Get number of measurements for a phase

        Args:
            phase_name: Name of the phase

        Returns:
            Number of measurements
        """
        with self._lock:
            return len(self._phase_timings.get(phase_name, []))

    def get_phase_names(self) -> List[str]:
        """
        Get names of all tracked phases

        Returns:
            List of phase names
        """
        with self._lock:
            return list(self._phase_timings.keys())

    def clear(self):
        """Clear all recorded phase timings"""
        with self._lock:
            self._phase_timings.clear()
            self._phase_metadata.clear()
            self._total_time_ms = 0.0

    def clear_phase(self, phase_name: str):
        """
        Clear recorded timings for a specific phase

        Args:
            phase_name: Name of the phase to clear
        """
        with self._lock:
            if phase_name in self._phase_timings:
                # Subtract from total time
                self._total_time_ms -= sum(self._phase_timings[phase_name])
                # Remove data
                del self._phase_timings[phase_name]
                if phase_name in self._phase_metadata:
                    del self._phase_metadata[phase_name]

    def to_dict(self) -> Dict:
        """
        Export all data to dictionary

        Returns:
            Dictionary with all phase statistics and raw data
        """
        with self._lock:
            result = {
                "total_time_ms": self._total_time_ms,
                "phases": {},
            }

            for phase_name in self._phase_timings.keys():
                stats = self.get_phase_statistics(phase_name)
                if stats:
                    result["phases"][phase_name] = {
                        "statistics": stats.to_dict(),
                        "raw_timings": list(self._phase_timings[phase_name]),
                    }

            return result

    def print_summary(self):
        """Print a summary of all phase statistics"""
        stats = self.get_all_statistics()
        if not stats:
            print("No phase data recorded")
            return

        print("\n" + "=" * 80)
        print("Performance Summary")
        print("=" * 80)
        print(f"Total Time: {self._total_time_ms:.2f} ms\n")

        # Sort by total time (descending)
        sorted_stats = sorted(
            stats.items(),
            key=lambda x: x[1].total_time_ms,
            reverse=True
        )

        for phase_name, phase_stats in sorted_stats:
            print(f"\n{phase_name}:")
            print(f"  Count: {phase_stats.count}")
            print(f"  Total: {phase_stats.total_time_ms:.2f} ms ({phase_stats.percentage_of_total:.1f}%)")
            print(f"  Mean:  {phase_stats.mean_ms:.2f} ms")
            print(f"  Min:   {phase_stats.min_ms:.2f} ms")
            print(f"  Max:   {phase_stats.max_ms:.2f} ms")
            print(f"  P50:   {phase_stats.p50_ms:.2f} ms")
            print(f"  P95:   {phase_stats.p95_ms:.2f} ms")
            print(f"  P99:   {phase_stats.p99_ms:.2f} ms")
            print(f"  Std:   {phase_stats.std_ms:.2f} ms")

        print("\n" + "=" * 80)
