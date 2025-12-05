"""
System Monitor - Background thread for resource monitoring
"""
import time
import threading
from collections import deque
from typing import List, Optional
import numpy as np

from ..config import ResourceSnapshot, ResourceStatistics
from ..collectors import CPUCollector, MemoryCollector, GPUCollector
from ..utils import calculate_percentile


class SystemMonitor:
    """
    Background system resource monitor

    Runs a background thread that samples CPU, memory, and GPU usage
    at regular intervals. Stores samples in a circular buffer.

    Example:
        >>> monitor = SystemMonitor(sampling_interval=0.1, enable_gpu=True)
        >>> monitor.start()
        >>> # Do some work...
        >>> time.sleep(5)
        >>> monitor.stop()
        >>> stats = monitor.get_statistics()
        >>> print(f"Average CPU: {stats.cpu_mean:.1f}%")
    """

    def __init__(
        self,
        sampling_interval: float = 0.1,
        max_samples: int = 10000,
        enable_gpu: bool = True,
        gpu_device_id: int = 0,
    ):
        """
        Initialize system monitor

        Args:
            sampling_interval: Time between samples in seconds (default: 0.1)
            max_samples: Maximum number of samples to retain (default: 10000)
            enable_gpu: Enable GPU monitoring if available (default: True)
            gpu_device_id: GPU device ID to monitor (default: 0)
        """
        self.sampling_interval = sampling_interval
        self.max_samples = max_samples
        self.enable_gpu = enable_gpu

        # Collectors
        self.cpu_collector = CPUCollector()
        self.memory_collector = MemoryCollector()
        self.gpu_collector = GPUCollector(device_id=gpu_device_id) if enable_gpu else None

        # Check GPU availability
        self.gpu_available = (
            self.gpu_collector.is_available() if self.gpu_collector else False
        )

        # Sample storage (circular buffer)
        self._samples = deque(maxlen=max_samples)
        self._lock = threading.Lock()

        # Background thread control
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # Start time
        self._start_time: Optional[float] = None

    def start(self):
        """
        Start background monitoring thread

        Raises:
            RuntimeError: If monitor is already running
        """
        if self._running:
            raise RuntimeError("SystemMonitor is already running")

        self._stop_event.clear()
        self._running = True
        self._start_time = time.time()

        # Start background thread
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """
        Stop background monitoring thread

        Blocks until the thread has stopped.
        """
        if not self._running:
            return

        self._stop_event.set()
        self._running = False

        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def is_running(self) -> bool:
        """
        Check if monitor is currently running

        Returns:
            True if monitor is running, False otherwise
        """
        return self._running

    def _monitor_loop(self):
        """Background monitoring loop (runs in separate thread)"""
        while not self._stop_event.is_set():
            try:
                # Collect current snapshot
                snapshot = self._collect_snapshot()

                # Store snapshot (thread-safe)
                with self._lock:
                    self._samples.append(snapshot)

            except Exception:
                # Silently ignore errors to keep monitoring running
                pass

            # Sleep for sampling interval
            self._stop_event.wait(self.sampling_interval)

    def _collect_snapshot(self) -> ResourceSnapshot:
        """
        Collect a single resource snapshot

        Returns:
            ResourceSnapshot with current resource usage
        """
        timestamp = time.time()

        # Collect CPU and memory
        cpu_percent = self.cpu_collector.get_cpu_percent(interval=None)
        memory_mb, memory_percent = self.memory_collector.get_process_memory()

        # Collect GPU if available
        gpu_util = None
        gpu_mem_used = None
        gpu_mem_total = None

        if self.gpu_available and self.gpu_collector:
            gpu_util, gpu_mem_used, gpu_mem_total = self.gpu_collector.get_gpu_usage()

        return ResourceSnapshot(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            gpu_available=self.gpu_available,
            gpu_utilization=gpu_util,
            gpu_memory_used_mb=gpu_mem_used,
            gpu_memory_total_mb=gpu_mem_total,
        )

    def get_current_snapshot(self) -> Optional[ResourceSnapshot]:
        """
        Get the most recent resource snapshot

        Returns:
            Most recent ResourceSnapshot, or None if no samples collected yet
        """
        with self._lock:
            return self._samples[-1] if self._samples else None

    def get_history(
        self,
        duration_seconds: Optional[float] = None
    ) -> List[ResourceSnapshot]:
        """
        Get historical snapshots

        Args:
            duration_seconds: Return samples from last N seconds.
                            If None, return all samples.

        Returns:
            List of ResourceSnapshot objects
        """
        with self._lock:
            if not self._samples:
                return []

            if duration_seconds is None:
                return list(self._samples)

            # Filter by time
            cutoff_time = time.time() - duration_seconds
            return [s for s in self._samples if s.timestamp >= cutoff_time]

    def get_statistics(self) -> ResourceStatistics:
        """
        Get statistical summary of resource usage

        Returns:
            ResourceStatistics with mean, max, P95 values
        """
        with self._lock:
            samples = list(self._samples)

        if not samples:
            return ResourceStatistics(
                duration_seconds=0.0,
                sample_count=0,
                cpu_mean=0.0,
                cpu_max=0.0,
                cpu_p95=0.0,
                memory_mean_mb=0.0,
                memory_max_mb=0.0,
                memory_p95_mb=0.0,
                gpu_available=self.gpu_available,
            )

        # Calculate duration
        duration = samples[-1].timestamp - samples[0].timestamp

        # Extract CPU values
        cpu_values = [s.cpu_percent for s in samples]
        cpu_mean = float(np.mean(cpu_values))
        cpu_max = float(np.max(cpu_values))
        cpu_p95 = calculate_percentile(cpu_values, 95)

        # Extract memory values
        memory_values = [s.memory_mb for s in samples]
        memory_mean = float(np.mean(memory_values))
        memory_max = float(np.max(memory_values))
        memory_p95 = calculate_percentile(memory_values, 95)

        # Extract GPU values if available
        gpu_mean = None
        gpu_max = None
        gpu_p95 = None
        gpu_mem_mean = None
        gpu_mem_max = None
        gpu_mem_p95 = None

        if self.gpu_available:
            gpu_values = [s.gpu_utilization for s in samples if s.gpu_utilization is not None]
            if gpu_values:
                gpu_mean = float(np.mean(gpu_values))
                gpu_max = float(np.max(gpu_values))
                gpu_p95 = calculate_percentile(gpu_values, 95)

            gpu_mem_values = [s.gpu_memory_used_mb for s in samples if s.gpu_memory_used_mb is not None]
            if gpu_mem_values:
                gpu_mem_mean = float(np.mean(gpu_mem_values))
                gpu_mem_max = float(np.max(gpu_mem_values))
                gpu_mem_p95 = calculate_percentile(gpu_mem_values, 95)

        return ResourceStatistics(
            duration_seconds=duration,
            sample_count=len(samples),
            cpu_mean=cpu_mean,
            cpu_max=cpu_max,
            cpu_p95=cpu_p95,
            memory_mean_mb=memory_mean,
            memory_max_mb=memory_max,
            memory_p95_mb=memory_p95,
            gpu_available=self.gpu_available,
            gpu_mean=gpu_mean,
            gpu_max=gpu_max,
            gpu_p95=gpu_p95,
            gpu_memory_mean_mb=gpu_mem_mean,
            gpu_memory_max_mb=gpu_mem_max,
            gpu_memory_p95_mb=gpu_mem_p95,
        )

    def clear_history(self):
        """Clear all stored samples"""
        with self._lock:
            self._samples.clear()

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
        return False

    def __del__(self):
        """Cleanup on deletion"""
        if self._running:
            self.stop()
