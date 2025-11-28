"""
Memory usage tracking utilities.

This module provides utilities for monitoring RAM and GPU memory usage
during model loading and inference.
"""

import psutil
import torch
from typing import List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MemorySnapshot:
    """
    A snapshot of memory usage at a specific point in time.

    Attributes:
        timestamp: When the snapshot was taken
        label: Description of this snapshot
        ram_used_gb: RAM usage in GB
        ram_percent: RAM usage as percentage of total
        gpu_allocated_gb: GPU memory allocated in GB (0 if no GPU)
        gpu_reserved_gb: GPU memory reserved in GB (0 if no GPU)
    """
    timestamp: datetime
    label: str
    ram_used_gb: float
    ram_percent: float
    gpu_allocated_gb: float = 0.0
    gpu_reserved_gb: float = 0.0


class MemoryTracker:
    """
    Track memory usage over time.

    This class helps monitor RAM and GPU memory during model operations,
    making it easy to identify memory bottlenecks and verify optimizations.

    Example:
        >>> tracker = MemoryTracker()
        >>> tracker.snapshot("Before loading")
        >>> # ... load model ...
        >>> tracker.snapshot("After loading")
        >>> tracker.report()
        Memory: After loading
          RAM: 12.34 GB (+8.50 GB)
          GPU: 10.20 GB (+10.20 GB)
        >>> tracker.summary()
        Peak RAM: 12.34 GB
        Peak GPU: 10.20 GB
    """

    def __init__(self):
        """Initialize memory tracker."""
        self.snapshots: List[MemorySnapshot] = []
        self.process = psutil.Process()

    def snapshot(self, label: str):
        """
        Record a memory snapshot.

        Args:
            label: Description of this snapshot (e.g., "After loading model")

        Example:
            >>> tracker = MemoryTracker()
            >>> tracker.snapshot("Initial state")
            >>> tracker.snapshot("After loading")
        """
        # Get RAM usage
        mem_info = self.process.memory_info()
        ram_gb = mem_info.rss / (1024 ** 3)
        ram_percent = self.process.memory_percent()

        # Get GPU usage
        gpu_alloc_gb = 0.0
        gpu_reserved_gb = 0.0
        if torch.cuda.is_available():
            gpu_alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            gpu_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)

        snap = MemorySnapshot(
            timestamp=datetime.now(),
            label=label,
            ram_used_gb=ram_gb,
            ram_percent=ram_percent,
            gpu_allocated_gb=gpu_alloc_gb,
            gpu_reserved_gb=gpu_reserved_gb,
        )

        self.snapshots.append(snap)

    def report(self):
        """
        Report memory change since last snapshot.

        Prints the current memory usage and change since previous snapshot.

        Example:
            >>> tracker.snapshot("Before")
            >>> # ... some operation ...
            >>> tracker.snapshot("After")
            >>> tracker.report()
            Memory: After
              RAM: 12.34 GB (+4.50 GB)
              GPU: 10.20 GB (+5.10 GB)
        """
        if len(self.snapshots) < 2:
            return

        prev = self.snapshots[-2]
        curr = self.snapshots[-1]

        delta_ram = curr.ram_used_gb - prev.ram_used_gb
        delta_gpu = curr.gpu_allocated_gb - prev.gpu_allocated_gb

        print(f"  Memory: {curr.label}")
        print(f"    RAM: {curr.ram_used_gb:.2f} GB ({delta_ram:+.2f} GB)")
        if torch.cuda.is_available():
            print(f"    GPU: {curr.gpu_allocated_gb:.2f} GB ({delta_gpu:+.2f} GB)")

    def summary(self):
        """
        Print summary of all memory snapshots.

        Shows peak memory usage and timeline of all snapshots.

        Example:
            >>> tracker.summary()
            === Memory Usage Summary ===
            Peak RAM: 15.67 GB
            Peak GPU: 12.34 GB

            Timeline:
              Initial               RAM:   1.23 GB  GPU:   0.00 GB
              After loading         RAM:  15.67 GB  GPU:  12.34 GB
              After inference       RAM:  16.01 GB  GPU:  12.40 GB
        """
        if not self.snapshots:
            print("No memory snapshots recorded")
            return

        print("\n=== Memory Usage Summary ===")

        # Find peaks
        peak_ram = max(s.ram_used_gb for s in self.snapshots)
        peak_gpu = max(s.gpu_allocated_gb for s in self.snapshots)

        print(f"Peak RAM: {peak_ram:.2f} GB")
        if torch.cuda.is_available():
            print(f"Peak GPU: {peak_gpu:.2f} GB")

        # Timeline
        print("\nTimeline:")
        for snap in self.snapshots:
            gpu_str = f"GPU: {snap.gpu_allocated_gb:6.2f} GB" if torch.cuda.is_available() else ""
            print(f"  {snap.label:25s} RAM: {snap.ram_used_gb:6.2f} GB  {gpu_str}")

    def get_current_usage(self) -> dict:
        """
        Get current memory usage without recording a snapshot.

        Returns:
            Dictionary with current RAM and GPU usage

        Example:
            >>> tracker = MemoryTracker()
            >>> usage = tracker.get_current_usage()
            >>> usage['ram_gb']
            4.56
            >>> usage['gpu_gb']
            2.34
        """
        mem_info = self.process.memory_info()
        ram_gb = mem_info.rss / (1024 ** 3)

        gpu_gb = 0.0
        if torch.cuda.is_available():
            gpu_gb = torch.cuda.memory_allocated() / (1024 ** 3)

        return {
            'ram_gb': ram_gb,
            'gpu_gb': gpu_gb,
            'ram_percent': self.process.memory_percent(),
        }

    def clear(self):
        """
        Clear all recorded snapshots.

        Example:
            >>> tracker = MemoryTracker()
            >>> tracker.snapshot("Test")
            >>> len(tracker.snapshots)
            1
            >>> tracker.clear()
            >>> len(tracker.snapshots)
            0
        """
        self.snapshots.clear()
