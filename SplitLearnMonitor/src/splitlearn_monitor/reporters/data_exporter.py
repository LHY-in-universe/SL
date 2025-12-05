"""
Data Exporter - Export monitoring data to JSON and CSV
"""
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from ..core import SystemMonitor, PerformanceTracker
from ..config import ResourceSnapshot, PhaseStatistics


class DataExporter:
    """
    Export monitoring data to various formats

    Supports JSON and CSV exports of resource snapshots and performance statistics.

    Example:
        >>> exporter = DataExporter(system_monitor, perf_tracker)
        >>> exporter.export_to_json("monitoring_data.json")
        >>> exporter.export_to_csv("output_dir")
    """

    def __init__(
        self,
        system_monitor: Optional[SystemMonitor] = None,
        performance_tracker: Optional[PerformanceTracker] = None
    ):
        """
        Initialize data exporter

        Args:
            system_monitor: SystemMonitor instance (optional)
            performance_tracker: PerformanceTracker instance (optional)
        """
        self.system_monitor = system_monitor
        self.performance_tracker = performance_tracker

    def export_to_json(
        self,
        output_path: str,
        include_raw_snapshots: bool = True,
        include_raw_timings: bool = True,
        pretty: bool = True
    ):
        """
        Export all monitoring data to JSON

        Args:
            output_path: Path to output JSON file
            include_raw_snapshots: Include raw resource snapshots
            include_raw_timings: Include raw phase timings
            pretty: Use pretty printing (indented)
        """
        data = self._collect_all_data(include_raw_snapshots, include_raw_timings)

        # Add metadata
        data["metadata"] = {
            "export_time": datetime.now().isoformat(),
            "format_version": "1.0.0"
        }

        # Write JSON
        with open(output_path, 'w') as f:
            if pretty:
                json.dump(data, f, indent=2)
            else:
                json.dump(data, f)

    def export_to_csv(self, output_dir: str):
        """
        Export monitoring data to multiple CSV files

        Creates separate CSV files for:
        - resource_snapshots.csv: Resource usage over time
        - phase_statistics.csv: Phase performance statistics
        - phase_timings.csv: Raw phase timings (optional, can be large)

        Args:
            output_dir: Directory to write CSV files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export resource snapshots
        if self.system_monitor:
            self._export_resource_snapshots_csv(output_path / "resource_snapshots.csv")

        # Export phase statistics
        if self.performance_tracker:
            self._export_phase_statistics_csv(output_path / "phase_statistics.csv")
            self._export_phase_timings_csv(output_path / "phase_timings.csv")

    def _collect_all_data(
        self,
        include_raw_snapshots: bool = True,
        include_raw_timings: bool = True
    ) -> Dict:
        """Collect all monitoring data into a dictionary"""
        data = {}

        # System monitoring data
        if self.system_monitor:
            resource_stats = self.system_monitor.get_statistics()
            data["resource_statistics"] = resource_stats.to_dict()

            if include_raw_snapshots:
                snapshots = self.system_monitor.get_history()
                data["resource_snapshots"] = [s.to_dict() for s in snapshots]

        # Performance tracking data
        if self.performance_tracker:
            all_stats = self.performance_tracker.get_all_statistics()
            data["phase_statistics"] = {
                name: stats.to_dict()
                for name, stats in all_stats.items()
            }

            if include_raw_timings:
                data["phase_timings"] = {}
                for phase_name in self.performance_tracker.get_phase_names():
                    data["phase_timings"][phase_name] = self.performance_tracker.get_phase_timings(phase_name)

            data["total_time_ms"] = self.performance_tracker.get_total_time()

        return data

    def _export_resource_snapshots_csv(self, output_path: Path):
        """Export resource snapshots to CSV"""
        if not self.system_monitor:
            return

        snapshots = self.system_monitor.get_history()
        if not snapshots:
            return

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = [
                "timestamp",
                "cpu_percent",
                "memory_mb",
                "memory_percent",
                "gpu_available",
                "gpu_utilization",
                "gpu_memory_used_mb",
                "gpu_memory_total_mb"
            ]
            writer.writerow(header)

            # Data rows
            for snapshot in snapshots:
                row = [
                    snapshot.timestamp,
                    snapshot.cpu_percent,
                    snapshot.memory_mb,
                    snapshot.memory_percent,
                    snapshot.gpu_available,
                    snapshot.gpu_utilization if snapshot.gpu_utilization is not None else "",
                    snapshot.gpu_memory_used_mb if snapshot.gpu_memory_used_mb is not None else "",
                    snapshot.gpu_memory_total_mb if snapshot.gpu_memory_total_mb is not None else ""
                ]
                writer.writerow(row)

    def _export_phase_statistics_csv(self, output_path: Path):
        """Export phase statistics to CSV"""
        if not self.performance_tracker:
            return

        all_stats = self.performance_tracker.get_all_statistics()
        if not all_stats:
            return

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = [
                "phase_name",
                "count",
                "total_time_ms",
                "mean_ms",
                "median_ms",
                "min_ms",
                "max_ms",
                "p50_ms",
                "p95_ms",
                "p99_ms",
                "std_ms",
                "percentage_of_total"
            ]
            writer.writerow(header)

            # Data rows
            for phase_name, stats in all_stats.items():
                row = [
                    stats.phase_name,
                    stats.count,
                    stats.total_time_ms,
                    stats.mean_ms,
                    stats.median_ms,
                    stats.min_ms,
                    stats.max_ms,
                    stats.p50_ms,
                    stats.p95_ms,
                    stats.p99_ms,
                    stats.std_ms,
                    stats.percentage_of_total
                ]
                writer.writerow(row)

    def _export_phase_timings_csv(self, output_path: Path):
        """Export raw phase timings to CSV"""
        if not self.performance_tracker:
            return

        phase_names = self.performance_tracker.get_phase_names()
        if not phase_names:
            return

        # Get all timings
        all_timings = {}
        max_length = 0
        for phase_name in phase_names:
            timings = self.performance_tracker.get_phase_timings(phase_name)
            all_timings[phase_name] = timings
            max_length = max(max_length, len(timings))

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(["index"] + phase_names)

            # Data rows
            for i in range(max_length):
                row = [i]
                for phase_name in phase_names:
                    timings = all_timings[phase_name]
                    if i < len(timings):
                        row.append(timings[i])
                    else:
                        row.append("")
                writer.writerow(row)

    def export_summary_txt(self, output_path: str):
        """
        Export a text summary of monitoring data

        Args:
            output_path: Path to output text file
        """
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Monitoring Summary Report\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Resource statistics
            if self.system_monitor:
                stats = self.system_monitor.get_statistics()
                f.write("Resource Usage Statistics\n")
                f.write("-" * 80 + "\n")
                f.write(f"Duration: {stats.duration_seconds:.1f} seconds\n")
                f.write(f"Sample Count: {stats.sample_count}\n\n")

                f.write(f"CPU Usage:\n")
                f.write(f"  Mean: {stats.cpu_mean:.1f}%\n")
                f.write(f"  Max:  {stats.cpu_max:.1f}%\n")
                f.write(f"  P95:  {stats.cpu_p95:.1f}%\n\n")

                f.write(f"Memory Usage:\n")
                f.write(f"  Mean: {stats.memory_mean_mb:.1f} MB\n")
                f.write(f"  Max:  {stats.memory_max_mb:.1f} MB\n")
                f.write(f"  P95:  {stats.memory_p95_mb:.1f} MB\n\n")

                if stats.gpu_available:
                    f.write(f"GPU Usage:\n")
                    f.write(f"  Mean: {stats.gpu_mean:.1f}%\n")
                    f.write(f"  Max:  {stats.gpu_max:.1f}%\n")
                    f.write(f"  P95:  {stats.gpu_p95:.1f}%\n\n")

            # Performance statistics
            if self.performance_tracker:
                f.write("\nPerformance Statistics\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total Time: {self.performance_tracker.get_total_time():.2f} ms\n\n")

                all_stats = self.performance_tracker.get_all_statistics()
                for phase_name, phase_stats in all_stats.items():
                    f.write(f"{phase_name}:\n")
                    f.write(f"  Count:      {phase_stats.count}\n")
                    f.write(f"  Total:      {phase_stats.total_time_ms:.2f} ms ({phase_stats.percentage_of_total:.1f}%)\n")
                    f.write(f"  Mean:       {phase_stats.mean_ms:.2f} ms\n")
                    f.write(f"  Median:     {phase_stats.median_ms:.2f} ms\n")
                    f.write(f"  Min/Max:    {phase_stats.min_ms:.2f} / {phase_stats.max_ms:.2f} ms\n")
                    f.write(f"  P95/P99:    {phase_stats.p95_ms:.2f} / {phase_stats.p99_ms:.2f} ms\n")
                    f.write(f"  Std Dev:    {phase_stats.std_ms:.2f} ms\n\n")

            f.write("=" * 80 + "\n")
