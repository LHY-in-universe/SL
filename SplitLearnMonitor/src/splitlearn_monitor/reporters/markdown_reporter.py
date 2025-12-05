"""
Markdown Reporter - Generate markdown reports
"""
from typing import Optional
from datetime import datetime
from pathlib import Path

from ..core import SystemMonitor, PerformanceTracker
from ..utils import format_duration, format_bytes


class MarkdownReporter:
    """
    Generate Markdown format reports

    Creates human-readable markdown reports with monitoring statistics.

    Example:
        >>> reporter = MarkdownReporter(system_monitor, perf_tracker)
        >>> reporter.generate_report("report.md")
    """

    def __init__(
        self,
        system_monitor: Optional[SystemMonitor] = None,
        performance_tracker: Optional[PerformanceTracker] = None
    ):
        """
        Initialize markdown reporter

        Args:
            system_monitor: SystemMonitor instance (optional)
            performance_tracker: PerformanceTracker instance (optional)
        """
        self.system_monitor = system_monitor
        self.performance_tracker = performance_tracker

    def generate_report(self, output_path: str, title: str = "Monitoring Report"):
        """
        Generate markdown report

        Args:
            output_path: Path to output markdown file
            title: Report title
        """
        lines = []

        # Header
        lines.append(f"# {title}\n")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("---\n")

        # Resource statistics
        if self.system_monitor:
            lines.append("\n## System Resource Usage\n")
            stats = self.system_monitor.get_statistics()

            lines.append(f"- **Duration:** {format_duration(stats.duration_seconds)}")
            lines.append(f"- **Samples Collected:** {stats.sample_count}\n")

            lines.append("### CPU Usage\n")
            lines.append(f"- **Mean:** {stats.cpu_mean:.1f}%")
            lines.append(f"- **Maximum:** {stats.cpu_max:.1f}%")
            lines.append(f"- **P95:** {stats.cpu_p95:.1f}%\n")

            lines.append("### Memory Usage\n")
            lines.append(f"- **Mean:** {stats.memory_mean_mb:.1f} MB")
            lines.append(f"- **Maximum:** {stats.memory_max_mb:.1f} MB")
            lines.append(f"- **P95:** {stats.memory_p95_mb:.1f} MB\n")

            if stats.gpu_available and stats.gpu_mean is not None:
                lines.append("### GPU Usage\n")
                lines.append(f"- **Mean Utilization:** {stats.gpu_mean:.1f}%")
                lines.append(f"- **Maximum Utilization:** {stats.gpu_max:.1f}%")
                lines.append(f"- **P95 Utilization:** {stats.gpu_p95:.1f}%")
                if stats.gpu_memory_mean_mb is not None:
                    lines.append(f"- **Mean Memory:** {stats.gpu_memory_mean_mb:.1f} MB")
                    lines.append(f"- **Maximum Memory:** {stats.gpu_memory_max_mb:.1f} MB\n")

        # Performance statistics
        if self.performance_tracker:
            lines.append("\n## Performance Statistics\n")
            total_time = self.performance_tracker.get_total_time()
            lines.append(f"**Total Time:** {total_time:.2f} ms\n")

            all_stats = self.performance_tracker.get_all_statistics()
            if all_stats:
                lines.append("### Phase Breakdown\n")

                # Create table
                lines.append("| Phase | Count | Total (ms) | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | % of Total |")
                lines.append("|-------|-------|------------|-----------|-------------|----------|----------|------------|")

                # Sort by total time descending
                sorted_stats = sorted(
                    all_stats.items(),
                    key=lambda x: x[1].total_time_ms,
                    reverse=True
                )

                for phase_name, stats in sorted_stats:
                    lines.append(
                        f"| {phase_name} | {stats.count} | "
                        f"{stats.total_time_ms:.1f} | {stats.mean_ms:.1f} | "
                        f"{stats.median_ms:.1f} | {stats.p95_ms:.1f} | "
                        f"{stats.p99_ms:.1f} | {stats.percentage_of_total:.1f}% |"
                    )

        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

    def generate_summary(self) -> str:
        """
        Generate a brief summary string

        Returns:
            Summary text
        """
        lines = []

        if self.system_monitor:
            stats = self.system_monitor.get_statistics()
            lines.append(f"Duration: {format_duration(stats.duration_seconds)}")
            lines.append(f"CPU: {stats.cpu_mean:.1f}% (mean), {stats.cpu_max:.1f}% (max)")
            lines.append(f"Memory: {stats.memory_mean_mb:.1f} MB (mean), {stats.memory_max_mb:.1f} MB (max)")

        if self.performance_tracker:
            total_time = self.performance_tracker.get_total_time()
            lines.append(f"Total Processing Time: {total_time:.2f} ms")

        return "\n".join(lines)
