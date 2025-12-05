"""
Client Monitor - Simplified monitoring for Split Learning clients
"""
from typing import Optional, List, Dict
from contextlib import contextmanager

from ..core import SystemMonitor, PerformanceTracker
from ..reporters import HTMLReporter, DataExporter


class ClientMonitor:
    """
    Simplified monitoring interface for Split Learning clients

    Provides an easy-to-use wrapper around SystemMonitor and PerformanceTracker
    specifically designed for client-side monitoring.

    Example:
        >>> monitor = ClientMonitor(session_name="my_session")
        >>> monitor.start()
        >>>
        >>> # Track your phases
        >>> with monitor.track_phase("bottom_model"):
        ...     hidden = bottom(input_ids)
        >>>
        >>> with monitor.track_phase("trunk_remote"):
        ...     hidden = trunk_client.compute(hidden)
        >>>
        >>> with monitor.track_phase("top_model"):
        ...     output = top(hidden)
        >>>
        >>> monitor.stop()
        >>> monitor.save_report("report.html")
    """

    def __init__(
        self,
        session_name: str = "client_session",
        sampling_interval: float = 0.1,
        enable_gpu: bool = True,
        auto_start: bool = False
    ):
        """
        Initialize client monitor

        Args:
            session_name: Name for this monitoring session
            sampling_interval: Time between resource samples in seconds
            enable_gpu: Enable GPU monitoring if available
            auto_start: Automatically start monitoring on initialization
        """
        self.session_name = session_name

        # Initialize monitors
        self.system_monitor = SystemMonitor(
            sampling_interval=sampling_interval,
            enable_gpu=enable_gpu
        )
        self.performance_tracker = PerformanceTracker()

        # Track whether monitoring is active
        self._is_active = False

        if auto_start:
            self.start()

    def start(self):
        """Start monitoring"""
        if not self._is_active:
            self.system_monitor.start()
            self._is_active = True

    def stop(self):
        """Stop monitoring"""
        if self._is_active:
            self.system_monitor.stop()
            self._is_active = False

    def track_phase(self, phase_name: str, metadata: Optional[dict] = None):
        """
        Context manager for tracking a phase

        Args:
            phase_name: Name of the phase
            metadata: Optional metadata

        Example:
            >>> with monitor.track_phase("computation"):
            ...     result = compute()
        """
        return self.performance_tracker.track_phase(phase_name, metadata)

    def record_phase(self, phase_name: str, duration_ms: float, metadata: Optional[dict] = None):
        """
        Manually record a phase timing

        Args:
            phase_name: Name of the phase
            duration_ms: Duration in milliseconds
            metadata: Optional metadata
        """
        self.performance_tracker.record_phase(phase_name, duration_ms, metadata)

    def save_report(
        self,
        output_path: Optional[str] = None,
        format: str = "html",
        include_raw_data: bool = False
    ) -> str:
        """
        Save monitoring report

        Args:
            output_path: Output file path (auto-generated if None)
            format: Report format ("html", "markdown", "json")
            include_raw_data: Include raw data in report

        Returns:
            Path to generated report
        """
        # Auto-generate output path if not provided
        if output_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.session_name}_{timestamp}_report.{format}"

        if format == "html":
            reporter = HTMLReporter(self.system_monitor, self.performance_tracker)
            reporter.generate_report(output_path, title=f"Client Monitoring - {self.session_name}",
                                   include_raw_data=include_raw_data)
        elif format == "json":
            exporter = DataExporter(self.system_monitor, self.performance_tracker)
            exporter.export_to_json(output_path, include_raw_snapshots=include_raw_data,
                                  include_raw_timings=include_raw_data)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return output_path

    def save_merged_report(
        self,
        server_monitoring_snapshots: List[Dict],
        output_path: Optional[str] = None,
        include_raw_data: bool = False
    ) -> str:
        """
        保存合并的客户端+服务端监控报告

        Args:
            server_monitoring_snapshots: 服务端监控快照列表（从 GRPCComputeClient 获取）
            output_path: 输出文件路径（自动生成如果为 None）
            include_raw_data: 是否包含原始数据

        Returns:
            报告文件路径
        """
        from ..reporters.merged_reporter import MergedHTMLReporter
        from datetime import datetime

        # 自动生成输出路径
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.session_name}_merged_{timestamp}_report.html"

        # 生成合并报告
        reporter = MergedHTMLReporter(
            client_system_monitor=self.system_monitor,
            client_performance_tracker=self.performance_tracker,
            server_monitoring_snapshots=server_monitoring_snapshots
        )

        reporter.generate_report(
            output_path,
            title=f"合并监控报告 - {self.session_name}",
            include_raw_data=include_raw_data
        )

        return output_path

    def get_summary(self) -> dict:
        """
        Get monitoring summary

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "session_name": self.session_name,
            "is_active": self._is_active,
        }

        # Resource statistics
        if self.system_monitor:
            resource_stats = self.system_monitor.get_statistics()
            summary["resource_stats"] = resource_stats.to_dict()

        # Performance statistics
        if self.performance_tracker:
            phase_stats = self.performance_tracker.get_all_statistics()
            summary["phase_stats"] = {
                name: stats.to_dict()
                for name, stats in phase_stats.items()
            }
            summary["total_time_ms"] = self.performance_tracker.get_total_time()

        return summary

    def print_summary(self):
        """Print monitoring summary to console"""
        print(f"\n{'='*80}")
        print(f"Monitoring Summary - {self.session_name}")
        print(f"{'='*80}")

        # Resource summary
        if self.system_monitor:
            stats = self.system_monitor.get_statistics()
            print(f"\nResource Usage:")
            print(f"  CPU:    Mean={stats.cpu_mean:.1f}%, Max={stats.cpu_max:.1f}%")
            print(f"  Memory: Mean={stats.memory_mean_mb:.1f}MB, Max={stats.memory_max_mb:.1f}MB")
            if stats.gpu_available and stats.gpu_mean:
                print(f"  GPU:    Mean={stats.gpu_mean:.1f}%, Max={stats.gpu_max:.1f}%")

        # Performance summary
        if self.performance_tracker:
            print(f"\nPerformance:")
            print(f"  Total Time: {self.performance_tracker.get_total_time():.2f}ms")
            phase_breakdown = self.performance_tracker.get_phase_breakdown()
            for phase, percentage in sorted(phase_breakdown.items(), key=lambda x: x[1], reverse=True):
                print(f"    {phase}: {percentage:.1f}%")

        print(f"{'='*80}\n")

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
        return False


@contextmanager
def quick_monitor(session_name: str = "session", auto_save: bool = True, **kwargs):
    """
    Quick monitoring context manager

    Automatically starts monitoring, tracks the code block, and saves a report.

    Args:
        session_name: Name for the monitoring session
        auto_save: Automatically save report on exit
        **kwargs: Additional arguments for ClientMonitor

    Example:
        >>> with quick_monitor("my_task") as monitor:
        ...     with monitor.track_phase("computation"):
        ...         result = compute()
        # Automatically generates report: my_task_TIMESTAMP_report.html
    """
    monitor = ClientMonitor(session_name=session_name, **kwargs)
    monitor.start()

    try:
        yield monitor
    finally:
        monitor.stop()
        if auto_save:
            report_path = monitor.save_report()
            print(f"Report saved: {report_path}")
