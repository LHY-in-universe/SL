"""
Server Monitor - Simplified monitoring for Split Learning servers
"""
from typing import Optional

from ..core import SystemMonitor, PerformanceTracker
from ..reporters import HTMLReporter, DataExporter


class ServerMonitor:
    """
    Simplified monitoring interface for Split Learning servers

    Provides monitoring capabilities specifically designed for server-side use,
    with features like automatic report generation and periodic snapshots.

    Example:
        >>> from splitlearn_manager import ManagedServer
        >>> from splitlearn_monitor import ServerMonitor
        >>>
        >>> # Create server with monitoring
        >>> server = ManagedServer(port=50052, ...)
        >>> monitor = ServerMonitor(server_name="trunk_server")
        >>> monitor.start()
        >>>
        >>> # Server runs...
        >>> # Monitor automatically collects data
        >>>
        >>> # Generate report
        >>> monitor.save_report("server_report.html")
    """

    def __init__(
        self,
        server_name: str = "split_server",
        sampling_interval: float = 0.1,
        enable_gpu: bool = True,
        auto_start: bool = False
    ):
        """
        Initialize server monitor

        Args:
            server_name: Name of the server being monitored
            sampling_interval: Time between resource samples in seconds
            enable_gpu: Enable GPU monitoring if available
            auto_start: Automatically start monitoring on initialization
        """
        self.server_name = server_name

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

    def track_request(self, request_type: str = "compute"):
        """
        Context manager for tracking a request

        Args:
            request_type: Type of request (e.g., "compute", "health_check")

        Example:
            >>> with monitor.track_request("compute"):
            ...     result = model(input_data)
        """
        return self.performance_tracker.track_phase(f"request_{request_type}")

    def record_request(self, request_type: str, duration_ms: float, metadata: Optional[dict] = None):
        """
        Manually record a request timing

        Args:
            request_type: Type of request
            duration_ms: Duration in milliseconds
            metadata: Optional metadata
        """
        self.performance_tracker.record_phase(f"request_{request_type}", duration_ms, metadata)

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
            format: Report format ("html", "json")
            include_raw_data: Include raw data in report

        Returns:
            Path to generated report
        """
        # Auto-generate output path if not provided
        if output_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.server_name}_{timestamp}_report.{format}"

        if format == "html":
            reporter = HTMLReporter(self.system_monitor, self.performance_tracker)
            reporter.generate_report(output_path, title=f"Server Monitoring - {self.server_name}",
                                   include_raw_data=include_raw_data)
        elif format == "json":
            exporter = DataExporter(self.system_monitor, self.performance_tracker)
            exporter.export_to_json(output_path, include_raw_snapshots=include_raw_data,
                                  include_raw_timings=include_raw_data)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return output_path

    def get_current_stats(self) -> dict:
        """
        Get current monitoring statistics

        Returns:
            Dictionary with current statistics
        """
        stats = {
            "server_name": self.server_name,
            "is_active": self._is_active,
        }

        # Current resource snapshot
        if self.system_monitor:
            current_snapshot = self.system_monitor.get_current_snapshot()
            if current_snapshot:
                stats["current_resources"] = current_snapshot.to_dict()

            # Overall statistics
            resource_stats = self.system_monitor.get_statistics()
            stats["resource_stats"] = resource_stats.to_dict()

        # Performance statistics
        if self.performance_tracker:
            phase_stats = self.performance_tracker.get_all_statistics()
            stats["phase_stats"] = {
                name: stats_obj.to_dict()
                for name, stats_obj in phase_stats.items()
            }

        return stats

    def print_status(self):
        """Print current monitoring status to console"""
        print(f"\n{'='*80}")
        print(f"Server Monitor Status - {self.server_name}")
        print(f"{'='*80}")
        print(f"Status: {'Active' if self._is_active else 'Inactive'}")

        if self._is_active and self.system_monitor:
            snapshot = self.system_monitor.get_current_snapshot()
            if snapshot:
                print(f"\nCurrent Resource Usage:")
                print(f"  CPU:    {snapshot.cpu_percent:.1f}%")
                print(f"  Memory: {snapshot.memory_mb:.1f}MB ({snapshot.memory_percent:.1f}%)")
                if snapshot.gpu_available and snapshot.gpu_utilization:
                    print(f"  GPU:    {snapshot.gpu_utilization:.1f}%")

        if self.performance_tracker:
            total_time = self.performance_tracker.get_total_time()
            if total_time > 0:
                print(f"\nTotal Processing Time: {total_time:.2f}ms")

        print(f"{'='*80}\n")

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
        return False
