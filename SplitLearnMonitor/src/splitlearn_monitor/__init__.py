"""
SplitLearnMonitor - Performance monitoring library for Split Learning

This library provides comprehensive monitoring capabilities for Split Learning
systems, including CPU, GPU, and memory tracking, as well as phase-based
performance analysis.

Example:
    >>> from splitlearn_monitor import SystemMonitor, PerformanceTracker, HTMLReporter
    >>>
    >>> # Start monitoring
    >>> sys_monitor = SystemMonitor(sampling_interval=0.1, enable_gpu=True)
    >>> perf_tracker = PerformanceTracker()
    >>>
    >>> sys_monitor.start()
    >>>
    >>> # Track your phases
    >>> with perf_tracker.track_phase("bottom_model"):
    ...     # Your code here
    ...     pass
    >>>
    >>> sys_monitor.stop()
    >>>
    >>> # Generate report
    >>> reporter = HTMLReporter(sys_monitor, perf_tracker)
    >>> reporter.generate_report("monitoring_report.html")
"""

from .__version__ import __version__, __author__, __license__

# Core monitoring
from .core import SystemMonitor, PerformanceTracker

# Configuration
from .config import (
    MonitorConfig,
    ResourceSnapshot,
    ResourceStatistics,
    PhaseStatistics,
)

# Collectors
from .collectors import (
    CPUCollector,
    MemoryCollector,
    GPUCollector,
    GPU_AVAILABLE,
)

# Visualizers
from .visualizers import (
    TimeSeriesVisualizer,
    ComparisonVisualizer,
    DistributionVisualizer,
)

# Reporters
from .reporters import (
    HTMLReporter,
    MarkdownReporter,
    DataExporter,
    MergedHTMLReporter,
)

# Integrations
from .integrations import (
    ClientMonitor,
    ServerMonitor,
    FullModelMonitor,
)

# Utilities
from .utils import (
    calculate_percentile,
    calculate_statistics,
    format_bytes,
    format_duration,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Core
    "SystemMonitor",
    "PerformanceTracker",
    # Config
    "MonitorConfig",
    "ResourceSnapshot",
    "ResourceStatistics",
    "PhaseStatistics",
    # Collectors
    "CPUCollector",
    "MemoryCollector",
    "GPUCollector",
    "GPU_AVAILABLE",
    # Visualizers
    "TimeSeriesVisualizer",
    "ComparisonVisualizer",
    "DistributionVisualizer",
    # Reporters
    "HTMLReporter",
    "MarkdownReporter",
    "DataExporter",
    "MergedHTMLReporter",
    # Integrations
    "ClientMonitor",
    "ServerMonitor",
    "FullModelMonitor",
    # Utilities
    "calculate_percentile",
    "calculate_statistics",
    "format_bytes",
    "format_duration",
]
