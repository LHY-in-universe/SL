"""
Configuration classes for SplitLearnMonitor
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MonitorConfig:
    """
    Configuration for the monitoring system

    Example:
        >>> config = MonitorConfig(sampling_interval=0.2, enable_gpu=False)
        >>> config.validate()
        True
    """
    # System monitoring
    sampling_interval: float = 0.1  # seconds between samples
    max_samples: int = 10000  # maximum number of samples to retain
    enable_gpu: bool = True  # enable GPU monitoring if available

    # Performance tracking
    track_phases: List[str] = field(default_factory=lambda: [
        "bottom", "trunk", "top", "network"
    ])
    enable_nested_tracking: bool = True  # allow nested phase tracking

    # Visualization
    viz_backend: str = "matplotlib"  # "matplotlib" or "plotly"
    chart_theme: str = "default"  # chart color theme
    output_format: str = "png"  # "png", "html", "svg"
    dpi: int = 100  # DPI for rasterized outputs
    figure_size: tuple = (12, 8)  # default figure size in inches

    # Reporting
    include_raw_data: bool = False  # include raw data in reports
    compress_reports: bool = False  # compress large reports
    report_format: str = "html"  # "html", "markdown", "json"

    # GPU settings
    gpu_device_id: int = 0  # which GPU device to monitor

    def validate(self) -> bool:
        """
        Validate configuration parameters

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        if self.sampling_interval <= 0:
            raise ValueError("sampling_interval must be positive")

        if self.max_samples <= 0:
            raise ValueError("max_samples must be positive")

        if self.viz_backend not in ["matplotlib", "plotly"]:
            raise ValueError("viz_backend must be 'matplotlib' or 'plotly'")

        if self.output_format not in ["png", "html", "svg", "pdf"]:
            raise ValueError("output_format must be one of: png, html, svg, pdf")

        if self.report_format not in ["html", "markdown", "json"]:
            raise ValueError("report_format must be one of: html, markdown, json")

        if self.gpu_device_id < 0:
            raise ValueError("gpu_device_id must be non-negative")

        return True

    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()


@dataclass
class ResourceSnapshot:
    """
    Snapshot of system resource usage at a point in time

    Attributes:
        timestamp: Unix timestamp of the snapshot
        cpu_percent: CPU utilization percentage (0-100)
        memory_mb: Memory usage in megabytes
        memory_percent: Memory utilization percentage (0-100)
        gpu_available: Whether GPU monitoring is available
        gpu_utilization: GPU utilization percentage (0-100) if available
        gpu_memory_used_mb: GPU memory used in megabytes if available
        gpu_memory_total_mb: Total GPU memory in megabytes if available
    """
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    gpu_available: bool
    gpu_utilization: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "memory_percent": self.memory_percent,
            "gpu_available": self.gpu_available,
            "gpu_utilization": self.gpu_utilization,
            "gpu_memory_used_mb": self.gpu_memory_used_mb,
            "gpu_memory_total_mb": self.gpu_memory_total_mb,
        }


@dataclass
class ResourceStatistics:
    """
    Statistical summary of resource usage

    Attributes:
        duration_seconds: Duration of monitoring period
        cpu_mean: Mean CPU utilization
        cpu_max: Maximum CPU utilization
        cpu_p95: 95th percentile CPU utilization
        memory_mean_mb: Mean memory usage in MB
        memory_max_mb: Maximum memory usage in MB
        memory_p95_mb: 95th percentile memory usage in MB
        gpu_available: Whether GPU statistics are available
        gpu_mean: Mean GPU utilization (if available)
        gpu_max: Maximum GPU utilization (if available)
        gpu_p95: 95th percentile GPU utilization (if available)
        gpu_memory_mean_mb: Mean GPU memory usage in MB (if available)
        gpu_memory_max_mb: Maximum GPU memory usage in MB (if available)
    """
    duration_seconds: float
    sample_count: int
    cpu_mean: float
    cpu_max: float
    cpu_p95: float
    memory_mean_mb: float
    memory_max_mb: float
    memory_p95_mb: float
    gpu_available: bool
    gpu_mean: Optional[float] = None
    gpu_max: Optional[float] = None
    gpu_p95: Optional[float] = None
    gpu_memory_mean_mb: Optional[float] = None
    gpu_memory_max_mb: Optional[float] = None
    gpu_memory_p95_mb: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "duration_seconds": self.duration_seconds,
            "sample_count": self.sample_count,
            "cpu_mean": self.cpu_mean,
            "cpu_max": self.cpu_max,
            "cpu_p95": self.cpu_p95,
            "memory_mean_mb": self.memory_mean_mb,
            "memory_max_mb": self.memory_max_mb,
            "memory_p95_mb": self.memory_p95_mb,
            "gpu_available": self.gpu_available,
            "gpu_mean": self.gpu_mean,
            "gpu_max": self.gpu_max,
            "gpu_p95": self.gpu_p95,
            "gpu_memory_mean_mb": self.gpu_memory_mean_mb,
            "gpu_memory_max_mb": self.gpu_memory_max_mb,
            "gpu_memory_p95_mb": self.gpu_memory_p95_mb,
        }


@dataclass
class PhaseStatistics:
    """
    Statistical analysis for a specific phase

    Attributes:
        phase_name: Name of the phase (e.g., "bottom_model", "trunk_remote")
        count: Number of times this phase was executed
        total_time_ms: Total time spent in this phase in milliseconds
        mean_ms: Mean execution time in milliseconds
        median_ms: Median execution time in milliseconds
        min_ms: Minimum execution time in milliseconds
        max_ms: Maximum execution time in milliseconds
        p50_ms: 50th percentile (median) execution time in milliseconds
        p95_ms: 95th percentile execution time in milliseconds
        p99_ms: 99th percentile execution time in milliseconds
        std_ms: Standard deviation of execution time in milliseconds
        percentage_of_total: Percentage of total time spent in this phase
    """
    phase_name: str
    count: int
    total_time_ms: float
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float
    percentage_of_total: float

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "phase_name": self.phase_name,
            "count": self.count,
            "total_time_ms": self.total_time_ms,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "std_ms": self.std_ms,
            "percentage_of_total": self.percentage_of_total,
        }
