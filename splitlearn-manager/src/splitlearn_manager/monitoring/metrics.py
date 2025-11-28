"""
Prometheus metrics collection.
"""

import logging
from typing import Optional
from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects and exposes Prometheus metrics.

    Metrics:
    - model_load_total: Total number of model loads
    - model_unload_total: Total number of model unloads
    - models_loaded: Currently loaded models
    - inference_requests_total: Total inference requests
    - inference_duration_seconds: Inference duration histogram
    - resource_usage: CPU, memory, GPU usage gauges
    """

    def __init__(self, port: int = 8000):
        """
        Initialize metrics collector.

        Args:
            port: Port for Prometheus HTTP server
        """
        self.port = port
        self._initialize_metrics()

    def _initialize_metrics(self):
        """Initialize Prometheus metrics."""

        # Model lifecycle metrics
        self.model_load_total = Counter(
            'model_load_total',
            'Total number of model loads',
            ['model_id', 'status']
        )

        self.model_unload_total = Counter(
            'model_unload_total',
            'Total number of model unloads',
            ['model_id']
        )

        self.models_loaded = Gauge(
            'models_loaded',
            'Number of currently loaded models'
        )

        # Inference metrics
        self.inference_requests_total = Counter(
            'inference_requests_total',
            'Total number of inference requests',
            ['model_id', 'status']
        )

        self.inference_duration_seconds = Histogram(
            'inference_duration_seconds',
            'Inference duration in seconds',
            ['model_id'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )

        # Resource metrics
        self.cpu_usage_percent = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage'
        )

        self.memory_usage_mb = Gauge(
            'memory_usage_mb',
            'Memory usage in MB'
        )

        self.memory_usage_percent = Gauge(
            'memory_usage_percent',
            'Memory usage percentage'
        )

        self.gpu_memory_mb = Gauge(
            'gpu_memory_mb',
            'GPU memory usage in MB',
            ['gpu_id']
        )

        logger.info("Metrics initialized")

    def start_server(self):
        """Start Prometheus HTTP server."""
        try:
            start_http_server(self.port)
            logger.info(f"Metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")

    def record_model_load(self, model_id: str, success: bool):
        """Record a model load event."""
        status = "success" if success else "failure"
        self.model_load_total.labels(model_id=model_id, status=status).inc()

    def record_model_unload(self, model_id: str):
        """Record a model unload event."""
        self.model_unload_total.labels(model_id=model_id).inc()

    def update_models_loaded(self, count: int):
        """Update currently loaded models count."""
        self.models_loaded.set(count)

    def record_inference_request(
        self,
        model_id: str,
        duration: float,
        success: bool
    ):
        """
        Record an inference request.

        Args:
            model_id: Model identifier
            duration: Request duration in seconds
            success: Whether request succeeded
        """
        status = "success" if success else "failure"
        self.inference_requests_total.labels(
            model_id=model_id,
            status=status
        ).inc()

        if success:
            self.inference_duration_seconds.labels(model_id=model_id).observe(duration)

    def update_resource_metrics(
        self,
        cpu_percent: float,
        memory_mb: float,
        memory_percent: float,
        gpu_memory_mb: Optional[dict] = None
    ):
        """
        Update resource usage metrics.

        Args:
            cpu_percent: CPU usage percentage
            memory_mb: Memory usage in MB
            memory_percent: Memory usage percentage
            gpu_memory_mb: GPU memory usage dict {gpu_id: mb}
        """
        self.cpu_usage_percent.set(cpu_percent)
        self.memory_usage_mb.set(memory_mb)
        self.memory_usage_percent.set(memory_percent)

        if gpu_memory_mb:
            for gpu_id, mem_mb in gpu_memory_mb.items():
                self.gpu_memory_mb.labels(gpu_id=str(gpu_id)).set(mem_mb)
