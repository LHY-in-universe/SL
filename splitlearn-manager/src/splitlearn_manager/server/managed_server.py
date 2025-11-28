"""
Managed gRPC server with model lifecycle management.
"""

import logging
import time
import threading
from typing import Optional, Dict
import torch

from splitlearn_comm import GRPCComputeServer
from splitlearn_comm.core import ComputeFunction

from ..config import ModelConfig, ServerConfig
from ..core import ModelManager, ResourceManager
from ..monitoring import MetricsCollector, HealthChecker
from ..routing import ModelRouter

logger = logging.getLogger(__name__)


class ManagedComputeFunction(ComputeFunction):
    """
    ComputeFunction that routes requests through ModelManager.
    """

    def __init__(self, model_manager: ModelManager, router: ModelRouter, metrics: Optional[MetricsCollector] = None):
        """
        Initialize managed compute function.

        Args:
            model_manager: Model manager instance
            router: Model router instance
            metrics: Metrics collector (optional)
        """
        self.model_manager = model_manager
        self.router = router
        self.metrics = metrics

    def compute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform computation using managed models.

        Args:
            input_tensor: Input tensor

        Returns:
            Output tensor

        Raises:
            RuntimeError: If no models available or computation fails
        """
        start_time = time.time()

        # Route to a model
        model_id = self.router.route_to_model()

        if not model_id:
            raise RuntimeError("No models available")

        # Get managed model
        managed_model = self.model_manager.get_model(model_id)

        if not managed_model:
            raise RuntimeError(f"Model {model_id} not found")

        # Perform computation
        try:
            with torch.no_grad():
                output = managed_model.model(input_tensor)

            # Record success metrics
            if self.metrics:
                duration = time.time() - start_time
                self.metrics.record_inference_request(model_id, duration, True)

            return output

        except Exception as e:
            # Record failure metrics
            if self.metrics:
                duration = time.time() - start_time
                self.metrics.record_inference_request(model_id, duration, False)

            logger.error(f"Computation failed for model {model_id}: {e}")
            raise RuntimeError(f"Computation failed: {e}")

    def get_info(self) -> Dict:
        """Get service information."""
        models_info = self.model_manager.list_models()

        return {
            "name": "ManagedComputeService",
            "num_models": len(models_info),
            "models": [
                {
                    "model_id": info["model_id"],
                    "status": info["status"],
                    "request_count": info["request_count"]
                }
                for info in models_info
            ]
        }


class ManagedServer:
    """
    Managed gRPC server with full model lifecycle management.

    Features:
    - Model loading/unloading
    - Resource management
    - Request routing
    - Health checks
    - Prometheus metrics
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        """
        Initialize managed server.

        Args:
            config: Server configuration (uses defaults if None)
        """
        self.config = config or ServerConfig()
        self.config.validate()

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Initialize components
        logger.info("Initializing ManagedServer...")

        self.resource_manager = ResourceManager()
        self.model_manager = ModelManager(
            resource_manager=self.resource_manager,
            max_models=self.config.max_models
        )
        self.router = ModelRouter(self.model_manager)
        self.health_checker = HealthChecker(
            resource_manager=self.resource_manager,
            model_manager=self.model_manager
        )

        # Metrics (optional)
        self.metrics = None
        if self.config.enable_monitoring:
            self.metrics = MetricsCollector(port=self.config.metrics_port)

        # Create compute function
        compute_fn = ManagedComputeFunction(
            model_manager=self.model_manager,
            router=self.router,
            metrics=self.metrics
        )

        # Create gRPC server
        self.grpc_server = GRPCComputeServer(
            compute_fn=compute_fn,
            host=self.config.host,
            port=self.config.port,
            max_workers=self.config.max_workers
        )

        # Background threads
        self.monitoring_thread = None
        self.running = False

        logger.info("ManagedServer initialized successfully")

    def load_model(self, config: ModelConfig) -> bool:
        """
        Load a model.

        Args:
            config: Model configuration

        Returns:
            True if loaded successfully
        """
        try:
            success = self.model_manager.load_model(config)

            # Update metrics
            if self.metrics:
                self.metrics.record_model_load(config.model_id, success)
                self.metrics.update_models_loaded(len(self.model_manager.models))

            return success

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            if self.metrics:
                self.metrics.record_model_load(config.model_id, False)
            raise

    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model.

        Args:
            model_id: Model identifier

        Returns:
            True if unloaded successfully
        """
        try:
            success = self.model_manager.unload_model(model_id)

            # Update metrics
            if self.metrics:
                self.metrics.record_model_unload(model_id)
                self.metrics.update_models_loaded(len(self.model_manager.models))

            return success

        except Exception as e:
            logger.error(f"Failed to unload model: {e}")
            raise

    def start(self):
        """Start the managed server."""
        logger.info("Starting ManagedServer...")

        self.running = True

        # Start metrics server
        if self.metrics:
            self.metrics.start_server()

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()

        # Start gRPC server
        self.grpc_server.start()

        logger.info(
            f"ManagedServer started on {self.config.host}:{self.config.port}"
        )

    def wait_for_termination(self):
        """Wait for server termination."""
        self.grpc_server.wait_for_termination()

    def stop(self, grace: float = 5.0):
        """
        Stop the managed server.

        Args:
            grace: Grace period in seconds
        """
        logger.info("Stopping ManagedServer...")

        self.running = False

        # Stop gRPC server
        self.grpc_server.stop(grace=grace)

        # Shutdown model manager
        self.model_manager.shutdown()

        logger.info("ManagedServer stopped")

    def _monitoring_loop(self):
        """Background monitoring loop."""
        logger.info("Monitoring loop started")

        while self.running:
            try:
                # Update resource metrics
                if self.metrics:
                    usage = self.resource_manager.get_current_usage()
                    self.metrics.update_resource_metrics(
                        cpu_percent=usage.cpu_percent,
                        memory_mb=usage.memory_mb,
                        memory_percent=usage.memory_percent,
                        gpu_memory_mb=usage.gpu_memory_mb
                    )

                # Perform health check
                health_result = self.health_checker.check_health()
                if health_result["status"] != "healthy":
                    logger.warning(f"Health check: {health_result['status']}")

                # Sleep until next check
                time.sleep(self.config.health_check_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config.health_check_interval)

        logger.info("Monitoring loop stopped")

    def get_status(self) -> Dict:
        """
        Get server status.

        Returns:
            Status dictionary
        """
        return {
            "server": {
                "host": self.config.host,
                "port": self.config.port,
                "running": self.running
            },
            "models": self.model_manager.list_models(),
            "statistics": self.model_manager.get_statistics(),
            "health": self.health_checker.check_health(),
            "resources": self.resource_manager.get_current_usage().__dict__
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
