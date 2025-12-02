"""
splitlearn-manager - Model deployment and lifecycle management

This library provides comprehensive model management for distributed deep learning:
- Model loading/unloading
- Resource management (CPU, memory, GPU)
- Request routing and load balancing
- Health monitoring
- Prometheus metrics

Example:
    >>> from splitlearn_manager import ManagedServer, ModelConfig
    >>>
    >>> # Create server
    >>> server = ManagedServer()
    >>>
    >>> # Load a model
    >>> config = ModelConfig(
    ...     model_id="my_model",
    ...     model_path="path/to/model.pt",
    ...     device="cuda"
    ... )
    >>> server.load_model(config)
    >>>
    >>> # Start serving
    >>> server.start()
    >>> server.wait_for_termination()
"""

from .__version__ import __version__

# Configuration
from .config import ModelConfig, ServerConfig

# Core
from .core import ModelManager, ModelLoader, ResourceManager

# Async Core
from .core.async_model_manager import AsyncModelManager, AsyncManagedModel

# Server
from .server import ManagedServer

# Monitoring
from .monitoring import MetricsCollector, HealthChecker

# Routing
from .routing import ModelRouter

__all__ = [
    # Version
    "__version__",

    # Configuration
    "ModelConfig",
    "ServerConfig",

    # Core
    "ModelManager",
    "ModelLoader",
    "ResourceManager",

    # Async Core
    "AsyncModelManager",
    "AsyncManagedModel",

    # Server
    "ManagedServer",

    # Monitoring
    "MetricsCollector",
    "HealthChecker",

    # Routing
    "ModelRouter",
]
