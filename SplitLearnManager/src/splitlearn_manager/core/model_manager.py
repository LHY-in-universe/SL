"""
Model lifecycle management.
"""

import logging
import threading
from typing import Dict, Optional, List
from datetime import datetime
from enum import Enum

import torch.nn as nn

from ..config import ModelConfig
from .model_loader import ModelLoader
from .resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model status states."""
    LOADING = "loading"
    READY = "ready"
    UNLOADING = "unloading"
    FAILED = "failed"


class ManagedModel:
    """
    Wrapper for a managed model with metadata.

    Attributes:
        model_id: Unique model identifier
        model: PyTorch model
        config: Model configuration
        status: Current status
        loaded_at: Load timestamp
        request_count: Total requests processed
        last_used: Last request timestamp
    """

    def __init__(self, model_id: str, model: nn.Module, config: ModelConfig):
        self.model_id = model_id
        self.model = model
        self.config = config
        self.status = ModelStatus.READY
        self.loaded_at = datetime.now()
        self.request_count = 0
        self.last_used = datetime.now()
        self.lock = threading.Lock()

    def get_info(self) -> Dict:
        """Get model information."""
        return {
            "model_id": self.model_id,
            "status": self.status.value,
            "loaded_at": self.loaded_at.isoformat(),
            "last_used": self.last_used.isoformat(),
            "request_count": self.request_count,
            "device": self.config.device,
            "model_type": self.config.model_type,
        }

    def increment_request_count(self):
        """Increment request counter and update last_used."""
        with self.lock:
            self.request_count += 1
            self.last_used = datetime.now()


class ModelManager:
    """
    Manages model lifecycle: loading, unloading, and monitoring.

    Features:
    - Load/unload models on demand
    - Resource management
    - Model status tracking
    - Thread-safe operations
    """

    def __init__(
        self,
        resource_manager: Optional[ResourceManager] = None,
        max_models: int = 5
    ):
        """
        Initialize model manager.

        Args:
            resource_manager: Resource manager instance
            max_models: Maximum number of models to keep loaded
        """
        self.models: Dict[str, ManagedModel] = {}
        self.resource_manager = resource_manager or ResourceManager()
        self.max_models = max_models
        self.lock = threading.Lock()
        self.loader = ModelLoader()

        logger.info(f"ModelManager initialized (max_models={max_models})")

    def load_model(self, config: ModelConfig) -> bool:
        """
        Load a model based on configuration.

        Args:
            config: Model configuration

        Returns:
            True if loaded successfully

        Raises:
            ValueError: If model_id already exists
            RuntimeError: If loading fails
        """
        # Validate configuration
        config.validate()

        with self.lock:
            # Check if already loaded
            if config.model_id in self.models:
                raise ValueError(f"Model {config.model_id} already loaded")

            # Check if we need to unload a model first
            if len(self.models) >= self.max_models:
                logger.info(f"Max models ({self.max_models}) reached, unloading LRU model")
                self._unload_lru_model()

            # Check resources
            if config.max_memory_mb:
                if not self.resource_manager.check_available_resources(
                    required_memory_mb=config.max_memory_mb,
                    required_gpu=config.device.startswith("cuda")
                ):
                    raise RuntimeError("Insufficient resources to load model")

            logger.info(f"Loading model {config.model_id}...")

            try:
                # Load model
                model = self.loader.load_from_config(config)

                # Create managed model
                managed_model = ManagedModel(
                    model_id=config.model_id,
                    model=model,
                    config=config
                )

                # Store
                self.models[config.model_id] = managed_model

                logger.info(
                    f"Successfully loaded model {config.model_id} "
                    f"on {config.device}"
                )

                # Log resource usage
                self.resource_manager.log_resource_usage()

                return True

            except Exception as e:
                logger.error(f"Failed to load model {config.model_id}: {e}")
                raise RuntimeError(f"Model loading failed: {e}")

    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model.

        Args:
            model_id: Model identifier

        Returns:
            True if unloaded successfully

        Raises:
            KeyError: If model not found
        """
        with self.lock:
            if model_id not in self.models:
                raise KeyError(f"Model {model_id} not found")

            logger.info(f"Unloading model {model_id}...")

            managed_model = self.models[model_id]
            managed_model.status = ModelStatus.UNLOADING

            try:
                # Delete model to free memory
                del managed_model.model

                # Remove from registry
                del self.models[model_id]

                # Force garbage collection for GPU memory
                if managed_model.config.device.startswith("cuda"):
                    import torch
                    torch.cuda.empty_cache()

                logger.info(f"Successfully unloaded model {model_id}")
                self.resource_manager.log_resource_usage()

                return True

            except Exception as e:
                logger.error(f"Error unloading model {model_id}: {e}")
                managed_model.status = ModelStatus.FAILED
                return False

    def get_model(self, model_id: str) -> Optional[ManagedModel]:
        """
        Get a managed model.

        Args:
            model_id: Model identifier

        Returns:
            ManagedModel or None if not found
        """
        managed_model = self.models.get(model_id)

        if managed_model:
            managed_model.increment_request_count()

        return managed_model

    def list_models(self) -> List[Dict]:
        """
        List all loaded models with their information.

        Returns:
            List of model information dictionaries
        """
        with self.lock:
            return [model.get_info() for model in self.models.values()]

    def get_model_info(self, model_id: str) -> Dict:
        """
        Get detailed information about a model.

        Args:
            model_id: Model identifier

        Returns:
            Model information dictionary

        Raises:
            KeyError: If model not found
        """
        if model_id not in self.models:
            raise KeyError(f"Model {model_id} not found")

        managed_model = self.models[model_id]
        info = managed_model.get_info()

        # Add additional info
        loader_info = self.loader.get_model_info(
            managed_model.model,
            managed_model.config.device
        )
        info.update(loader_info)

        return info

    def reload_model(self, model_id: str) -> bool:
        """
        Reload a model (unload then load again).

        Args:
            model_id: Model identifier

        Returns:
            True if reloaded successfully
        """
        if model_id not in self.models:
            raise KeyError(f"Model {model_id} not found")

        logger.info(f"Reloading model {model_id}...")

        # Save config
        config = self.models[model_id].config

        # Unload
        self.unload_model(model_id)

        # Load
        return self.load_model(config)

    def _unload_lru_model(self) -> None:
        """Unload least recently used model."""
        if not self.models:
            return

        # Find LRU model
        lru_model_id = min(
            self.models.keys(),
            key=lambda k: self.models[k].last_used
        )

        logger.info(f"Unloading LRU model: {lru_model_id}")
        self.unload_model(lru_model_id)

    def get_statistics(self) -> Dict:
        """
        Get overall statistics.

        Returns:
            Statistics dictionary
        """
        with self.lock:
            total_requests = sum(
                model.request_count for model in self.models.values()
            )

            return {
                "total_models": len(self.models),
                "max_models": self.max_models,
                "total_requests": total_requests,
                "models": {
                    model_id: {
                        "request_count": model.request_count,
                        "status": model.status.value,
                    }
                    for model_id, model in self.models.items()
                }
            }

    def shutdown(self) -> None:
        """Shutdown manager and unload all models."""
        logger.info("Shutting down ModelManager...")

        with self.lock:
            model_ids = list(self.models.keys())

        for model_id in model_ids:
            try:
                self.unload_model(model_id)
            except Exception as e:
                logger.error(f"Error unloading {model_id} during shutdown: {e}")

        logger.info("ModelManager shutdown complete")
