"""
Model configuration management.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import yaml


@dataclass
class ModelConfig:
    """
    Configuration for a managed model.

    Attributes:
        model_id: Unique identifier for the model
        model_path: Path to model weights/checkpoint
        model_type: Type of model (e.g., "gpt2", "bert", "custom")
        device: Device to load model on ("cpu", "cuda", "cuda:0", etc.)
        batch_size: Maximum batch size for inference
        max_memory_mb: Maximum memory allocation in MB
        warmup: Whether to warmup model after loading
        config: Additional model-specific configuration
    """

    model_id: str
    model_path: str
    model_type: str = "custom"
    device: str = "cpu"
    batch_size: int = 32
    max_memory_mb: Optional[int] = None
    warmup: bool = True
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        """
        Load model configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            ModelConfig instance

        Example:
            >>> config = ModelConfig.from_yaml("model.yaml")
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, yaml_path: str) -> None:
        """
        Save model configuration to YAML file.

        Args:
            yaml_path: Path to save YAML configuration

        Example:
            >>> config.to_yaml("model.yaml")
        """
        data = {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "model_type": self.model_type,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_memory_mb": self.max_memory_mb,
            "warmup": self.warmup,
            "config": self.config,
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "model_type": self.model_type,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_memory_mb": self.max_memory_mb,
            "warmup": self.warmup,
            "config": self.config,
        }

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.model_id:
            raise ValueError("model_id cannot be empty")

        if not self.model_path:
            raise ValueError("model_path cannot be empty")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.max_memory_mb is not None and self.max_memory_mb <= 0:
            raise ValueError(f"max_memory_mb must be positive, got {self.max_memory_mb}")

        valid_devices = ["cpu", "cuda", "mps"]
        device_prefix = self.device.split(":")[0]
        if device_prefix not in valid_devices:
            raise ValueError(f"Invalid device: {self.device}")

        return True
