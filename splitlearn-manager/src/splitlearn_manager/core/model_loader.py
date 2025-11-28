"""
Model loading functionality.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from pathlib import Path

from ..config import ModelConfig

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Handles model loading from various sources.

    Supports loading from:
    - PyTorch checkpoints (.pt, .pth)
    - Hugging Face models
    - Custom model implementations
    """

    @staticmethod
    def load_pytorch_checkpoint(
        model_path: str,
        device: str = "cpu",
        map_location: Optional[str] = None
    ) -> nn.Module:
        """
        Load model from PyTorch checkpoint.

        Args:
            model_path: Path to .pt or .pth file
            device: Device to load model on
            map_location: Override device for loading

        Returns:
            Loaded PyTorch model

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If loading fails
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loading PyTorch checkpoint from {model_path}")

        try:
            # Load checkpoint
            # Use weights_only=False for compatibility with PyTorch 2.6+
            checkpoint = torch.load(
                model_path,
                map_location=map_location or device,
                weights_only=False
            )

            # Extract model if checkpoint contains additional info
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model = checkpoint['model']
                elif 'model_state_dict' in checkpoint:
                    # Need to reconstruct model from state dict
                    raise ValueError(
                        "Checkpoint contains state_dict but no model. "
                        "Please provide model architecture separately."
                    )
                else:
                    # Assume entire checkpoint is the model
                    model = checkpoint
            else:
                model = checkpoint

            # Move to device and set to eval mode
            model = model.to(device)
            model.eval()

            logger.info(f"Successfully loaded model on {device}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    @staticmethod
    def load_from_config(config: ModelConfig) -> nn.Module:
        """
        Load model based on configuration.

        Args:
            config: Model configuration

        Returns:
            Loaded model

        Raises:
            ValueError: If model_type is unsupported
        """
        logger.info(f"Loading model {config.model_id} of type {config.model_type}")

        if config.model_type == "pytorch":
            model = ModelLoader.load_pytorch_checkpoint(
                config.model_path,
                config.device
            )

        elif config.model_type == "huggingface":
            model = ModelLoader._load_huggingface_model(config)

        elif config.model_type == "custom":
            model = ModelLoader._load_custom_model(config)

        else:
            raise ValueError(f"Unsupported model_type: {config.model_type}")

        # Warmup if requested
        if config.warmup:
            ModelLoader._warmup_model(model, config)

        return model

    @staticmethod
    def _load_huggingface_model(config: ModelConfig) -> nn.Module:
        """Load Hugging Face model."""
        try:
            from transformers import AutoModel

            logger.info(f"Loading Hugging Face model from {config.model_path}")
            model = AutoModel.from_pretrained(config.model_path)
            model = model.to(config.device)
            model.eval()

            return model

        except ImportError:
            raise ImportError(
                "transformers library required for Hugging Face models. "
                "Install with: pip install transformers"
            )

    @staticmethod
    def _load_custom_model(config: ModelConfig) -> nn.Module:
        """Load custom model."""
        # For custom models, users should provide their own loading logic
        # This is a placeholder that loads from checkpoint
        return ModelLoader.load_pytorch_checkpoint(
            config.model_path,
            config.device
        )

    @staticmethod
    def _warmup_model(model: nn.Module, config: ModelConfig) -> None:
        """
        Warmup model with dummy inputs.

        This helps initialize CUDA kernels and optimize performance.
        """
        logger.info("Warming up model...")

        try:
            # Create dummy input based on config
            dummy_shape = config.config.get("input_shape", (1, 10, 768))
            dummy_input = torch.randn(*dummy_shape, device=config.device)

            # Run a few forward passes
            with torch.no_grad():
                for _ in range(3):
                    _ = model(dummy_input)

            logger.info("Model warmup completed")

        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    @staticmethod
    def get_model_info(model: nn.Module, device: str) -> Dict[str, Any]:
        """
        Get information about loaded model.

        Args:
            model: PyTorch model
            device: Device model is on

        Returns:
            Dictionary with model information
        """
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        info = {
            "model_class": model.__class__.__name__,
            "num_parameters": num_params,
            "trainable_parameters": trainable_params,
            "device": device,
            "training_mode": model.training,
        }

        # Add memory info if on CUDA
        if device.startswith("cuda"):
            try:
                memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
                memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)

                info.update({
                    "memory_allocated_mb": round(memory_allocated, 2),
                    "memory_reserved_mb": round(memory_reserved, 2),
                })
            except Exception as e:
                logger.warning(f"Failed to get CUDA memory info: {e}")

        return info
