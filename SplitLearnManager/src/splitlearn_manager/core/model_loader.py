"""
Model loading functionality.
"""

import logging
import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from pathlib import Path

from ..config import ModelConfig

logger = logging.getLogger(__name__)


# 全局标志，确保只设置一次
_pytorch_threads_configured = False

def configure_pytorch_threads(single_threaded: bool = True):
    """
    配置 PyTorch 线程数，避免多线程竞争和 mutex 警告。
    
    注意：这个函数必须在任何 PyTorch 并行操作之前调用。
    如果已经调用过或已经启动并行工作，会跳过设置以避免错误。
    
    Args:
        single_threaded: 如果为 True，设置为单线程模式
    """
    global _pytorch_threads_configured
    
    if not single_threaded:
        return
    
    # 如果已经配置过，跳过
    if _pytorch_threads_configured:
        logger.debug("PyTorch threads already configured, skipping")
        return
    
    try:
        # 设置环境变量（必须在导入 torch 之前或最早设置）
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
        
        # 设置 PyTorch 线程数（必须在任何并行工作开始之前）
        # 使用 try-except 避免重复设置错误
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError as e:
            # 如果已经启动过并行工作，记录警告但继续
            logger.warning(f"Could not set PyTorch threads (may have started already): {e}")
            return
        
        _pytorch_threads_configured = True
        logger.debug("PyTorch configured for single-threaded mode")
    except Exception as e:
        logger.warning(f"Failed to configure PyTorch threads: {e}")


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
        # 在加载模型之前，配置 PyTorch 为单线程模式
        # 这样可以避免模型加载过程中的 mutex 警告
        configure_pytorch_threads(single_threaded=True)
        
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

        elif config.model_type in ["gpt2", "qwen2", "gemma"]:
            # 使用 SplitLearnCore 加载分割模型
            model = ModelLoader._load_split_model(config)

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
    def _load_split_model(config: ModelConfig) -> nn.Module:
        """Load split model using SplitLearnCore."""
        try:
            from splitlearn_core.quickstart import load_split_model

            # 从 config 中获取 component 和 split_points
            component = config.config.get("component", "trunk")
            split_points = config.config.get("split_points", [2, 10])
            cache_dir = config.config.get("cache_dir", "./models")

            logger.info(
                f"Loading split model: type={config.model_type}, "
                f"component={component}, split_points={split_points}"
            )

            # 加载分割模型
            bottom, trunk, top = load_split_model(
                model_type=config.model_type,
                split_points=split_points,
                cache_dir=cache_dir
            )

            # 根据 component 返回对应的模型
            if component == "bottom":
                model = bottom
            elif component == "trunk":
                model = trunk
            elif component == "top":
                model = top
            else:
                raise ValueError(f"Invalid component: {component}. Must be 'bottom', 'trunk', or 'top'")

            # 移动到指定设备并设置为评估模式
            model = model.to(config.device)
            model.eval()

            logger.info(f"Successfully loaded {component} model on {config.device}")
            return model

        except ImportError:
            raise ImportError(
                "splitlearn-core library required for split models. "
                "Install with: pip install splitlearn-core"
            )
        except Exception as e:
            logger.error(f"Failed to load split model: {e}")
            raise RuntimeError(f"Split model loading failed: {e}")

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
