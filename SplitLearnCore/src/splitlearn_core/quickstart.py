"""
Quickstart API for SplitLearnCore

This module provides simplified, high-level APIs for quickly loading
and splitting models, with automatic model downloading and caching.

Example:
    >>> from splitlearn_core.quickstart import load_split_model
    >>> bottom, trunk, top = load_split_model("gpt2", split_points=[2, 10])
"""

import os
from typing import Optional, Tuple, Union, List
from pathlib import Path
import logging

import torch
import torch.nn as nn

from .factory import ModelFactory
from .core import BaseBottomModel, BaseTrunkModel, BaseTopModel

logger = logging.getLogger(__name__)


def load_split_model(
    model_type: str,
    split_points: List[int],
    model_name_or_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    force_download: bool = False,
) -> Tuple[BaseBottomModel, BaseTrunkModel, BaseTopModel]:
    """
    Load and split a model with automatic configuration.

    This function provides a simplified interface to:
    - Load pretrained models (from local cache or HuggingFace)
    - Automatically split into Bottom/Trunk/Top components
    - Handle device placement
    - Manage caching

    Args:
        model_type: Type of model to load (e.g., "gpt2", "qwen2", "gemma")
        split_points: List of two integers [split_point_1, split_point_2]:
            - split_point_1: Number of layers for Bottom model (client-side)
            - split_point_2: Starting layer for Top model (client-side)
            - Layers between split_point_1 and split_point_2 go to Trunk (server-side)
        model_name_or_path: Model name (e.g., "gpt2") or path to local model.
            If None, defaults to model_type.
        cache_dir: Directory to cache downloaded models. If None, uses default:
            - First checks $SPLITLEARN_CACHE_DIR environment variable
            - Then checks local "./models" directory
            - Finally uses HuggingFace's default cache
        device: Device to place models on ("cpu", "cuda", etc.). If None, auto-detects.
        torch_dtype: Data type for model weights (e.g., torch.float16, torch.bfloat16).
            If None, uses model's default.
        force_download: Force redownload even if model exists in cache

    Returns:
        Tuple of (bottom_model, trunk_model, top_model)

    Example:
        Basic usage with auto-configuration:
        >>> bottom, trunk, top = load_split_model("gpt2", split_points=[2, 10])

        With custom cache directory:
        >>> bottom, trunk, top = load_split_model(
        ...     "gpt2",
        ...     split_points=[2, 10],
        ...     cache_dir="./my_models"
        ... )

        With GPU and mixed precision:
        >>> bottom, trunk, top = load_split_model(
        ...     "qwen2",
        ...     split_points=[4, 20],
        ...     model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        ...     device="cuda",
        ...     torch_dtype=torch.bfloat16
        ... )
    """
    # Determine model name or path
    if model_name_or_path is None:
        model_name_or_path = model_type
        logger.info(f"Using default model name: {model_name_or_path}")

    # Determine cache directory (local-first strategy)
    if cache_dir is None:
        # Priority 1: Environment variable
        cache_dir = os.getenv("SPLITLEARN_CACHE_DIR")

        # Priority 2: Local "./models" directory if it exists
        if cache_dir is None:
            local_cache = Path("./models")
            if local_cache.exists() and local_cache.is_dir():
                cache_dir = str(local_cache)
                logger.info(f"Using local cache directory: {cache_dir}")

        # Priority 3: None (will use HuggingFace default)
        if cache_dir is None:
            logger.info("Using HuggingFace default cache directory")

    # Check if model exists locally
    if cache_dir and not force_download:
        model_path = Path(cache_dir) / model_name_or_path
        if model_path.exists():
            model_name_or_path = str(model_path)
            logger.info(f"Found model in local cache: {model_name_or_path}")
        else:
            logger.info(f"Model not found in cache, will download from HuggingFace")

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Auto-detected device: {device}")

    # Validate split points
    if len(split_points) != 2:
        raise ValueError(
            f"split_points must be a list of 2 integers, got {len(split_points)}"
        )

    split_point_1, split_point_2 = split_points

    if split_point_1 >= split_point_2:
        raise ValueError(
            f"split_point_1 ({split_point_1}) must be less than "
            f"split_point_2 ({split_point_2})"
        )

    # Log configuration
    logger.info(f"Loading split model:")
    logger.info(f"  Model type: {model_type}")
    logger.info(f"  Model name/path: {model_name_or_path}")
    logger.info(f"  Split points: Bottom(0-{split_point_1}), "
                f"Trunk({split_point_1}-{split_point_2}), "
                f"Top({split_point_2}-end)")
    logger.info(f"  Device: {device}")
    logger.info(f"  dtype: {torch_dtype or 'default'}")

    # Create split models using ModelFactory
    try:
        bottom, trunk, top = ModelFactory.create_split_models(
            model_type=model_type,
            model_name_or_path=model_name_or_path,
            split_point_1=split_point_1,
            split_point_2=split_point_2,
            device=device,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )

        logger.info("✓ Models loaded successfully")
        logger.info(f"  Bottom: {sum(p.numel() for p in bottom.parameters())/1e6:.2f}M parameters")
        logger.info(f"  Trunk:  {sum(p.numel() for p in trunk.parameters())/1e6:.2f}M parameters")
        logger.info(f"  Top:    {sum(p.numel() for p in top.parameters())/1e6:.2f}M parameters")

        return bottom, trunk, top

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(
            f"Failed to load split model '{model_type}' from '{model_name_or_path}': {e}"
        ) from e


def load_bottom_model(
    model_type: str,
    end_layer: int,
    model_name_or_path: Optional[str] = None,
    **kwargs
) -> BaseBottomModel:
    """
    Load only the bottom (client-side front) model.

    Args:
        model_type: Type of model (e.g., "gpt2")
        end_layer: Number of layers to include
        model_name_or_path: Model name or path
        **kwargs: Additional arguments passed to load_split_model()

    Returns:
        Bottom model

    Example:
        >>> bottom = load_bottom_model("gpt2", end_layer=2)
    """
    # We still need to create full split, but only return bottom
    # This is because the factory needs split points
    split_point_2 = end_layer + 1  # Dummy value
    bottom, _, _ = load_split_model(
        model_type=model_type,
        split_points=[end_layer, split_point_2],
        model_name_or_path=model_name_or_path,
        **kwargs
    )
    return bottom


def load_top_model(
    model_type: str,
    start_layer: int,
    model_name_or_path: Optional[str] = None,
    **kwargs
) -> BaseTopModel:
    """
    Load only the top (client-side back) model.

    Args:
        model_type: Type of model (e.g., "gpt2")
        start_layer: Starting layer index
        model_name_or_path: Model name or path
        **kwargs: Additional arguments passed to load_split_model()

    Returns:
        Top model

    Example:
        >>> top = load_top_model("gpt2", start_layer=10)
    """
    # We still need to create full split, but only return top
    split_point_1 = start_layer - 1  # Dummy value
    _, _, top = load_split_model(
        model_type=model_type,
        split_points=[split_point_1, start_layer],
        model_name_or_path=model_name_or_path,
        **kwargs
    )
    return top


__all__ = [
    "load_split_model",
    "load_bottom_model",
    "load_top_model",
]
