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
from transformers import AutoTokenizer, AutoModelForCausalLM

from .factory import ModelFactory, _configure_pytorch_threads_for_loading
from .core import BaseBottomModel, BaseTrunkModel, BaseTopModel

logger = logging.getLogger(__name__)


def _resolve_cache_dir(cache_dir: Optional[str]) -> Optional[str]:
    """Resolve cache directory with local-first strategy."""
    if cache_dir is not None:
        return cache_dir

    cache_dir = os.getenv("SPLITLEARN_CACHE_DIR")
    if cache_dir:
        return cache_dir

    local_cache = Path("./models")
    if local_cache.exists() and local_cache.is_dir():
        return str(local_cache)

    return None


def load_full_model(
    model_name_or_path: str = "gpt2",
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    dtype: Optional[torch.dtype] = None,
    low_cpu_mem_usage: bool = True,
    force_download: bool = False,
):
    """
    Download and load a full Hugging Face causal LM model with safe defaults.

    Args:
        model_name_or_path: Hugging Face model name or local path (default: "gpt2")
        cache_dir: Cache directory. Priority: argument > $SPLITLEARN_CACHE_DIR > ./models > HF default
        device: Target device ("cpu", "cuda", "mps"). Auto-detect if None.
        torch_dtype/dtype: Torch dtype for weights (e.g., torch.float16, torch.bfloat16, torch.float32)
        low_cpu_mem_usage: Use HF low-memory loading to reduce peak usage
        force_download: Force re-download even if cache exists

    Returns:
        (model, tokenizer)
    """
    # Resolve cache dir
    cache_dir = _resolve_cache_dir(cache_dir)

    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Determine dtype (support alias "dtype")
    dtype_to_use = dtype or torch_dtype

    # If a local cached copy exists, prefer it
    if cache_dir and not force_download:
        candidate = Path(cache_dir) / model_name_or_path
        if candidate.exists():
            model_name_or_path = str(candidate)
            logger.info(f"Using cached model at: {model_name_or_path}")

    logger.info("Loading full Hugging Face causal LM:")
    logger.info(f"  model: {model_name_or_path}")
    logger.info(f"  device: {device}")
    logger.info(f"  dtype: {dtype_to_use or 'default'}")
    logger.info(f"  cache_dir: {cache_dir or 'hf-default'}")
    logger.info(f"  low_cpu_mem_usage: {low_cpu_mem_usage}")

    # Configure torch threads before loading to avoid contention
    _configure_pytorch_threads_for_loading()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
        )

        # 使用 dtype 参数（新版本推荐），兼容 torch_dtype
        model_kwargs = {
            "cache_dir": cache_dir,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "force_download": force_download,
        }
        if dtype_to_use is not None:
            model_kwargs["dtype"] = dtype_to_use
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs
        )

        # Ensure pad_token exists for safe generation
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

        model.eval()
        model.to(device)

        logger.info("✓ Full model loaded successfully")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load full model '{model_name_or_path}': {e}")
        raise RuntimeError(
            f"Failed to load full model '{model_name_or_path}': {e}"
        ) from e


def load_split_model(
    model_type: str,
    split_points: List[int],
    model_name_or_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    force_download: bool = False,
    parts: Optional[List[str]] = None,
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
        parts: Optional subset of {"bottom","trunk","top"}，若提供则只加载对应部分，
               其余返回 None（默认全加载）。

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

    # 对 qwen2_vl / qwen3_vl 允许 split_point_1 为 0（仅视觉塔在 bottom）
    if model_type in ["qwen2_vl", "qwen3_vl"]:
        if not (0 <= split_point_1 < split_point_2):
            raise ValueError(
                f"Invalid split points for {model_type}: "
                f"require 0 <= split_point_1 < split_point_2, got {split_points}"
            )
    else:
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

    # 在加载模型之前配置 PyTorch 线程数（在调用 ModelFactory 之前）
    # 这确保 HuggingFace 的 from_pretrained() 使用单线程
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # 如果已经设置过，忽略错误

    # Create split models using ModelFactory
    try:
        bottom, trunk, top = ModelFactory.create_split_models(
            model_type=model_type,
            model_name_or_path=model_name_or_path,
            split_point_1=split_point_1,
            split_point_2=split_point_2,
            device=device,
            torch_dtype=torch_dtype,
            parts=parts,
        )

        logger.info("✓ Models loaded successfully")
        if bottom is not None:
            logger.info(f"  Bottom: {sum(p.numel() for p in bottom.parameters())/1e6:.2f}M parameters")
        if trunk is not None:
            logger.info(f"  Trunk:  {sum(p.numel() for p in trunk.parameters())/1e6:.2f}M parameters")
        if top is not None:
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
    "load_full_model",
    "load_split_model",
    "load_bottom_model",
    "load_top_model",
]
