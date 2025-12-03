"""
SplitLearn - A library for physically splitting large language models.

This library provides tools to split transformer-based models into
Bottom, Trunk, and Top components for distributed inference.

Example:
    >>> from splitlearn_core import ModelFactory
    >>>
    >>> bottom, trunk, top = ModelFactory.create_split_models(
    ...     model_type='gpt2',
    ...     model_name_or_path='gpt2',
    ...     split_point_1=2,
    ...     split_point_2=10,
    ... )
"""

from .__version__ import __version__

# Core classes
from .core import (
    BaseSplitModel,
    BaseBottomModel,
    BaseTrunkModel,
    BaseTopModel,
)

# Factory and Registry
from .factory import ModelFactory
from .registry import ModelRegistry

# Utilities
from .utils import ParamMapper, StorageManager

# Import models to trigger registration
from . import models

# Model implementations
from .models.gpt2 import GPT2BottomModel, GPT2TrunkModel, GPT2TopModel
# from .models.qwen2 import Qwen2BottomModel, Qwen2TrunkModel, Qwen2TopModel  # Requires transformers >= 4.37

__all__ = [
    # Version
    "__version__",

    # Core
    "BaseSplitModel",
    "BaseBottomModel",
    "BaseTrunkModel",
    "BaseTopModel",

    # Factory & Registry
    "ModelFactory",
    "ModelRegistry",

    # Utils
    "ParamMapper",
    "StorageManager",

    # GPT-2
    "GPT2BottomModel",
    "GPT2TrunkModel",
    "GPT2TopModel",

    # Qwen2 (requires transformers >= 4.37)
    # "Qwen2BottomModel",
    # "Qwen2TrunkModel",
    # "Qwen2TopModel",
]
