"""
Model implementations for different architectures.
"""

# Import model implementations to trigger registration
from . import gpt2
from . import gemma
from . import qwen2

__all__ = ['gpt2', 'gemma', 'qwen2']
