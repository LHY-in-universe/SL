"""
Base abstract classes for split models

This module provides the foundational abstract classes that all model-specific
implementations should inherit from.
"""

from typing import Dict, Optional
import torch

from .base import BaseSplitModel
from .bottom import BaseBottomModel
from .trunk import BaseTrunkModel
from .top import BaseTopModel


# Backward compatibility functions for legacy tests
def get_block_num(param_name: str) -> int:
    """
    Extract block/layer number from GPT-2 parameter name (legacy function).

    This is a backward compatibility wrapper. New code should use
    ParamMapper.get_layer_number() instead.

    Args:
        param_name: Parameter name (e.g., 'transformer.h.5.attn.weight')

    Returns:
        Layer index, or -1 if not a layer parameter
    """
    import re
    match = re.search(r'\.h\.([0-9]+)', param_name)
    return int(match.group(1)) if match else -1


def filter_state_dict(
    state_dict: Dict[str, torch.Tensor],
    include_embedding: bool = False,
    include_lm_head: bool = False,
    include_ln_f: bool = False,
    layer_start: Optional[int] = None,
    layer_end: Optional[int] = None,
    remap_layers: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Filter state dict for GPT-2 model splitting (legacy function).

    This is a backward compatibility wrapper. New code should use
    ParamMapper.filter_and_remap_state_dict() instead.

    Args:
        state_dict: Source state dict
        include_embedding: Include embedding layers (wte, wpe)
        include_lm_head: Include LM head
        include_ln_f: Include final layer norm
        layer_start: Start layer index (inclusive)
        layer_end: End layer index (exclusive)
        remap_layers: If True, remap layer indices to start from 0

    Returns:
        Filtered state dict
    """
    # Import here to avoid circular dependency
    from splitlearn.utils.param_mapper import ParamMapper

    return ParamMapper.filter_and_remap_state_dict(
        state_dict,
        model_type='gpt2',
        include_embedding=include_embedding,
        include_lm_head=include_lm_head,
        include_final_norm=include_ln_f,
        layer_start=layer_start,
        layer_end=layer_end,
        remap_layers=remap_layers,
    )


__all__ = [
    'BaseSplitModel',
    'BaseBottomModel',
    'BaseTrunkModel',
    'BaseTopModel',
    'get_block_num',
    'filter_state_dict',
]
