"""
Parameter Mapper - Handle parameter naming differences across model architectures

Different models use different naming conventions for layers and components.
This module provides utilities to extract layer numbers, remap indices, and
filter parameters in a model-agnostic way.
"""
import re
from typing import Dict, Any, Optional, List
import torch


class ParamMapper:
    """
    Utilities for mapping parameters across different model architectures

    Handles differences in parameter naming conventions between models like:
    - GPT-2: transformer.h.0.attn.c_attn.weight
    - LLaMA2: model.layers.0.self_attn.q_proj.weight
    - Qwen2: model.layers.0.self_attn.q_proj.weight
    """

    # Regex patterns for different model architectures
    LAYER_PATTERNS = {
        'gpt2': r'\.h\.([0-9]+)',           # transformer.h.5.attn
        'llama2': r'\.layers\.([0-9]+)',    # model.layers.5.self_attn
        'llama': r'\.layers\.([0-9]+)',     # Same as llama2
        'gpt-j': r'\.h\.([0-9]+)',          # Similar to GPT-2
        'qwen2': r'\.layers\.([0-9]+)',     # model.layers.5.self_attn (same as llama)
        'gemma': r'\.layers\.([0-9]+)',     # model.layers.5.self_attn (same as llama/qwen2)
    }

    # Component name patterns
    COMPONENT_PATTERNS = {
        'gpt2': {
            'embedding': r'transformer\.(wte|wpe)\.weight',
            'final_norm': r'transformer\.ln_f',
            'lm_head': r'lm_head\.weight',
        },
        'llama2': {
            'embedding': r'model\.embed_tokens\.weight',
            'final_norm': r'model\.norm',
            'lm_head': r'lm_head\.weight',
        },
        'llama': {
            'embedding': r'model\.embed_tokens\.weight',
            'final_norm': r'model\.norm',
            'lm_head': r'lm_head\.weight',
        },
        'qwen2': {
            'embedding': r'model\.embed_tokens\.weight',
            'final_norm': r'model\.norm',
            'lm_head': r'lm_head\.weight',
        },
        'gemma': {
            'embedding': r'model\.embed_tokens\.weight',
            'final_norm': r'model\.norm',
            'lm_head': r'lm_head\.weight',
        },
    }

    @classmethod
    def get_layer_number(cls, param_name: str, model_type: str) -> int:
        """
        Extract layer number from parameter name

        Args:
            param_name: Full parameter name
            model_type: Model architecture type

        Returns:
            int: Layer index, or -1 if not a layer parameter

        Example:
            >>> ParamMapper.get_layer_number('transformer.h.5.attn.weight', 'gpt2')
            5
            >>> ParamMapper.get_layer_number('model.layers.10.mlp.weight', 'llama2')
            10
            >>> ParamMapper.get_layer_number('lm_head.weight', 'gpt2')
            -1
        """
        if model_type not in cls.LAYER_PATTERNS:
            available = ', '.join(sorted(cls.LAYER_PATTERNS.keys()))
            raise ValueError(
                f"Unknown model_type: '{model_type}'. "
                f"Available types: {available}"
            )

        pattern = cls.LAYER_PATTERNS[model_type]
        match = re.search(pattern, param_name)

        return int(match.group(1)) if match else -1

    @classmethod
    def remap_layer_index(
        cls,
        param_name: str,
        old_idx: int,
        new_idx: int,
        model_type: str
    ) -> str:
        """
        Remap layer index in parameter name

        Args:
            param_name: Original parameter name
            old_idx: Old layer index
            new_idx: New layer index
            model_type: Model architecture type

        Returns:
            str: Parameter name with remapped index

        Example:
            >>> ParamMapper.remap_layer_index(
            ...     'transformer.h.5.attn.weight', 5, 0, 'gpt2'
            ... )
            'transformer.h.0.attn.weight'
        """
        if model_type == 'gpt2' or model_type == 'gpt-j':
            return param_name.replace(f'.h.{old_idx}.', f'.h.{new_idx}.')
        elif model_type in ['llama2', 'llama', 'qwen2', 'gemma']:
            return param_name.replace(f'.layers.{old_idx}.', f'.layers.{new_idx}.')
        else:
            available = ', '.join(sorted(cls.LAYER_PATTERNS.keys()))
            raise ValueError(
                f"Unknown model_type: '{model_type}'. "
                f"Available types: {available}"
            )

    @classmethod
    def is_embedding(cls, param_name: str, model_type: str) -> bool:
        """Check if parameter is an embedding layer"""
        if model_type not in cls.COMPONENT_PATTERNS:
            return False

        pattern = cls.COMPONENT_PATTERNS[model_type].get('embedding', '')
        return bool(re.search(pattern, param_name))

    @classmethod
    def is_final_norm(cls, param_name: str, model_type: str) -> bool:
        """Check if parameter is the final normalization layer"""
        if model_type not in cls.COMPONENT_PATTERNS:
            return False

        pattern = cls.COMPONENT_PATTERNS[model_type].get('final_norm', '')
        return bool(re.search(pattern, param_name))

    @classmethod
    def is_lm_head(cls, param_name: str, model_type: str) -> bool:
        """Check if parameter is the LM head"""
        if model_type not in cls.COMPONENT_PATTERNS:
            return False

        pattern = cls.COMPONENT_PATTERNS[model_type].get('lm_head', '')
        return bool(re.search(pattern, param_name))

    @classmethod
    def filter_and_remap_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        model_type: str,
        include_embedding: bool = False,
        include_lm_head: bool = False,
        include_final_norm: bool = False,
        layer_start: Optional[int] = None,
        layer_end: Optional[int] = None,
        remap_layers: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Filter and optionally remap layer indices in state dict

        Args:
            state_dict: Source state dict
            model_type: Model architecture type
            include_embedding: Include embedding layers
            include_lm_head: Include LM head
            include_final_norm: Include final normalization
            layer_start: Start layer index (inclusive)
            layer_end: End layer index (exclusive)
            remap_layers: If True, remap layer indices to start from 0

        Returns:
            Dict[str, torch.Tensor]: Filtered and remapped state dict

        Example:
            # Get trunk layers 2-10, remapped to 0-7
            trunk_dict = ParamMapper.filter_and_remap_state_dict(
                full_state_dict,
                model_type='gpt2',
                layer_start=2,
                layer_end=10,
                remap_layers=True
            )
        """
        filtered = {}

        for key, value in state_dict.items():
            # Check special components
            if cls.is_embedding(key, model_type):
                if include_embedding:
                    filtered[key] = value
                continue

            if cls.is_lm_head(key, model_type):
                if include_lm_head:
                    filtered[key] = value
                continue

            if cls.is_final_norm(key, model_type):
                if include_final_norm:
                    filtered[key] = value
                continue

            # Check layer parameters
            layer_num = cls.get_layer_number(key, model_type)

            if layer_num == -1:
                # Other parameters (dropout, etc.)
                filtered[key] = value
                continue

            # Apply layer range filter
            if layer_start is not None and layer_num < layer_start:
                continue
            if layer_end is not None and layer_num >= layer_end:
                continue

            # Remap if requested
            if remap_layers and layer_start is not None:
                new_key = cls.remap_layer_index(
                    key, layer_num, layer_num - layer_start, model_type
                )
                filtered[new_key] = value
            else:
                filtered[key] = value

        return filtered

    @classmethod
    def count_parameters(cls, state_dict: Dict[str, torch.Tensor]) -> int:
        """Count total parameters in state dict"""
        return sum(p.numel() for p in state_dict.values())

    @classmethod
    def memory_footprint_mb(cls, state_dict: Dict[str, torch.Tensor]) -> float:
        """Calculate memory footprint in MB (assuming float32)"""
        num_params = cls.count_parameters(state_dict)
        return num_params * 4 / (1024 ** 2)

    @classmethod
    def get_layer_statistics(
        cls,
        state_dict: Dict[str, torch.Tensor],
        model_type: str
    ) -> Dict[int, Dict[str, Any]]:
        """
        Get statistics for each layer

        Returns:
            Dict mapping layer_idx to {num_params, memory_mb, param_names}
        """
        stats = {}

        for key, value in state_dict.items():
            layer_num = cls.get_layer_number(key, model_type)

            if layer_num == -1:
                continue

            if layer_num not in stats:
                stats[layer_num] = {
                    'num_params': 0,
                    'param_names': [],
                }

            stats[layer_num]['num_params'] += value.numel()
            stats[layer_num]['param_names'].append(key)

        # Add memory footprint
        for layer_idx in stats:
            stats[layer_idx]['memory_mb'] = stats[layer_idx]['num_params'] * 4 / (1024 ** 2)

        return stats
