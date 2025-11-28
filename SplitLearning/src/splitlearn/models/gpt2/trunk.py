"""
GPT-2 Trunk Model (Server-side Middle Part)

Contains: Middle M transformer blocks
Input: Intermediate hidden states from Bottom
Output: Intermediate hidden states for Top
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from splitlearn.utils.param_mapper import ParamMapper
from splitlearn.core import BaseTrunkModel
from splitlearn.registry import ModelRegistry


@ModelRegistry.register("gpt2", "trunk")
class GPT2TrunkModel(BaseTrunkModel):
    """
    Trunk (middle) part of split GPT-2 model

    Architecture:
        - h: Middle M transformer blocks (h[start_layer:end_layer])

    Args:
        config: GPT2Config object
        start_layer: Starting layer index in original model (default: 2)
        end_layer: Ending layer index in original model (default: 10)
    """

    def __init__(
        self,
        config: GPT2Config,
        start_layer: int = 2,
        end_layer: int = 10,
    ):
        super().__init__(config, start_layer, end_layer)

        # Transformer blocks (indices remapped to 0-based)
        self.h = nn.ModuleList(
            [GPT2Block(config, layer_idx=i) for i in range(self.num_layers)]
        )

        # Fix attention implementation for newer transformers versions
        self._fix_attention_implementation()

        # Initialize weights
        self.apply(self._init_weights)

    # Implement abstract methods

    def get_transformer_blocks(self) -> nn.ModuleList:
        """Return transformer blocks"""
        return self.h

    def prepare_attention_mask(
        self, attention_mask: Optional[torch.Tensor], hidden_states: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Prepare GPT-2 specific 4D attention mask"""
        if attention_mask is None:
            return None

        batch_size = hidden_states.shape[0]

        # Create 4D attention mask from 2D tensor
        attention_mask = attention_mask.view(batch_size, -1)
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=hidden_states.dtype)
        attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        return attention_mask

    def get_layer_name_pattern(self) -> str:
        """Return GPT-2 layer naming pattern"""
        return r"\.h\.[0-9]+"

    @classmethod
    def from_pretrained_split(
        cls,
        full_state_dict: dict,
        config: GPT2Config,
        start_layer: int = 2,
        end_layer: int = 10,
    ):
        """
        Create Trunk model from full GPT-2 state dict

        Args:
            full_state_dict: State dict from full GPT2LMHeadModel
            config: GPT2Config
            start_layer: Starting layer index
            end_layer: Ending layer index

        Returns:
            GPT2TrunkModel instance with loaded weights
        """
        # Create model
        model = cls(config, start_layer=start_layer, end_layer=end_layer)

        # Filter and remap relevant weights
        trunk_dict = ParamMapper.filter_and_remap_state_dict(
            full_state_dict,
            model_type="gpt2",
            include_embedding=False,
            include_lm_head=False,
            include_final_norm=False,
            layer_start=start_layer,
            layer_end=end_layer,
            remap_layers=True,  # Remap h.2 -> h.0, etc.
        )

        # Remove 'transformer.' prefix from keys to match model structure
        trunk_dict_clean = {}
        for key, value in trunk_dict.items():
            if key.startswith('transformer.'):
                clean_key = key.replace('transformer.', '', 1)
                trunk_dict_clean[clean_key] = value
            else:
                trunk_dict_clean[key] = value

        # Load weights with strict=False to catch any mismatches
        missing_keys, unexpected_keys = model.load_state_dict(trunk_dict_clean, strict=False)
        if missing_keys:
            print(f"    Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"    Warning: Unexpected keys: {unexpected_keys}")

        return model
