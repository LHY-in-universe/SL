"""
Gemma Trunk Model (Server-side Middle Part)

Contains: Middle M transformer blocks
Output: Intermediate hidden states
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer

from ...utils.param_mapper import ParamMapper

from ...core import BaseTrunkModel
from ...registry import ModelRegistry


@ModelRegistry.register("gemma", "trunk")
class GemmaTrunkModel(BaseTrunkModel):
    """
    Trunk (middle) part of split Gemma model

    Architecture:
        - layers: Middle M transformer blocks (layers[start_layer:end_layer])

    This part is typically deployed on the server side and receives
    hidden states from the Bottom model via gRPC.

    Args:
        config: GemmaConfig object
        start_layer: Starting layer index in original model
        end_layer: Ending layer index (exclusive)
    """

    def __init__(self, config: GemmaConfig, start_layer: int = 8, end_layer: int = 16):
        super().__init__(config, start_layer, end_layer)

        # Transformer blocks (remapped to start from 0)
        self.layers = nn.ModuleList(
            [
                GemmaDecoderLayer(config, layer_idx=i)
                for i in range(self.num_layers)
            ]
        )

        # Fix attention implementation
        self._fix_attention_implementation()

        # Initialize weights
        self.apply(self._init_weights)

    def get_transformer_blocks(self) -> nn.ModuleList:
        """Return transformer blocks"""
        return self.layers

    def prepare_attention_mask(
        self, attention_mask: Optional[torch.Tensor], hidden_states: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Prepare Gemma specific 4D attention mask
        """
        if attention_mask is None:
            return None

        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Expand attention mask to 4D
        if attention_mask.dim() == 2:
            # Create causal mask
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
                diagonal=1
            )
            # Expand attention_mask
            expanded_mask = attention_mask[:, None, None, :].to(dtype=dtype)
            expanded_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
            attention_mask = expanded_mask + causal_mask.unsqueeze(0)

        return attention_mask

    def get_layer_name_pattern(self) -> str:
        """Return Gemma layer naming pattern"""
        return r"\.layers\.[0-9]+"

    @classmethod
    def from_pretrained_split(
        cls,
        full_state_dict: dict,
        config: GemmaConfig,
        start_layer: int = 8,
        end_layer: int = 16,
    ):
        """
        Create Trunk model from full Gemma state dict

        Args:
            full_state_dict: State dict from full GemmaForCausalLM
            config: GemmaConfig
            start_layer: Starting layer index
            end_layer: Ending layer index (exclusive)

        Returns:
            GemmaTrunkModel instance with loaded weights
        """
        # Create model
        model = cls(config, start_layer=start_layer, end_layer=end_layer)

        # Filter and remap relevant weights using ParamMapper
        trunk_dict = ParamMapper.filter_and_remap_state_dict(
            full_state_dict,
            model_type="gemma",
            include_embedding=False,
            include_lm_head=False,
            include_final_norm=False,
            layer_start=start_layer,
            layer_end=end_layer,
            remap_layers=True,  # Remap to start from 0
        )

        # Remove 'model.' prefix from keys
        trunk_dict_clean = {}
        for key, value in trunk_dict.items():
            if key.startswith('model.'):
                clean_key = key.replace('model.', '', 1)
                trunk_dict_clean[clean_key] = value
            else:
                trunk_dict_clean[key] = value

        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(trunk_dict_clean, strict=False)
        if missing_keys:
            print(f"    Warning: Missing keys in GemmaTrunkModel: {missing_keys}")
        if unexpected_keys:
            print(f"    Warning: Unexpected keys in GemmaTrunkModel: {unexpected_keys}")

        return model

    def _fix_attention_implementation(self):
        """
        Fix _attn_implementation for Gemma transformer blocks.
        Gemma uses self_attn instead of attn.
        """
        blocks = self.get_transformer_blocks()
        for block in blocks:
            if hasattr(block, "self_attn") and hasattr(block.self_attn, "config"):
                # Set if not present or if None
                if not hasattr(block.self_attn.config, "_attn_implementation") or \
                   block.self_attn.config._attn_implementation is None:
                    block.self_attn.config._attn_implementation = "eager"

    def _init_weights(self, module):
        """Initialize weights for Gemma model"""
        std = self.config.initializer_range if hasattr(self.config, 'initializer_range') else 0.02
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

