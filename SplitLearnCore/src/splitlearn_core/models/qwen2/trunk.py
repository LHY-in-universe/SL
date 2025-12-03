"""
Qwen2 Trunk Model (Server-side Middle Part)

Contains: Middle M transformer blocks
Output: Intermediate hidden states
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from ...utils.param_mapper import ParamMapper

from ...core import BaseTrunkModel
from ...registry import ModelRegistry


@ModelRegistry.register("qwen2", "trunk")
class Qwen2TrunkModel(BaseTrunkModel):
    """
    Trunk (middle) part of split Qwen2 model

    Architecture:
        - layers: Middle M transformer blocks (layers[start_layer:end_layer])

    This part is typically deployed on the server side and receives
    hidden states from the Bottom model via gRPC.

    Args:
        config: Qwen2Config object
        start_layer: Starting layer index in original model
        end_layer: Ending layer index (exclusive)
    """

    def __init__(self, config: Qwen2Config, start_layer: int = 8, end_layer: int = 24):
        super().__init__(config, start_layer, end_layer)

        # Transformer blocks (remapped to start from 0)
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(config, layer_idx=i)
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
        Prepare Qwen2 specific 4D attention mask
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
        """Return Qwen2 layer naming pattern"""
        return r"\.layers\.[0-9]+"

    @classmethod
    def from_pretrained_split(
        cls,
        full_state_dict: dict,
        config: Qwen2Config,
        start_layer: int = 8,
        end_layer: int = 24,
    ):
        """
        Create Trunk model from full Qwen2 state dict

        Args:
            full_state_dict: State dict from full Qwen2ForCausalLM
            config: Qwen2Config
            start_layer: Starting layer index
            end_layer: Ending layer index (exclusive)

        Returns:
            Qwen2TrunkModel instance with loaded weights
        """
        # Create model
        model = cls(config, start_layer=start_layer, end_layer=end_layer)

        # Filter and remap relevant weights using ParamMapper
        trunk_dict = ParamMapper.filter_and_remap_state_dict(
            full_state_dict,
            model_type="qwen2",
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
            print(f"    Warning: Missing keys in Qwen2TrunkModel: {missing_keys}")
        if unexpected_keys:
            print(f"    Warning: Unexpected keys in Qwen2TrunkModel: {unexpected_keys}")

        return model

    def _fix_attention_implementation(self):
        """
        Fix _attn_implementation for Qwen2 transformer blocks.
        Qwen2 uses self_attn instead of attn.
        """
        blocks = self.get_transformer_blocks()
        for block in blocks:
            if hasattr(block, "self_attn") and hasattr(block.self_attn, "config"):
                # Set if not present or if None
                if not hasattr(block.self_attn.config, "_attn_implementation") or \
                   block.self_attn.config._attn_implementation is None:
                    block.self_attn.config._attn_implementation = "eager"

    def _init_weights(self, module):
        """Initialize weights for Qwen2 model"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
