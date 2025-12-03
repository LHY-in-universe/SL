"""
Gemma Bottom Model (Client-side Front Part)

Contains: Embedding layer + First N transformer blocks
Output: Intermediate hidden states
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer

from ...utils.param_mapper import ParamMapper

from ...core import BaseBottomModel
from ...registry import ModelRegistry


@ModelRegistry.register("gemma", "bottom")
class GemmaBottomModel(BaseBottomModel):
    """
    Bottom part of split Gemma model

    Architecture:
        - embed_tokens: Token embedding
        - layers: First N transformer blocks (layers[0:end_layer])

    Note: Gemma uses RoPE (Rotary Position Embedding) which is applied
    inside each transformer block, so no separate position embedding is needed.

    Args:
        config: GemmaConfig object
        end_layer: Number of transformer layers to include (default: 8)
    """

    def __init__(self, config: GemmaConfig, end_layer: int = 8):
        super().__init__(config, end_layer=end_layer)

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx=i) for i in range(end_layer)]
        )

        # Fix attention implementation for newer transformers versions
        self._fix_attention_implementation()

        # Initialize weights
        self.apply(self._init_weights)

    # Implement abstract methods from BaseBottomModel

    def get_embeddings(self) -> nn.Module:
        """Return token embedding layer"""
        return self.embed_tokens

    def apply_position_encoding(
        self, inputs_embeds: torch.Tensor, position_ids: Optional[torch.LongTensor]
    ) -> torch.Tensor:
        """
        For Gemma, RoPE is applied inside each attention layer,
        so we just return the token embeddings as-is.
        """
        return inputs_embeds

    def get_transformer_blocks(self) -> nn.ModuleList:
        """Return transformer blocks"""
        return self.layers

    def prepare_attention_mask(
        self, attention_mask: Optional[torch.Tensor], hidden_states: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Prepare Gemma specific 4D attention mask

        Gemma uses a similar attention mask format as LLaMA2/Qwen2.
        """
        if attention_mask is None:
            return None

        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Expand attention mask to 4D: [batch_size, 1, seq_len, seq_len]
        if attention_mask.dim() == 2:
            # Create causal mask
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
                diagonal=1
            )
            # Expand attention_mask from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
            expanded_mask = attention_mask[:, None, None, :].to(dtype=dtype)
            # Combine with causal mask
            # Where attention_mask is 0, set to -inf
            expanded_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
            # Add causal mask
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
        end_layer: int = 8,
    ):
        """
        Create Bottom model from full Gemma state dict

        Args:
            full_state_dict: State dict from full GemmaForCausalLM
            config: GemmaConfig
            end_layer: Number of layers to include

        Returns:
            GemmaBottomModel instance with loaded weights
        """
        # Create model
        model = cls(config, end_layer=end_layer)

        # Filter relevant weights using ParamMapper
        bottom_dict = ParamMapper.filter_and_remap_state_dict(
            full_state_dict,
            model_type="gemma",
            include_embedding=True,
            include_lm_head=False,
            include_final_norm=False,
            layer_start=0,
            layer_end=end_layer,
            remap_layers=False,
        )

        # Remove 'model.' prefix from keys to match model structure
        # GemmaForCausalLM uses 'model.embed_tokens.weight' but our model expects 'embed_tokens.weight'
        bottom_dict_clean = {}
        for key, value in bottom_dict.items():
            if key.startswith('model.'):
                clean_key = key.replace('model.', '', 1)
                bottom_dict_clean[clean_key] = value
            else:
                bottom_dict_clean[key] = value

        # Load weights with strict=False to allow for missing/unexpected keys
        missing_keys, unexpected_keys = model.load_state_dict(bottom_dict_clean, strict=False)
        if missing_keys:
            print(f"    Warning: Missing keys in GemmaBottomModel: {missing_keys}")
        if unexpected_keys:
            print(f"    Warning: Unexpected keys in GemmaBottomModel: {unexpected_keys}")

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
        """
        Initialize weights for Gemma model

        Gemma uses similar initialization to Qwen2/LLaMA
        """
        std = self.config.initializer_range if hasattr(self.config, 'initializer_range') else 0.02
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

