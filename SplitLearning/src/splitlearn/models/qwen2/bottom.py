"""
Qwen2 Bottom Model (Client-side Front Part)

Contains: Embedding layer + First N transformer blocks
Output: Intermediate hidden states
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from splitlearn.utils.param_mapper import ParamMapper

from splitlearn.core import BaseBottomModel
from splitlearn.registry import ModelRegistry


@ModelRegistry.register("qwen2", "bottom")
class Qwen2BottomModel(BaseBottomModel):
    """
    Bottom part of split Qwen2 model

    Architecture:
        - embed_tokens: Token embedding
        - layers: First N transformer blocks (layers[0:end_layer])

    Note: Qwen2 uses RoPE (Rotary Position Embedding) which is applied
    inside each transformer block, so no separate position embedding is needed.

    Args:
        config: Qwen2Config object
        end_layer: Number of transformer layers to include (default: 8)
    """

    def __init__(self, config: Qwen2Config, end_layer: int = 8):
        super().__init__(config, end_layer=end_layer)

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx=i) for i in range(end_layer)]
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
        For Qwen2, RoPE is applied inside each attention layer,
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
        Prepare Qwen2 specific 4D attention mask

        Qwen2 uses a similar attention mask format as LLaMA2.
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
        """Return Qwen2 layer naming pattern"""
        return r"\.layers\.[0-9]+"

    @classmethod
    def from_pretrained_split(
        cls,
        full_state_dict: dict,
        config: Qwen2Config,
        end_layer: int = 8,
    ):
        """
        Create Bottom model from full Qwen2 state dict

        Args:
            full_state_dict: State dict from full Qwen2ForCausalLM
            config: Qwen2Config
            end_layer: Number of layers to include

        Returns:
            Qwen2BottomModel instance with loaded weights
        """
        # Create model
        model = cls(config, end_layer=end_layer)

        # Filter relevant weights using ParamMapper
        bottom_dict = ParamMapper.filter_and_remap_state_dict(
            full_state_dict,
            model_type="qwen2",
            include_embedding=True,
            include_lm_head=False,
            include_final_norm=False,
            layer_start=0,
            layer_end=end_layer,
            remap_layers=False,
        )

        # Remove 'model.' prefix from keys to match model structure
        # Qwen2ForCausalLM uses 'model.embed_tokens.weight' but our model expects 'embed_tokens.weight'
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
            print(f"    Warning: Missing keys in Qwen2BottomModel: {missing_keys}")
        if unexpected_keys:
            print(f"    Warning: Unexpected keys in Qwen2BottomModel: {unexpected_keys}")

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
        """
        Initialize weights for Qwen2 model

        Qwen2 uses different initialization than GPT-2
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
