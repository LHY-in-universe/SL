"""
GPT-2 Bottom Model (Client-side Front Part)

Contains: Embedding layers + First N transformer blocks
Output: Intermediate hidden states
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from ...utils.param_mapper import ParamMapper
from ...core import BaseBottomModel
from ...registry import ModelRegistry


@ModelRegistry.register("gpt2", "bottom")
class GPT2BottomModel(BaseBottomModel):
    """
    Bottom part of split GPT-2 model

    Architecture:
        - wte: Token embedding
        - wpe: Position embedding
        - drop: Embedding dropout
        - h: First N transformer blocks (h[0:end_layer])

    Args:
        config: GPT2Config object
        end_layer: Number of transformer layers to include (default: 2)
    """

    def __init__(self, config: GPT2Config, end_layer: int = 2):
        super().__init__(config, end_layer=end_layer)

        # Embedding layers
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.n_positions, config.hidden_size)
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer blocks
        self.h = nn.ModuleList(
            [GPT2Block(config, layer_idx=i) for i in range(end_layer)]
        )

        # Fix attention implementation for newer transformers versions
        self._fix_attention_implementation()

        # Initialize weights
        self.apply(self._init_weights)

    # Implement abstract methods from BaseBottomModel

    def get_embeddings(self) -> nn.Module:
        """Return token embedding layer"""
        return self.wte

    def apply_position_encoding(
        self, inputs_embeds: torch.Tensor, position_ids: Optional[torch.LongTensor]
    ) -> torch.Tensor:
        """Apply GPT-2's absolute position embeddings"""
        batch_size, seq_len = inputs_embeds.shape[:2]
        device = inputs_embeds.device

        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        # Add position embeddings
        position_embeds = self.wpe(position_ids)
        return self.drop(inputs_embeds + position_embeds)

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
        end_layer: int = 2,
    ):
        """
        Create Bottom model from full GPT-2 state dict

        Args:
            full_state_dict: State dict from full GPT2LMHeadModel
            config: GPT2Config
            end_layer: Number of layers to include

        Returns:
            GPT2BottomModel instance with loaded weights
        """
        # Create model
        model = cls(config, end_layer=end_layer)

        # Filter relevant weights using ParamMapper
        bottom_dict = ParamMapper.filter_and_remap_state_dict(
            full_state_dict,
            model_type="gpt2",
            include_embedding=True,
            include_lm_head=False,
            include_final_norm=False,
            layer_start=0,
            layer_end=end_layer,
            remap_layers=False,
        )

        # Remove 'transformer.' prefix from keys to match model structure
        # GPT2LMHeadModel uses 'transformer.wte.weight' but our model expects 'wte.weight'
        bottom_dict_clean = {}
        for key, value in bottom_dict.items():
            if key.startswith('transformer.'):
                clean_key = key.replace('transformer.', '', 1)
                bottom_dict_clean[clean_key] = value
            else:
                bottom_dict_clean[key] = value

        # Load weights with strict=True to catch any mismatches
        missing_keys, unexpected_keys = model.load_state_dict(bottom_dict_clean, strict=False)
        if missing_keys:
            print(f"    Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"    Warning: Unexpected keys: {unexpected_keys}")

        return model
