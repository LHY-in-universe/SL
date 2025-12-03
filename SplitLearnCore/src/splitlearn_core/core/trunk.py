"""
Base abstract class for Trunk models

Trunk models handle: Middle M transformer layers (server-side)
"""
from typing import Optional
import torch
import torch.nn as nn

from .base import BaseSplitModel


class BaseTrunkModel(BaseSplitModel):
    """
    Abstract base class for Trunk split models

    Trunk models are responsible for:
    1. Receiving intermediate hidden states from Bottom
    2. Processing through middle M transformer layers
    3. Outputting hidden states for Top

    Trunk models are typically deployed on the server side.
    """

    def __init__(self, config, start_layer: int, end_layer: int):
        """
        Initialize Trunk model

        Args:
            config: Model configuration
            start_layer: Starting layer index in original model
            end_layer: Ending layer index (exclusive)
        """
        super().__init__(config, start_layer, end_layer)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple] = None,
        use_cache: Optional[bool] = None,
    ):
        """
        Forward pass through Trunk model

        Args:
            hidden_states: Input from Bottom [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            past_key_values: Tuple of past key value states
            use_cache: Whether to return the key/value attentions

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]: 
                - Output hidden states [batch_size, seq_len, hidden_size]
                - (Optional) Present key/value states if use_cache=True
        """
        # Prepare attention mask
        attention_mask = self.prepare_attention_mask(attention_mask, hidden_states)

        # Pass through transformer blocks
        presents = () if use_cache else None
        
        for i, block in enumerate(self.get_transformer_blocks()):
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                layer_past=layer_past,
                use_cache=use_cache,
            )
            hidden_states = outputs[0]
            if use_cache:
                presents = presents + (outputs[1],)

        if use_cache:
            return hidden_states, presents
        return hidden_states
