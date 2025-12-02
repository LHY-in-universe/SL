"""
Base abstract class for Bottom models

Bottom models handle: Embedding + First N transformer layers
"""
from abc import abstractmethod
from typing import Optional
import torch
import torch.nn as nn

from .base import BaseSplitModel


class BaseBottomModel(BaseSplitModel):
    """
    Abstract base class for Bottom split models

    Bottom models are responsible for:
    1. Token and position embeddings
    2. First N transformer layers
    3. Outputting intermediate hidden states for Trunk

    The main architectural differences between models (GPT-2 vs LLaMA2) are:
    - Embedding strategy (absolute vs RoPE)
    - Position encoding application
    """

    def __init__(self, config, end_layer: int):
        """
        Initialize Bottom model

        Args:
            config: Model configuration
            end_layer: Number of layers to include (from layer 0)
        """
        super().__init__(config, start_layer=0, end_layer=end_layer)
        self.end_layer = end_layer

    @abstractmethod
    def get_embeddings(self) -> nn.Module:
        """
        Get the token embedding layer

        Returns:
            nn.Module: Token embedding layer

        Example:
            GPT-2: return self.wte
            LLaMA2: return self.embed_tokens
        """
        raise NotImplementedError

    @abstractmethod
    def apply_position_encoding(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: Optional[torch.LongTensor]
    ) -> torch.Tensor:
        """
        Apply position encoding to embeddings

        Different models handle position differently:
        - GPT-2: Add learned position embeddings
        - LLaMA2: RoPE is applied in attention, so just return inputs_embeds

        Args:
            inputs_embeds: Token embeddings [batch_size, seq_len, hidden_size]
            position_ids: Position indices [batch_size, seq_len]

        Returns:
            torch.Tensor: Embeddings with position encoding applied
        """
        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple] = None,
        use_cache: Optional[bool] = None,
    ):
        """
        Forward pass through Bottom model

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            past_key_values: Tuple of past key value states
            use_cache: Whether to return the key/value attentions

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]: 
                - Hidden states [batch_size, seq_len, hidden_size]
                - (Optional) Present key/value states if use_cache=True
        """
        # Step 1: Get token embeddings
        embedding_layer = self.get_embeddings()
        inputs_embeds = embedding_layer(input_ids)

        # Step 2: Apply position encoding
        # Note: When using cache, we might need to adjust position_ids if not provided
        if position_ids is None and past_key_values is not None and len(past_key_values) > 0:
            # Calculate past length from the first layer's key
            # past_key_values structure: ((k, v), (k, v), ...)
            # k shape: [batch, heads, seq_len, head_dim]
            past_length = past_key_values[0][0].size(-2)
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(
                past_length, past_length + seq_len, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)

        hidden_states = self.apply_position_encoding(inputs_embeds, position_ids)

        # Step 3: Prepare attention mask
        # Note: Attention mask handling might need to be model-specific for cache
        # For now, we pass it through, but implementations might need to adjust
        attention_mask = self.prepare_attention_mask(attention_mask, hidden_states)

        # Step 4: Pass through transformer blocks
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
