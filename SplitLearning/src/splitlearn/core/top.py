"""
Base abstract class for Top models

Top models handle: Last K transformer layers + LM Head
"""
from abc import abstractmethod
from typing import Optional
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from .base import BaseSplitModel


class BaseTopModel(BaseSplitModel):
    """
    Abstract base class for Top split models

    Top models are responsible for:
    1. Last K transformer layers
    2. Final normalization
    3. Language modeling head
    4. Loss computation

    Top models are typically deployed on the client side.
    """

    def __init__(self, config, start_layer: int):
        """
        Initialize Top model

        Args:
            config: Model configuration
            start_layer: Starting layer index in original model
        """
        # Calculate end_layer based on total layers in config
        num_layers = getattr(config, 'n_layer', None) or getattr(config, 'num_hidden_layers', None)
        if num_layers is None:
            raise ValueError("Config must have 'n_layer' or 'num_hidden_layers' attribute")

        super().__init__(config, start_layer, num_layers)

    @abstractmethod
    def get_final_norm(self) -> nn.Module:
        """
        Get the final normalization layer

        Returns:
            nn.Module: Final normalization layer

        Example:
            GPT-2: return self.ln_f (LayerNorm)
            LLaMA2: return self.norm (RMSNorm)
        """
        raise NotImplementedError

    @abstractmethod
    def get_lm_head(self) -> nn.Module:
        """
        Get the language modeling head

        Returns:
            nn.Module: LM head linear layer

        Example:
            Both GPT-2 and LLaMA2: return self.lm_head
        """
        raise NotImplementedError

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple] = None,
        use_cache: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass through Top model

        Args:
            hidden_states: Input from Trunk [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for loss computation [batch_size, seq_len]
            past_key_values: Tuple of past key value states
            use_cache: Whether to return the key/value attentions

        Returns:
            CausalLMOutputWithPast: Output with logits, past_key_values, and optional loss
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

        # Final normalization
        hidden_states = self.get_final_norm()(hidden_states)

        # Language modeling head
        logits = self.get_lm_head()(hidden_states).float()

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for causal LM: use token i to predict token i+1
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten and compute cross entropy
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents if use_cache else None,
        )
