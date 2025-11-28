"""
Example of extending SplitLearn to support a new model architecture.

This demonstrates the architecture for adding support for new models.
Note: This is a template/skeleton showing the structure, not a complete implementation.
"""

from typing import Optional
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from splitlearn.core import BaseBottomModel, BaseTrunkModel, BaseTopModel
from splitlearn.registry import ModelRegistry
from splitlearn.utils import ParamMapper


# Step 1: Register your models with the registry
@ModelRegistry.register('my_model', 'bottom')
class MyBottomModel(BaseBottomModel):
    """
    Bottom model implementation for your custom architecture.

    Responsibilities:
    - Token embeddings
    - Position embeddings
    - First N layers
    """

    def __init__(self, config: PretrainedConfig, end_layer: int):
        super().__init__(config, end_layer)

        # Initialize embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # Initialize transformer layers (first end_layer layers)
        self.layers = nn.ModuleList([
            # YourTransformerLayer(config) for _ in range(end_layer)
        ])

        # Layer normalization or other components
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def get_embeddings(self):
        """Return the token embedding layer."""
        return self.embed_tokens

    def apply_position_encoding(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor):
        """
        Apply position encoding to input embeddings.

        Args:
            inputs_embeds: Token embeddings [batch_size, seq_len, hidden_size]
            position_ids: Position indices [batch_size, seq_len]

        Returns:
            Tensor with position encoding applied
        """
        position_embeds = self.position_embeddings(position_ids)
        return inputs_embeds + position_embeds

    def get_transformer_blocks(self):
        """Return the list of transformer layers."""
        return self.layers

    def prepare_attention_mask(self, attention_mask: Optional[torch.Tensor], hidden_states: torch.Tensor):
        """
        Prepare attention mask for your architecture.

        Different models have different attention mask formats:
        - GPT-2: causal mask, shape [batch_size, 1, 1, seq_len]
        - BERT: bidirectional mask
        - etc.

        Args:
            attention_mask: Optional mask [batch_size, seq_len]
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]

        Returns:
            Prepared attention mask in the format your model expects
        """
        if attention_mask is None:
            return None

        # Implement your attention mask preparation
        # Example for causal mask:
        batch_size, seq_len = attention_mask.shape
        # ... prepare mask according to your model's requirements

        return attention_mask

    def get_layer_name_pattern(self):
        """
        Return regex pattern for layer names in state dict.

        Examples:
        - GPT-2: r'\\.h\\.[0-9]+'
        - Qwen2/LLaMA: r'\\.layers\\.[0-9]+'
        - BERT: r'\\.layer\\.[0-9]+'
        """
        return r'\\.layers\\.[0-9]+'  # Adjust for your model

    @classmethod
    def from_pretrained_split(
        cls,
        full_state_dict: dict,
        config: PretrainedConfig,
        end_layer: int
    ):
        """
        Create bottom model from full model's state dict.

        Args:
            full_state_dict: State dict from full pretrained model
            config: Model configuration
            end_layer: Number of layers to include (0 to end_layer-1)

        Returns:
            Initialized bottom model
        """
        model = cls(config, end_layer)

        # Use ParamMapper to filter and load weights
        param_mapper = ParamMapper(
            layer_pattern=model.get_layer_name_pattern(),
            start_layer=0,
            end_layer=end_layer
        )

        filtered_state_dict = param_mapper.filter_and_remap_state_dict(
            full_state_dict,
            include_embeddings=True,
            include_lm_head=False
        )

        model.load_state_dict(filtered_state_dict, strict=False)
        return model


@ModelRegistry.register('my_model', 'trunk')
class MyTrunkModel(BaseTrunkModel):
    """
    Trunk model for middle layers.

    Simpler than bottom/top - just transformer layers.
    """

    def __init__(self, config: PretrainedConfig, start_layer: int, end_layer: int):
        super().__init__(config, start_layer, end_layer)

        num_layers = end_layer - start_layer
        self.layers = nn.ModuleList([
            # YourTransformerLayer(config) for _ in range(num_layers)
        ])

    def get_transformer_blocks(self):
        return self.layers

    def prepare_attention_mask(self, attention_mask: Optional[torch.Tensor], hidden_states: torch.Tensor):
        # Same as bottom model
        return attention_mask

    def get_layer_name_pattern(self):
        return r'\\.layers\\.[0-9]+'

    @classmethod
    def from_pretrained_split(
        cls,
        full_state_dict: dict,
        config: PretrainedConfig,
        start_layer: int,
        end_layer: int
    ):
        model = cls(config, start_layer, end_layer)

        param_mapper = ParamMapper(
            layer_pattern=model.get_layer_name_pattern(),
            start_layer=start_layer,
            end_layer=end_layer
        )

        filtered_state_dict = param_mapper.filter_and_remap_state_dict(
            full_state_dict,
            include_embeddings=False,
            include_lm_head=False
        )

        model.load_state_dict(filtered_state_dict, strict=False)
        return model


@ModelRegistry.register('my_model', 'top')
class MyTopModel(BaseTopModel):
    """
    Top model for final layers + LM head.
    """

    def __init__(self, config: PretrainedConfig, start_layer: int):
        super().__init__(config, start_layer)

        num_layers = config.num_hidden_layers - start_layer
        self.layers = nn.ModuleList([
            # YourTransformerLayer(config) for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_transformer_blocks(self):
        return self.layers

    def get_lm_head(self):
        return self.lm_head

    def prepare_attention_mask(self, attention_mask: Optional[torch.Tensor], hidden_states: torch.Tensor):
        return attention_mask

    def get_layer_name_pattern(self):
        return r'\\.layers\\.[0-9]+'

    @classmethod
    def from_pretrained_split(
        cls,
        full_state_dict: dict,
        config: PretrainedConfig,
        start_layer: int
    ):
        model = cls(config, start_layer)

        param_mapper = ParamMapper(
            layer_pattern=model.get_layer_name_pattern(),
            start_layer=start_layer,
            end_layer=config.num_hidden_layers
        )

        filtered_state_dict = param_mapper.filter_and_remap_state_dict(
            full_state_dict,
            include_embeddings=False,
            include_lm_head=True
        )

        model.load_state_dict(filtered_state_dict, strict=False)
        return model


def main():
    """
    After implementing your models, you can use them with ModelFactory:

    from splitlearn import ModelFactory

    bottom, trunk, top = ModelFactory.create_split_models(
        model_type='my_model',  # Your registered model type
        model_name_or_path='path/to/pretrained',
        split_point_1=4,
        split_point_2=8,
        device='cpu'
    )
    """
    print("This is a template for extending SplitLearn to new architectures.")
    print("\nKey steps:")
    print("1. Inherit from Base*Model classes")
    print("2. Implement required abstract methods")
    print("3. Register with @ModelRegistry.register decorator")
    print("4. Implement from_pretrained_split() classmethod")
    print("5. Use ModelFactory.create_split_models() with your model_type")
    print("\nSee the GPT-2 and Qwen2 implementations in src/splitlearn/models/ for reference.")


if __name__ == "__main__":
    main()
