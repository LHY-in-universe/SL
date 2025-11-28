"""
Base abstract class for all split models

Defines the common interface that all split model parts must implement.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import json
import torch
import torch.nn as nn


class BaseSplitModel(nn.Module, ABC):
    """
    Abstract base class for all split model components

    This class defines the common interface that Bottom, Trunk, and Top models
    must implement, regardless of the underlying architecture (GPT-2, LLaMA2, etc.)

    Attributes:
        config: Model-specific configuration object
        start_layer: Starting layer index in the original full model
        end_layer: Ending layer index (exclusive) in the original full model
        num_layers: Number of transformer layers in this split part
    """

    def __init__(self, config, start_layer: int, end_layer: int):
        """
        Initialize base split model

        Args:
            config: Model configuration (e.g., GPT2Config, LlamaConfig)
            start_layer: Starting layer index
            end_layer: Ending layer index (exclusive)
        """
        super().__init__()
        self.config = config
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.num_layers = end_layer - start_layer

        # Ensure _attn_implementation is set (required by newer transformers versions)
        if not hasattr(self.config, "_attn_implementation"):
            self.config._attn_implementation = "eager"

    @abstractmethod
    def get_transformer_blocks(self) -> nn.ModuleList:
        """
        Get the ModuleList containing transformer blocks

        Returns:
            nn.ModuleList: List of transformer layers

        Example:
            For GPT-2: return self.h
            For LLaMA2: return self.layers
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        hidden_states: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Prepare model-specific attention mask format

        Different models use different attention mask formats:
        - GPT-2: 4D causal mask with -inf for masked positions
        - LLaMA2: Different format with RoPE considerations

        Args:
            attention_mask: Input attention mask [batch_size, seq_len]
            hidden_states: Hidden states for determining dimensions

        Returns:
            Prepared attention mask in model-specific format
        """
        raise NotImplementedError

    @abstractmethod
    def get_layer_name_pattern(self) -> str:
        """
        Get regex pattern for extracting layer numbers from parameter names

        Returns:
            str: Regex pattern string

        Example:
            GPT-2: r'\\.h\\.[0-9]+'
            LLaMA2: r'\\.layers\\.[0-9]+'
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_pretrained_split(
        cls,
        full_state_dict: Dict[str, Any],
        config,
        **kwargs
    ):
        """
        Create a split model instance from full model state dict

        This method should:
        1. Filter the state dict to include only relevant layers
        2. Remap layer indices if necessary
        3. Load the filtered weights into the new model

        Args:
            full_state_dict: State dict from full model
            config: Model configuration
            **kwargs: Additional model-specific arguments

        Returns:
            Instance of the split model with loaded weights
        """
        raise NotImplementedError

    def num_parameters(self) -> int:
        """
        Count total number of parameters

        Returns:
            int: Total parameter count
        """
        return sum(p.numel() for p in self.parameters())

    def memory_footprint_mb(self) -> float:
        """
        Calculate memory footprint in MB (assuming float32)

        Returns:
            float: Memory usage in megabytes
        """
        num_params = self.num_parameters()
        return num_params * 4 / (1024 ** 2)

    def _fix_attention_implementation(self):
        """
        Fix _attn_implementation for all transformer blocks.

        This is needed for compatibility with newer transformers versions.
        Should be called after transformer blocks are initialized.
        """
        blocks = self.get_transformer_blocks()
        for block in blocks:
            if hasattr(block, "attn") and hasattr(block.attn, "config"):
                # Set if not present or if None
                if not hasattr(block.attn.config, "_attn_implementation") or \
                   block.attn.config._attn_implementation is None:
                    block.attn.config._attn_implementation = "eager"

    def _init_weights(self, module):
        """
        Initialize model weights (common implementation for GPT-2 style models).

        This method initializes:
        - Linear and Embedding layers with normal distribution (std = config.initializer_range)
        - LayerNorm layers with zeros for bias and ones for weight

        Subclasses can override this method for model-specific initialization.

        Args:
            module: The module to initialize
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def save_split_model(
        self,
        save_path: Union[str, Path],
        save_metadata: bool = True
    ) -> None:
        """
        Save split model to disk.

        Args:
            save_path: Path to save model checkpoint
            save_metadata: Whether to save metadata JSON file

        Saves:
            - <name>.pt: PyTorch state dict
            - <name>_metadata.json: Model configuration and split info (if save_metadata=True)

        Example:
            >>> model.save_split_model("./models/bottom/gpt2_bottom.pt")
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state dict
        torch.save(self.state_dict(), save_path)

        # Save metadata if requested
        if save_metadata:
            metadata = self._generate_metadata()
            metadata_path = save_path.parent / f"{save_path.stem}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

    def _generate_metadata(self) -> dict:
        """
        Generate metadata dictionary for saved model.

        Returns:
            Dictionary containing model metadata

        Metadata includes:
            - model_class: Name of the model class
            - component: Component type (bottom/trunk/top)
            - start_layer: Starting layer index
            - end_layer: Ending layer index
            - num_layers: Number of layers
            - num_parameters: Total parameter count
            - memory_mb: Memory footprint in MB
            - config: Model configuration
            - saved_at: ISO timestamp
        """
        # Determine component type from class name
        class_name = self.__class__.__name__
        if "Bottom" in class_name:
            component = "bottom"
        elif "Trunk" in class_name:
            component = "trunk"
        elif "Top" in class_name:
            component = "top"
        else:
            component = "unknown"

        # Get config as dict if possible
        config_dict = {}
        if hasattr(self.config, 'to_dict'):
            config_dict = self.config.to_dict()
        elif hasattr(self.config, '__dict__'):
            config_dict = {k: v for k, v in self.config.__dict__.items()
                          if not k.startswith('_')}

        return {
            "model_class": class_name,
            "component": component,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "num_layers": self.num_layers,
            "num_parameters": self.num_parameters(),
            "memory_mb": self.memory_footprint_mb(),
            "config": config_dict,
            "saved_at": datetime.now().isoformat()
        }
