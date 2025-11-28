"""
Model Factory - Unified interface for creating split models

Provides a single entry point for creating split models of any supported architecture.
"""
from typing import Tuple
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from splitlearn.registry import ModelRegistry


class ModelFactory:
    """
    Factory for creating split model instances

    Provides a unified interface to create Bottom/Trunk/Top models for any
    registered model architecture.
    """

    # Map common model names to their HuggingFace model classes
    _MODEL_CLASS_MAP = {
        'gpt2': 'GPT2LMHeadModel',
        'llama2': 'LlamaForCausalLM',
        'llama': 'LlamaForCausalLM',
        'gpt-j': 'GPTJForCausalLM',
    }

    @staticmethod
    def create_split_models(
        model_type: str,
        model_name_or_path: str,
        split_point_1: int,
        split_point_2: int,
        device: str = 'cpu',
    ) -> Tuple:
        """
        Create all three split model parts for any supported architecture

        Args:
            model_type: Model architecture ('gpt2', 'llama2', etc.)
            model_name_or_path: HuggingFace model ID or local path
            split_point_1: End of Bottom layers (exclusive)
            split_point_2: Start of Top layers (inclusive)
            device: Device to load models on

        Returns:
            Tuple: (bottom_model, trunk_model, top_model)

        Raises:
            KeyError: If model_type is not registered
            ValueError: If split points are invalid

        Example:
            bottom, trunk, top = ModelFactory.create_split_models(
                model_type='gpt2',
                model_name_or_path='gpt2',
                split_point_1=2,
                split_point_2=10,
            )
        """
        # Validate model type is registered
        if not ModelRegistry.is_model_registered(model_type):
            available = ModelRegistry.list_supported_models()
            raise KeyError(
                f"Model type '{model_type}' not registered. "
                f"Available models: {available}"
            )

        # Check all parts are registered
        if not ModelRegistry.is_complete(model_type):
            info = ModelRegistry.get_model_info()[model_type]
            raise ValueError(
                f"Model '{model_type}' is incomplete. "
                f"Registration status: {info}"
            )

        print(f"Loading pretrained model '{model_name_or_path}'...")

        # Load full model and config
        config = AutoConfig.from_pretrained(model_name_or_path)
        full_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        full_state_dict = full_model.state_dict()

        # Get number of layers
        num_layers = getattr(config, 'n_layer', None) or \
                    getattr(config, 'num_hidden_layers', None)

        if num_layers is None:
            raise ValueError(
                f"Cannot determine number of layers from config. "
                f"Config type: {type(config)}"
            )

        # Validate split points
        if not (0 < split_point_1 < split_point_2 < num_layers):
            raise ValueError(
                f"Invalid split points: 0 < {split_point_1} < {split_point_2} < {num_layers}"
            )

        print(f"\nSplitting {model_type} model:")
        print(f"  Total layers: {num_layers}")
        print(f"  Bottom: Layers[0:{split_point_1}]")
        print(f"  Trunk:  Layers[{split_point_1}:{split_point_2}]")
        print(f"  Top:    Layers[{split_point_2}:{num_layers}]")

        # Get model classes from registry
        BottomCls = ModelRegistry.get_model_class(model_type, 'bottom')
        TrunkCls = ModelRegistry.get_model_class(model_type, 'trunk')
        TopCls = ModelRegistry.get_model_class(model_type, 'top')

        # Create split models
        print("\nCreating Bottom model...")
        bottom_model = BottomCls.from_pretrained_split(
            full_state_dict, config, end_layer=split_point_1
        )

        print("Creating Trunk model...")
        trunk_model = TrunkCls.from_pretrained_split(
            full_state_dict, config,
            start_layer=split_point_1,
            end_layer=split_point_2
        )

        print("Creating Top model...")
        top_model = TopCls.from_pretrained_split(
            full_state_dict, config, start_layer=split_point_2
        )

        # Move to device
        bottom_model = bottom_model.to(device)
        trunk_model = trunk_model.to(device)
        top_model = top_model.to(device)

        # Print statistics
        print("\nModel Statistics:")
        print(f"  Bottom: {bottom_model.num_parameters():,} parameters "
              f"({bottom_model.memory_footprint_mb():.2f} MB)")
        print(f"  Trunk:  {trunk_model.num_parameters():,} parameters "
              f"({trunk_model.memory_footprint_mb():.2f} MB)")
        print(f"  Top:    {top_model.num_parameters():,} parameters "
              f"({top_model.memory_footprint_mb():.2f} MB)")

        total_split = (bottom_model.num_parameters() +
                      trunk_model.num_parameters() +
                      top_model.num_parameters())
        total_full = sum(p.numel() for p in full_model.parameters())

        print(f"  Total split: {total_split:,} parameters")
        print(f"  Full model:  {total_full:,} parameters")

        if abs(total_split - total_full) > 1000:
            print(f"  Warning: Parameter count mismatch! "
                  f"Difference: {abs(total_split - total_full):,}")

        return bottom_model, trunk_model, top_model

    @staticmethod
    def list_available_models():
        """
        List all available model types and their registration status

        Prints a table showing which models are fully registered.
        """
        print("\nRegistered Models:")
        print("=" * 60)

        info = ModelRegistry.get_model_info()

        if not info:
            print("No models registered yet.")
            return

        for model_type, status in sorted(info.items()):
            status_str = "✓ Complete" if status['complete'] else "✗ Incomplete"
            parts_str = ", ".join([
                f"{part}: {'✓' if registered else '✗'}"
                for part, registered in status.items()
                if part != 'complete'
            ])

            print(f"{model_type:15s} {status_str:15s} ({parts_str})")

        print("=" * 60)
