"""
Model Factory - Unified interface for creating split models

Provides a single entry point for creating split models of any supported architecture.
Supports both traditional loading and incremental loading for sharded models.
"""
from typing import Tuple, Optional, Union, Dict, Any
import gc
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from splitlearn.registry import ModelRegistry


class ModelFactory:
    """
    Factory for creating split model instances

    Provides a unified interface to create Bottom/Trunk/Top models for any
    registered model architecture.

    Supports:
    - Traditional loading: Load full model then split
    - Incremental loading: Load only required shards for each component (low memory)
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
        device_map: Optional[Union[str, Dict]] = None,
        low_memory: bool = False,
        verbose: bool = False,
        storage_path: Optional[str] = None,
        auto_save: bool = False,
    ) -> Tuple:
        """
        Create all three split model parts for any supported architecture

        Args:
            model_type: Model architecture ('gpt2', 'qwen2', etc.)
            model_name_or_path: HuggingFace model ID or local path
            split_point_1: End of Bottom layers (exclusive)
            split_point_2: Start of Top layers (inclusive)
            device: Device to load models on ('cpu', 'cuda', etc.)
            device_map: Device mapping for multi-device support
                - None: Use single device for all components
                - 'auto': Automatically distribute across available devices
                - Dict: Manual mapping, e.g., {'bottom': 'cpu', 'trunk': 'cuda:0', 'top': 'cuda:1'}
            low_memory: Enable incremental loading for sharded models (reduces memory peak)
            verbose: Show detailed progress and memory usage
            storage_path: Base directory for saving split models (optional)
            auto_save: Whether to automatically save split models (default: False)

        Returns:
            Tuple: (bottom_model, trunk_model, top_model)

        Raises:
            KeyError: If model_type is not registered
            ValueError: If split points are invalid

        Example (traditional):
            >>> bottom, trunk, top = ModelFactory.create_split_models(
            ...     model_type='gpt2',
            ...     model_name_or_path='gpt2',
            ...     split_point_1=2,
            ...     split_point_2=10,
            ... )

        Example (low memory):
            >>> bottom, trunk, top = ModelFactory.create_split_models(
            ...     model_type='qwen2',
            ...     model_name_or_path='Qwen/Qwen2-7B',
            ...     split_point_1=8,
            ...     split_point_2=24,
            ...     low_memory=True,
            ...     verbose=True,
            ...     device_map='auto',
            ... )
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

        # Load config (lightweight)
        config = AutoConfig.from_pretrained(model_name_or_path)

        # Detect if model is sharded
        from splitlearn.utils.shard_loader import ShardLoader
        is_sharded = ShardLoader.is_sharded_model(model_name_or_path)

        if is_sharded and verbose:
            print(f"Detected sharded model")

        # Choose loading strategy
        if is_sharded and low_memory:
            if verbose:
                print("Using incremental loading (low_memory=True)")
            return ModelFactory._create_split_models_incremental(
                model_type=model_type,
                model_name_or_path=model_name_or_path,
                config=config,
                split_point_1=split_point_1,
                split_point_2=split_point_2,
                device=device,
                device_map=device_map,
                verbose=verbose,
                storage_path=storage_path,
                auto_save=auto_save,
            )
        else:
            if verbose and is_sharded:
                print("Using traditional loading (low_memory=False)")
            return ModelFactory._create_split_models_traditional(
                model_type=model_type,
                model_name_or_path=model_name_or_path,
                config=config,
                split_point_1=split_point_1,
                split_point_2=split_point_2,
                device=device,
                storage_path=storage_path,
                auto_save=auto_save,
            )

    @staticmethod
    def _create_split_models_traditional(
        model_type: str,
        model_name_or_path: str,
        config,
        split_point_1: int,
        split_point_2: int,
        device: str,
        storage_path: Optional[str],
        auto_save: bool,
    ) -> Tuple:
        """
        Traditional loading: Load full model then split.

        This is the original loading method, kept for backward compatibility
        and for non-sharded models.
        """
        # Load full model and config
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

        # Auto-save functionality
        if auto_save and storage_path:
            print("\n" + "="*60)
            print("Saving split models to storage...")
            print("="*60)

            from splitlearn.utils import StorageManager

            # Create storage directories
            dirs = StorageManager.create_storage_directories(storage_path)
            print(f"Storage directory: {storage_path}")

            # Generate split configuration string
            split_config = f"{split_point_1}-{split_point_2}"

            # Extract base model name
            model_basename = model_name_or_path.split('/')[-1]

            # Save models
            bottom_path = StorageManager.get_split_model_path(
                storage_path, model_basename, "bottom", split_config
            )
            bottom_model.save_split_model(bottom_path)
            print(f"  Bottom model saved: {bottom_path}")

            trunk_path = StorageManager.get_split_model_path(
                storage_path, model_basename, "trunk", split_config
            )
            trunk_model.save_split_model(trunk_path)
            print(f"  Trunk model saved: {trunk_path}")

            top_path = StorageManager.get_split_model_path(
                storage_path, model_basename, "top", split_config
            )
            top_model.save_split_model(top_path)
            print(f"  Top model saved: {top_path}")

            print("="*60)

        return bottom_model, trunk_model, top_model

    @staticmethod
    def _create_split_models_incremental(
        model_type: str,
        model_name_or_path: str,
        config,
        split_point_1: int,
        split_point_2: int,
        device: str,
        device_map: Optional[Union[str, Dict]],
        verbose: bool,
        storage_path: Optional[str],
        auto_save: bool,
    ) -> Tuple:
        """
        Incremental loading: Load only required shards for each component.

        This significantly reduces memory peak by loading components one at a time
        and releasing memory immediately after creating each component.
        """
        from splitlearn.utils.shard_loader import ShardLoader
        from splitlearn.utils.memory_tracker import MemoryTracker

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

        # Load shard index
        if verbose:
            print("\nLoading shard index...")
        index_json = ShardLoader.load_index_json(model_name_or_path)

        # Parse device map
        device_bottom, device_trunk, device_top = ModelFactory._parse_device_map(
            device, device_map
        )

        if verbose:
            print(f"\nDevice allocation:")
            print(f"  Bottom → {device_bottom}")
            print(f"  Trunk  → {device_trunk}")
            print(f"  Top    → {device_top}")

        # Create memory tracker
        mem_tracker = MemoryTracker() if verbose else None

        if verbose:
            mem_tracker.snapshot("Initial")

        # Create Bottom
        print("\n" + "="*60)
        print("Creating Bottom Model")
        print("="*60)

        bottom_model = ModelFactory._load_component_incremental(
            component='bottom',
            model_type=model_type,
            model_name_or_path=model_name_or_path,
            config=config,
            index_json=index_json,
            layer_range=(0, split_point_1),
            device=device_bottom,
            verbose=verbose,
        )

        if verbose:
            mem_tracker.snapshot("After Bottom")
            mem_tracker.report()

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Create Trunk
        print("\n" + "="*60)
        print("Creating Trunk Model")
        print("="*60)

        trunk_model = ModelFactory._load_component_incremental(
            component='trunk',
            model_type=model_type,
            model_name_or_path=model_name_or_path,
            config=config,
            index_json=index_json,
            layer_range=(split_point_1, split_point_2),
            device=device_trunk,
            verbose=verbose,
        )

        if verbose:
            mem_tracker.snapshot("After Trunk")
            mem_tracker.report()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Create Top
        print("\n" + "="*60)
        print("Creating Top Model")
        print("="*60)

        top_model = ModelFactory._load_component_incremental(
            component='top',
            model_type=model_type,
            model_name_or_path=model_name_or_path,
            config=config,
            index_json=index_json,
            layer_range=(split_point_2, num_layers),
            device=device_top,
            verbose=verbose,
        )

        if verbose:
            mem_tracker.snapshot("After Top")
            mem_tracker.report()

        gc.collect()

        # Print statistics
        print("\n" + "="*60)
        print("Model Statistics")
        print("="*60)
        print(f"  Bottom: {bottom_model.num_parameters():,} parameters "
              f"({bottom_model.memory_footprint_mb():.2f} MB)")
        print(f"  Trunk:  {trunk_model.num_parameters():,} parameters "
              f"({trunk_model.memory_footprint_mb():.2f} MB)")
        print(f"  Top:    {top_model.num_parameters():,} parameters "
              f"({top_model.memory_footprint_mb():.2f} MB)")

        total_split = (bottom_model.num_parameters() +
                      trunk_model.num_parameters() +
                      top_model.num_parameters())

        print(f"  Total split: {total_split:,} parameters")

        if verbose:
            mem_tracker.summary()

        # Auto-save functionality
        if auto_save and storage_path:
            print("\n" + "="*60)
            print("Saving split models to storage...")
            print("="*60)

            from splitlearn.utils import StorageManager

            dirs = StorageManager.create_storage_directories(storage_path)
            print(f"Storage directory: {storage_path}")

            split_config = f"{split_point_1}-{split_point_2}"
            model_basename = model_name_or_path.split('/')[-1]

            bottom_path = StorageManager.get_split_model_path(
                storage_path, model_basename, "bottom", split_config
            )
            bottom_model.save_split_model(bottom_path)
            print(f"  Bottom model saved: {bottom_path}")

            trunk_path = StorageManager.get_split_model_path(
                storage_path, model_basename, "trunk", split_config
            )
            trunk_model.save_split_model(trunk_path)
            print(f"  Trunk model saved: {trunk_path}")

            top_path = StorageManager.get_split_model_path(
                storage_path, model_basename, "top", split_config
            )
            top_model.save_split_model(top_path)
            print(f"  Top model saved: {top_path}")

            print("="*60)

        return bottom_model, trunk_model, top_model

    @staticmethod
    def _load_component_incremental(
        component: str,
        model_type: str,
        model_name_or_path: str,
        config,
        index_json: dict,
        layer_range: tuple,
        device: str,
        verbose: bool,
    ) -> Any:
        """
        Load a single component incrementally from shards.

        This method:
        1. Calculates required shards for the component
        2. Downloads shards if needed
        3. Loads parameters from shards (filtered)
        4. Creates the component model
        5. Moves to device
        6. Releases shard data
        """
        from splitlearn.utils.shard_loader import ShardLoader
        from splitlearn.utils.param_mapper import ParamMapper
        from tqdm import tqdm

        layer_start, layer_end = layer_range

        # Determine component configuration
        include_embedding = (component == 'bottom')
        include_final_norm = (component == 'top')
        include_lm_head = (component == 'top')

        # Calculate required shards
        if verbose:
            print(f"  Calculating required shards for {component}...")

        required_shards = ShardLoader.get_required_shards_for_component(
            index_json=index_json,
            component=component,
            layer_range=layer_range,
            model_type=model_type,
            include_embedding=include_embedding,
            include_final_norm=include_final_norm,
            include_lm_head=include_lm_head,
        )

        if verbose:
            print(f"  Required shards: {sorted(required_shards)}")

        # Download shards if needed
        if verbose:
            print(f"  Downloading/locating shards...")

        shard_paths = ShardLoader.download_shards_if_needed(
            model_name_or_path,
            required_shards,
            progress_bar=verbose
        )

        # Load parameters from shards
        if verbose:
            print(f"  Loading parameters from shards...")

        component_state_dict = {}

        # Create filter function
        def filter_fn(param_name: str) -> bool:
            """Check if parameter belongs to this component"""
            # Check special components
            if include_embedding and ParamMapper.is_embedding(param_name, model_type):
                return True
            if include_final_norm and ParamMapper.is_final_norm(param_name, model_type):
                return True
            if include_lm_head and ParamMapper.is_lm_head(param_name, model_type):
                return True

            # Check layer range
            layer_num = ParamMapper.get_layer_number(param_name, model_type)
            if layer_num != -1:
                return layer_start <= layer_num < layer_end

            return False

        # Load shards with progress bar
        iterator = tqdm(
            shard_paths.items(),
            desc=f"  Loading {component}",
            disable=not verbose
        )

        for shard_name, shard_path in iterator:
            partial_dict = ShardLoader.load_shard_partial(shard_path, filter_fn)
            component_state_dict.update(partial_dict)

            if verbose:
                iterator.set_postfix({
                    'params': len(component_state_dict),
                    'shard': shard_name
                })

        if verbose:
            print(f"  Loaded {len(component_state_dict)} parameters")

        # Create component model
        if verbose:
            print(f"  Creating {component} model...")

        ComponentCls = ModelRegistry.get_model_class(model_type, component)

        if component == 'bottom':
            model = ComponentCls.from_pretrained_split(
                component_state_dict, config, end_layer=layer_end
            )
        elif component == 'trunk':
            model = ComponentCls.from_pretrained_split(
                component_state_dict, config,
                start_layer=layer_start, end_layer=layer_end
            )
        else:  # top
            model = ComponentCls.from_pretrained_split(
                component_state_dict, config, start_layer=layer_start
            )

        # Move to device
        if verbose:
            print(f"  Moving to device: {device}")
        model = model.to(device)

        # Release memory
        del component_state_dict
        gc.collect()

        return model

    @staticmethod
    def _parse_device_map(
        device: str,
        device_map: Optional[Union[str, Dict]]
    ) -> Tuple[str, str, str]:
        """
        Parse device_map parameter into device assignments for each component.

        Returns:
            Tuple: (device_bottom, device_trunk, device_top)
        """
        if device_map is None:
            # Use single device for all
            return device, device, device

        if device_map == 'auto':
            # Automatic allocation
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                if num_gpus >= 3:
                    # Distribute across 3 GPUs
                    return 'cuda:0', 'cuda:1', 'cuda:2'
                elif num_gpus >= 2:
                    # Trunk on GPU 0 (most compute), Top on GPU 1, Bottom on CPU
                    return 'cpu', 'cuda:0', 'cuda:1'
                else:
                    # Trunk on GPU, others on CPU
                    return 'cpu', 'cuda:0', 'cpu'
            else:
                return 'cpu', 'cpu', 'cpu'

        if isinstance(device_map, dict):
            # Manual mapping
            return (
                device_map.get('bottom', device),
                device_map.get('trunk', device),
                device_map.get('top', device)
            )

        raise ValueError(
            f"Invalid device_map: {device_map}. "
            f"Expected None, 'auto', or dict with 'bottom'/'trunk'/'top' keys"
        )

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
