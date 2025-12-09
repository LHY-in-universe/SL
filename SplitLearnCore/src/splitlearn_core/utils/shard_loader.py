"""
Shard loading utilities for split learning models.

This module provides utilities for detecting, downloading, and loading
sharded models incrementally to reduce memory usage.
"""

from pathlib import Path
from typing import Dict, Set, Optional, Callable, Union
import json
import os

import torch


class ShardLoader:
    """
    Utilities for loading sharded models incrementally.

    Supports:
    - Detecting sharded models (safetensors or PyTorch format)
    - Loading shard index files
    - Calculating required shards for each component
    - Downloading shards from HuggingFace Hub
    - Partial loading of shards
    """

    @staticmethod
    def is_sharded_model(model_path: str) -> bool:
        """
        Detect if a model is sharded.

        Args:
            model_path: Path to model (local directory or HuggingFace model ID)

        Returns:
            True if model is sharded, False otherwise

        Example:
            >>> ShardLoader.is_sharded_model("Qwen/Qwen2-7B")
            True
            >>> ShardLoader.is_sharded_model("gpt2")
            False
        """
        # Check for local path
        if os.path.exists(model_path):
            model_dir = Path(model_path)

            # Check for safetensors index
            safetensors_index = model_dir / "model.safetensors.index.json"
            if safetensors_index.exists():
                return True

            # Check for PyTorch index
            pytorch_index = model_dir / "pytorch_model.bin.index.json"
            if pytorch_index.exists():
                return True

            return False

        # For HuggingFace Hub IDs, try to download index file
        try:
            from huggingface_hub import hf_hub_download

            # Try safetensors index
            try:
                hf_hub_download(
                    repo_id=model_path,
                    filename="model.safetensors.index.json",
                    local_files_only=False,
                )
                return True
            except:
                pass

            # Try PyTorch index
            try:
                hf_hub_download(
                    repo_id=model_path,
                    filename="pytorch_model.bin.index.json",
                    local_files_only=False,
                )
                return True
            except:
                pass

            return False

        except ImportError:
            # huggingface_hub not installed
            return False

    @staticmethod
    def load_index_json(model_path: str) -> dict:
        """
        Load shard index file.

        Args:
            model_path: Path to model (local directory or HuggingFace model ID)

        Returns:
            Index dictionary with structure:
            {
                "metadata": {"total_size": ...},
                "weight_map": {"param.name": "shard-file.safetensors", ...}
            }

        Raises:
            FileNotFoundError: If index file not found

        Example:
            >>> index = ShardLoader.load_index_json("Qwen/Qwen2-7B")
            >>> index["weight_map"]["model.embed_tokens.weight"]
            'model-00001-of-00004.safetensors'
        """
        # Try local path first
        if os.path.exists(model_path):
            model_dir = Path(model_path)

            # Try safetensors index
            safetensors_index = model_dir / "model.safetensors.index.json"
            if safetensors_index.exists():
                with open(safetensors_index, 'r') as f:
                    return json.load(f)

            # Try PyTorch index
            pytorch_index = model_dir / "pytorch_model.bin.index.json"
            if pytorch_index.exists():
                with open(pytorch_index, 'r') as f:
                    return json.load(f)

            raise FileNotFoundError(
                f"No shard index file found in {model_path}"
            )

        # Try HuggingFace Hub
        try:
            from huggingface_hub import hf_hub_download

            # Try safetensors index
            try:
                index_path = hf_hub_download(
                    repo_id=model_path,
                    filename="model.safetensors.index.json",
                )
                with open(index_path, 'r') as f:
                    return json.load(f)
            except:
                pass

            # Try PyTorch index
            try:
                index_path = hf_hub_download(
                    repo_id=model_path,
                    filename="pytorch_model.bin.index.json",
                )
                with open(index_path, 'r') as f:
                    return json.load(f)
            except:
                pass

            raise FileNotFoundError(
                f"No shard index file found for {model_path}"
            )

        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download models from HF Hub. "
                "Install with: pip install huggingface-hub"
            )

    @staticmethod
    def get_required_shards_for_component(
        index_json: dict,
        component: str,
        layer_range: tuple,
        model_type: str,
        include_embedding: bool = False,
        include_final_norm: bool = False,
        include_lm_head: bool = False,
        include_visual: bool = False,
    ) -> Set[str]:
        """
        Calculate which shard files are needed for a component.

        Args:
            index_json: Shard index dictionary from load_index_json()
            component: Component type ('bottom', 'trunk', 'top')
            layer_range: (start_layer, end_layer) tuple
            model_type: Model architecture ('gpt2', 'qwen2', etc.)
            include_embedding: Whether to include embedding parameters
            include_final_norm: Whether to include final normalization
            include_lm_head: Whether to include language model head

        Returns:
            Set of shard filenames needed for this component

        Example:
            >>> index = ShardLoader.load_index_json("Qwen/Qwen2-7B")
            >>> shards = ShardLoader.get_required_shards_for_component(
            ...     index, 'bottom', (0, 8), 'qwen2', include_embedding=True
            ... )
            >>> shards
            {'model-00001-of-00004.safetensors'}
        """
        from splitlearn_core.utils.param_mapper import ParamMapper

        weight_map = index_json.get("weight_map", {})
        required_shards = set()

        layer_start, layer_end = layer_range

        for param_name, shard_file in weight_map.items():
            # Check if parameter belongs to this component
            should_include = False

            # Check special components
            if include_visual and param_name.startswith("model.visual."):
                should_include = True
            elif include_embedding and ParamMapper.is_embedding(param_name, model_type):
                should_include = True
            elif include_final_norm and ParamMapper.is_final_norm(param_name, model_type):
                should_include = True
            elif include_lm_head and ParamMapper.is_lm_head(param_name, model_type):
                should_include = True
            else:
                # Check layer range
                layer_num = ParamMapper.get_layer_number(param_name, model_type)
                if layer_num != -1:
                    if layer_start <= layer_num < layer_end:
                        should_include = True

            if should_include:
                required_shards.add(shard_file)

        return required_shards

    @staticmethod
    def download_shards_if_needed(
        model_path: str,
        shard_files: Set[str],
        cache_dir: Optional[str] = None,
        progress_bar: bool = True
    ) -> Dict[str, Path]:
        """
        Download shard files if needed.

        Args:
            model_path: Path to model (local directory or HuggingFace model ID)
            shard_files: Set of shard filenames to download
            cache_dir: Optional cache directory for downloads
            progress_bar: Whether to show download progress

        Returns:
            Dictionary mapping shard filename to local path

        Example:
            >>> shard_paths = ShardLoader.download_shards_if_needed(
            ...     "Qwen/Qwen2-7B",
            ...     {"model-00001-of-00004.safetensors"},
            ...     progress_bar=True
            ... )
            >>> shard_paths["model-00001-of-00004.safetensors"]
            PosixPath('/home/.cache/huggingface/hub/.../model-00001-of-00004.safetensors')
        """
        shard_paths = {}

        # Check if local path
        if os.path.exists(model_path):
            model_dir = Path(model_path)
            for shard_file in shard_files:
                shard_path = model_dir / shard_file
                if not shard_path.exists():
                    raise FileNotFoundError(
                        f"Shard file not found: {shard_path}"
                    )
                shard_paths[shard_file] = shard_path
            return shard_paths

        # Download from HuggingFace Hub
        try:
            from huggingface_hub import hf_hub_download
            from tqdm import tqdm

            iterator = tqdm(shard_files, desc="Downloading shards") if progress_bar else shard_files

            for shard_file in iterator:
                try:
                    shard_path = hf_hub_download(
                        repo_id=model_path,
                        filename=shard_file,
                        cache_dir=cache_dir,
                    )
                    shard_paths[shard_file] = Path(shard_path)

                except Exception as e:
                    # Retry logic
                    import time
                    for retry in range(3):
                        try:
                            time.sleep(2 ** retry)  # Exponential backoff
                            shard_path = hf_hub_download(
                                repo_id=model_path,
                                filename=shard_file,
                                cache_dir=cache_dir,
                            )
                            shard_paths[shard_file] = Path(shard_path)
                            break
                        except Exception as retry_e:
                            if retry == 2:  # Last retry
                                raise RuntimeError(
                                    f"Failed to download {shard_file} after 3 retries: {retry_e}"
                                )

            return shard_paths

        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download models. "
                "Install with: pip install huggingface-hub"
            )

    @staticmethod
    def load_shard_partial(
        shard_path: Path,
        filter_fn: Callable[[str], bool]
    ) -> Dict[str, torch.Tensor]:
        """
        Partially load a shard file, only loading filtered parameters.

        Args:
            shard_path: Path to shard file (.safetensors or .bin)
            filter_fn: Function that takes parameter name and returns True if should load

        Returns:
            Dictionary of parameter name to tensor

        Example:
            >>> def filter_fn(name):
            ...     return 'layer.0' in name or 'embed' in name
            >>> state_dict = ShardLoader.load_shard_partial(
            ...     Path("model-00001.safetensors"),
            ...     filter_fn
            ... )
            >>> len(state_dict)
            42
        """
        shard_path = Path(shard_path)

        # Detect file format
        if shard_path.suffix == '.safetensors':
            # Use safetensors for efficient partial loading
            try:
                from safetensors import safe_open

                state_dict = {}
                with safe_open(shard_path, framework="pt") as f:
                    for key in f.keys():
                        if filter_fn(key):
                            state_dict[key] = f.get_tensor(key)

                return state_dict

            except ImportError:
                raise ImportError(
                    "safetensors is required for loading .safetensors files. "
                    "Install with: pip install safetensors"
                )

        elif shard_path.suffix == '.bin':
            # Load entire .bin file, then filter
            full_state_dict = torch.load(shard_path, map_location='cpu', weights_only=False)

            state_dict = {
                key: value
                for key, value in full_state_dict.items()
                if filter_fn(key)
            }

            return state_dict

        else:
            raise ValueError(
                f"Unsupported shard format: {shard_path.suffix}. "
                f"Supported formats: .safetensors, .bin"
            )
