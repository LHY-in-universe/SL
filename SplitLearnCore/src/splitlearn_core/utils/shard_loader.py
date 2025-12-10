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

    @staticmethod
    def extract_embed_tokens(
        model_path: str,
        model_type: str,
        device: str = 'cpu',
        torch_dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
    ) -> torch.nn.Embedding:
        """
        从模型中提取 embed_tokens 层，无需加载完整模型。

        支持分片和非分片模型。比加载完整模型更节省内存。

        参数：
            model_path: 模型路径（本地目录或 HuggingFace 模型 ID）
            model_type: 模型架构（'qwen3_vl'、'qwen2_vl'、'qwen2'、'gpt2' 等）
            device: 目标设备（'cpu'、'cuda'、'mps'）
            torch_dtype: 权重数据类型（如 torch.float16）。如果为 None，使用原始 dtype
            cache_dir: 可选的缓存目录用于下载

        返回：
            torch.nn.Embedding: embed_tokens 层，可直接使用

        异常：
            ValueError: 如果 model_type 无效或找不到 embed_tokens
            FileNotFoundError: 如果找不到模型文件

        示例：
            >>> # 高效提取 embed_tokens
            >>> embed_tokens = ShardLoader.extract_embed_tokens(
            ...     model_path="Qwen/Qwen3-VL-2B-Instruct",
            ...     model_type="qwen3_vl",
            ...     device="cuda",
            ...     torch_dtype=torch.float16,
            ...     cache_dir="./models"
            ... )
            >>>
            >>> # 用于文本处理
            >>> input_ids = torch.tensor([[1, 2, 3, 4, 5]])
            >>> text_embeds = embed_tokens(input_ids)
            >>> print(text_embeds.shape)  # [1, 5, hidden_size]

        注意：
            - 对于 qwen3_vl: 查找 'model.language_model.embed_tokens.weight'
            - 对于 qwen2_vl/qwen2: 查找 'model.embed_tokens.weight'
            - 对于 gpt2: 查找 'transformer.wte.weight'
            - 自动检测分片/非分片模型
            - 比加载完整模型快得多（秒级 vs 分钟级）
            - 使用更少内存（MB 级 vs GB 级）
        """
        from splitlearn_core.utils.param_mapper import ParamMapper

        # 检查模型是否分片
        is_sharded = ShardLoader.is_sharded_model(model_path)
        print(f"  检测模型类型: {'分片模型' if is_sharded else '非分片模型'}")

        if is_sharded:
            # 分片模型：从索引文件加载
            print(f"  正在加载分片索引文件...")
            index_json = ShardLoader.load_index_json(model_path)
            weight_map = index_json.get("weight_map", {})
        else:
            # 非分片模型：直接从单个文件加载
            print(f"  使用非分片模式加载...")
            return ShardLoader._extract_embed_tokens_from_single_file(
                model_path=model_path,
                model_type=model_type,
                device=device,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
            )

        # 查找 embed_tokens 参数名称
        embed_param_name = None
        for param_name in weight_map.keys():
            if ParamMapper.is_embedding(param_name, model_type):
                embed_param_name = param_name
                break

        if embed_param_name is None:
            raise ValueError(
                f"在 {model_path} 中找不到 embed_tokens（model_type='{model_type}'）。"
                f"可用参数: {list(weight_map.keys())[:5]}..."
            )

        # 获取包含 embed_tokens 的分片文件
        shard_file = weight_map[embed_param_name]

        # 如果需要则下载分片
        shard_paths = ShardLoader.download_shards_if_needed(
            model_path,
            {shard_file},
            cache_dir=cache_dir,
            progress_bar=False
        )
        shard_path = shard_paths[shard_file]

        # 仅从分片中加载 embed_tokens（非常高效！）
        def filter_fn(name):
            return ParamMapper.is_embedding(name, model_type)

        embed_dict = ShardLoader.load_shard_partial(shard_path, filter_fn)

        if embed_param_name not in embed_dict:
            raise RuntimeError(
                f"从 {shard_file} 加载 {embed_param_name} 失败。"
                f"已加载的键: {list(embed_dict.keys())}"
            )

        # 获取 embed_tokens 权重张量
        embed_weight = embed_dict[embed_param_name]

        # 如果指定了 dtype，进行转换
        if torch_dtype is not None and embed_weight.is_floating_point():
            embed_weight = embed_weight.to(dtype=torch_dtype)

        # 创建 Embedding 层
        vocab_size, hidden_size = embed_weight.shape
        embed_layer = torch.nn.Embedding(vocab_size, hidden_size)
        embed_layer.weight.data = embed_weight
        embed_layer.requires_grad_(False)  # 推理时通常冻结

        # 移动到目标设备
        embed_layer.to(device)

        return embed_layer

    @staticmethod
    def _extract_embed_tokens_from_single_file(
        model_path: str,
        model_type: str,
        device: str = 'cpu',
        torch_dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
    ) -> torch.nn.Embedding:
        """
        从非分片模型文件中提取 embed_tokens（内部辅助方法）。

        参数：
            model_path: 模型路径（本地目录或 HuggingFace 模型 ID）
            model_type: 模型架构
            device: 目标设备
            torch_dtype: 权重数据类型
            cache_dir: 缓存目录

        返回：
            torch.nn.Embedding: embed_tokens 层
        """
        from splitlearn_core.utils.param_mapper import ParamMapper

        # 查找模型文件
        model_file = None

        # 尝试本地路径
        print(f"  检查本地路径: {model_path}")
        if os.path.exists(model_path):
            model_dir = Path(model_path)
            print(f"  本地路径存在，查找模型文件...")

            # 优先尝试 safetensors
            safetensors_file = model_dir / "model.safetensors"
            if safetensors_file.exists():
                print(f"  找到 safetensors 文件: {safetensors_file}")
                model_file = safetensors_file
            else:
                # 尝试 PyTorch bin 文件
                pytorch_file = model_dir / "pytorch_model.bin"
                if pytorch_file.exists():
                    print(f"  找到 PyTorch bin 文件: {pytorch_file}")
                    model_file = pytorch_file
                else:
                    print(f"  本地未找到 model.safetensors 或 pytorch_model.bin")

        # 从 HuggingFace Hub 下载
        if model_file is None:
            print(f"  尝试从 HuggingFace Hub 下载...")
            try:
                from huggingface_hub import hf_hub_download

                # 优先尝试 safetensors
                try:
                    print(f"    尝试下载 model.safetensors...")
                    model_file = Path(hf_hub_download(
                        repo_id=model_path,
                        filename="model.safetensors",
                        cache_dir=cache_dir,
                    ))
                    print(f"    ✓ 下载成功: {model_file}")
                except Exception as e1:
                    print(f"    model.safetensors 下载失败: {e1}")
                    # 回退到 PyTorch bin
                    try:
                        print(f"    尝试下载 pytorch_model.bin...")
                        model_file = Path(hf_hub_download(
                            repo_id=model_path,
                            filename="pytorch_model.bin",
                            cache_dir=cache_dir,
                        ))
                        print(f"    ✓ 下载成功: {model_file}")
                    except Exception as e2:
                        raise FileNotFoundError(
                            f"无法找到 {model_path} 的模型文件。"
                            f"尝试了 model.safetensors 和 pytorch_model.bin。"
                            f"错误1: {e1}\n错误2: {e2}"
                        )

            except ImportError:
                raise ImportError(
                    "huggingface_hub is required to download models. "
                    "Install with: pip install huggingface-hub"
                )

        if model_file is None:
            raise FileNotFoundError(
                f"在 {model_path} 中找不到模型文件 "
                f"(model.safetensors 或 pytorch_model.bin)"
            )

        # 定义过滤函数
        def filter_fn(name):
            result = ParamMapper.is_embedding(name, model_type)
            if result:
                print(f"    找到 embedding 参数: {name}")
            return result

        # 使用现有的 load_shard_partial 方法加载
        print(f"  从文件加载 embed_tokens: {model_file}")
        embed_dict = ShardLoader.load_shard_partial(model_file, filter_fn)

        if not embed_dict:
            raise ValueError(
                f"在 {model_file} 中找不到 embed_tokens（model_type='{model_type}'）"
            )

        # 获取 embed_tokens 权重（应该只有一个键）
        embed_param_name = list(embed_dict.keys())[0]
        print(f"  提取参数: {embed_param_name}")
        embed_weight = embed_dict[embed_param_name]
        print(f"  权重形状: {embed_weight.shape}, dtype: {embed_weight.dtype}")

        # 如果指定了 dtype，进行转换
        if torch_dtype is not None and embed_weight.is_floating_point():
            embed_weight = embed_weight.to(dtype=torch_dtype)

        # 创建 Embedding 层
        vocab_size, hidden_size = embed_weight.shape
        embed_layer = torch.nn.Embedding(vocab_size, hidden_size)
        embed_layer.weight.data = embed_weight
        embed_layer.requires_grad_(False)

        # 移动到目标设备
        embed_layer.to(device)

        return embed_layer
