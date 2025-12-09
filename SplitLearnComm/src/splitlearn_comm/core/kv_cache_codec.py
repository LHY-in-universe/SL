"""
KV-Cache Codec for Serialization/Deserialization

Handles encoding and decoding of Key-Value cache tensors for efficient
transmission over gRPC.
"""

import logging
from typing import Tuple, Optional, List

import torch

from .tensor_codec import TensorCodec

logger = logging.getLogger(__name__)


class KVCacheCodec:
    """
    Codec for Key-Value cache serialization

    Handles encoding/decoding of past_key_values tuples for efficient
    transmission in distributed inference.

    KV-cache structure:
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
        - Outer tuple: one per layer
        - Inner tuple: (key_tensor, value_tensor)
        - key_tensor shape: [batch_size, num_heads, seq_len, head_dim]
        - value_tensor shape: [batch_size, num_heads, seq_len, head_dim]

    Example:
        >>> codec = KVCacheCodec()
        >>> # Encode
        >>> kv_entries = codec.encode(past_key_values)
        >>> # Decode
        >>> past_key_values = codec.decode(kv_entries)
    """

    def __init__(self, tensor_codec: Optional[TensorCodec] = None):
        """
        Args:
            tensor_codec: TensorCodec instance (creates new if None)
        """
        self.tensor_codec = tensor_codec or TensorCodec()

    def encode(
        self,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]
    ) -> List:
        """
        Encode past_key_values to protobuf KVCacheEntry list

        Args:
            past_key_values: Tuple of (key, value) pairs for each layer

        Returns:
            List of KVCacheEntry protobuf messages
        """
        if past_key_values is None:
            return []

        from splitlearn_comm.protocol import compute_service_pb2

        kv_entries = []

        for layer_idx, (key_tensor, value_tensor) in enumerate(past_key_values):
            # 验证 key 和 value 形状
            if key_tensor.shape != value_tensor.shape:
                raise ValueError(
                    f"Layer {layer_idx}: Key and Value tensors must have the same shape. "
                    f"Got key={key_tensor.shape}, value={value_tensor.shape}"
                )

            # 验证形状是否符合预期的 KV-cache 格式
            # 期望形状: [batch_size, num_heads, seq_len, head_dim]
            if len(key_tensor.shape) != 4:
                logger.warning(
                    f"Layer {layer_idx}: Expected 4D tensor [batch, num_heads, seq_len, head_dim], "
                    f"got shape {key_tensor.shape}"
                )

            # Encode key tensor
            key_data, key_shape = self.tensor_codec.encode(key_tensor)

            # Encode value tensor
            value_data, value_shape = self.tensor_codec.encode(value_tensor)

            # Create protobuf entry
            entry = compute_service_pb2.KVCacheEntry(
                key_data=key_data,
                key_shape=list(key_shape),
                value_data=value_data,
                value_shape=list(value_shape)
            )

            kv_entries.append(entry)

        logger.debug(f"Encoded {len(kv_entries)} KV-cache layers")
        return kv_entries

    def decode(
        self,
        kv_entries: List
    ) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
        """
        Decode KVCacheEntry list to past_key_values tuple

        Args:
            kv_entries: List of KVCacheEntry protobuf messages

        Returns:
            Tuple of (key, value) pairs for each layer, or None if empty
        """
        if not kv_entries:
            return None

        past_key_values = []

        for layer_idx, entry in enumerate(kv_entries):
            # 验证形状字段存在
            if not entry.key_shape or not entry.value_shape:
                raise ValueError(
                    f"Layer {layer_idx}: Missing shape information in KVCacheEntry"
                )

            key_shape = tuple(entry.key_shape)
            value_shape = tuple(entry.value_shape)

            # 验证 key 和 value 形状一致
            if key_shape != value_shape:
                raise ValueError(
                    f"Layer {layer_idx}: Key and Value shapes must match. "
                    f"Got key={key_shape}, value={value_shape}"
                )

            # Decode key tensor
            key_tensor = self.tensor_codec.decode(
                entry.key_data,
                key_shape
            )

            # Decode value tensor
            value_tensor = self.tensor_codec.decode(
                entry.value_data,
                value_shape
            )

            past_key_values.append((key_tensor, value_tensor))

        logger.debug(f"Decoded {len(past_key_values)} KV-cache layers")
        return tuple(past_key_values)

    def estimate_size(
        self,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]
    ) -> int:
        """
        Estimate serialized size in bytes

        Args:
            past_key_values: KV-cache to estimate

        Returns:
            Estimated size in bytes
        """
        if past_key_values is None:
            return 0

        total_size = 0
        for key_tensor, value_tensor in past_key_values:
            # Each tensor: num_elements * bytes_per_element
            total_size += key_tensor.numel() * key_tensor.element_size()
            total_size += value_tensor.numel() * value_tensor.element_size()

        return total_size

    def __repr__(self) -> str:
        return f"KVCacheCodec(tensor_codec={self.tensor_codec})"
