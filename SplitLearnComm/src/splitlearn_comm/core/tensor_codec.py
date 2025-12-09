"""
TensorCodec - Tensor 编解码器

提供高效的 Tensor 序列化和反序列化功能。
"""

from typing import Tuple, Optional, Union
import numpy as np
import torch


class TensorCodec:
    """
    Tensor 编解码器

    使用优化的 bytes 格式进行序列化，比 protobuf 的 repeated float 性能更好。
    支持可选的精度参数以减少带宽使用。

    Example:
        >>> codec = TensorCodec()
        >>> tensor = torch.randn(1, 10, 768)
        >>> data, shape = codec.encode(tensor)
        >>> restored = codec.decode(data, shape)
        >>> assert torch.allclose(tensor, restored)

        >>> # 使用 float16 以减少带宽
        >>> codec_fp16 = TensorCodec(dtype=np.float16)
        >>> data, shape = codec_fp16.encode(tensor)  # 减少 50% 带宽
    """

    def __init__(self, dtype: Optional[np.dtype] = None):
        """
        Args:
            dtype: numpy dtype for serialization (default: np.float32)
                   可选: np.float16 (减少带宽 50%), np.float32 (默认)
        """
        self.dtype = dtype or np.float32

        # 计算每个元素的字节数
        self.bytes_per_element = np.dtype(self.dtype).itemsize

    def encode(self, tensor: torch.Tensor, dtype: Optional[np.dtype] = None) -> Tuple[bytes, Tuple[int, ...]]:
        """
        将 Tensor 编码为 bytes

        Args:
            tensor: 输入张量
            dtype: 可选的覆盖 dtype（如果不指定，使用构造函数中的 dtype）

        Returns:
            (data, shape) 元组
            - data: 序列化后的字节数据
            - shape: 张量形状
        """
        # 使用指定的 dtype 或默认 dtype
        use_dtype = dtype or self.dtype

        # 转换为 numpy array 并序列化
        array = tensor.cpu().numpy().astype(use_dtype)
        data = array.tobytes()
        shape = tuple(tensor.shape)

        return data, shape

    def decode(self, data: bytes, shape: Tuple[int, ...], dtype: Optional[np.dtype] = None) -> torch.Tensor:
        """
        从 bytes 解码为 Tensor

        Args:
            data: 字节数据
            shape: 张量形状
            dtype: 可选的覆盖 dtype（如果不指定，使用构造函数中的 dtype）

        Returns:
            解码后的张量（float32）
        """
        # 使用指定的 dtype 或默认 dtype
        use_dtype = dtype or self.dtype

        # 反序列化为 numpy array
        array = np.frombuffer(data, dtype=use_dtype)
        array = array.reshape(shape)

        # 转换为 PyTorch tensor (统一转为 float32 以保证精度)
        # 复制数组以确保可写性，避免警告
        tensor = torch.from_numpy(array.copy()).float()

        return tensor

    def get_data_size(self, tensor: torch.Tensor, dtype: Optional[np.dtype] = None) -> int:
        """
        计算序列化后的数据大小（字节）

        Args:
            tensor: 输入张量
            dtype: 可选的覆盖 dtype（如果不指定，使用构造函数中的 dtype）

        Returns:
            字节数
        """
        use_dtype = dtype or self.dtype
        bytes_per_elem = np.dtype(use_dtype).itemsize
        return tensor.numel() * bytes_per_elem


class CompressedTensorCodec(TensorCodec):
    """
    压缩的 Tensor 编解码器（可选）

    在网络带宽受限的场景下，可以使用压缩来减少传输数据量。
    注意：压缩/解压会增加 CPU 开销。

    Example:
        >>> codec = CompressedTensorCodec(compression_level=6)
        >>> data, shape = codec.encode(tensor)
        >>> restored = codec.decode(data, shape)
    """

    def __init__(self, compression_level: int = 6, dtype: Optional[np.dtype] = None):
        """
        Args:
            compression_level: 压缩级别 (1-9，越高压缩率越高但速度越慢)
            dtype: numpy dtype for serialization (default: np.float32)
        """
        super().__init__(dtype=dtype)
        self.compression_level = compression_level

        try:
            import zlib
            self.zlib = zlib
        except ImportError:
            raise ImportError("zlib is required for compression")

    def encode(self, tensor: torch.Tensor) -> Tuple[bytes, Tuple[int, ...]]:
        """编码并压缩 Tensor"""
        data, shape = super().encode(tensor)
        compressed = self.zlib.compress(data, level=self.compression_level)
        return compressed, shape

    def decode(self, data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
        """解压并解码 Tensor"""
        decompressed = self.zlib.decompress(data)
        return super().decode(decompressed, shape)
