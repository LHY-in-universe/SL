"""
TensorCodec - Tensor 编解码器

提供高效的 Tensor 序列化和反序列化功能。
"""

from typing import Tuple
import numpy as np
import torch


class TensorCodec:
    """
    Tensor 编解码器

    使用优化的 bytes 格式进行序列化，比 protobuf 的 repeated float 性能更好。

    Example:
        >>> codec = TensorCodec()
        >>> tensor = torch.randn(1, 10, 768)
        >>> data, shape = codec.encode(tensor)
        >>> restored = codec.decode(data, shape)
        >>> assert torch.allclose(tensor, restored)
    """

    @staticmethod
    def encode(tensor: torch.Tensor) -> Tuple[bytes, Tuple[int, ...]]:
        """
        将 Tensor 编码为 bytes

        Args:
            tensor: 输入张量

        Returns:
            (data, shape) 元组
            - data: 序列化后的字节数据
            - shape: 张量形状
        """
        # 转换为 numpy array 并序列化
        array = tensor.cpu().numpy().astype(np.float32)
        data = array.tobytes()
        shape = tuple(tensor.shape)

        return data, shape

    @staticmethod
    def decode(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        从 bytes 解码为 Tensor

        Args:
            data: 字节数据
            shape: 张量形状

        Returns:
            解码后的张量
        """
        # 反序列化为 numpy array
        array = np.frombuffer(data, dtype=np.float32)
        array = array.reshape(shape)

        # 转换为 PyTorch tensor
        tensor = torch.from_numpy(array)

        return tensor

    @staticmethod
    def get_data_size(tensor: torch.Tensor) -> int:
        """
        计算序列化后的数据大小（字节）

        Args:
            tensor: 输入张量

        Returns:
            字节数
        """
        return tensor.numel() * 4  # float32 = 4 bytes


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

    def __init__(self, compression_level: int = 6):
        """
        Args:
            compression_level: 压缩级别 (1-9，越高压缩率越高但速度越慢)
        """
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
