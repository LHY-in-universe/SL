"""
Tests for TensorCodec and CompressedTensorCodec.
"""

import pytest
import torch
import numpy as np

from splitlearn_comm.core import TensorCodec, CompressedTensorCodec


class TestTensorCodec:
    """Tests for TensorCodec."""

    @pytest.fixture
    def codec(self):
        """Fixture providing a TensorCodec instance."""
        return TensorCodec()

    def test_encode_decode_roundtrip(self, codec, sample_tensor):
        """Test encode-decode roundtrip preserves tensor."""
        # Encode
        data, shape = codec.encode(sample_tensor)

        # Decode
        reconstructed = codec.decode(data, shape)

        # Should be identical
        assert torch.allclose(reconstructed, sample_tensor)
        assert reconstructed.shape == sample_tensor.shape

    def test_encode_output_types(self, codec, sample_tensor):
        """Test encode returns correct types."""
        data, shape = codec.encode(sample_tensor)

        assert isinstance(data, bytes)
        assert isinstance(shape, tuple)
        assert all(isinstance(s, int) for s in shape)

    def test_decode_output_type(self, codec, sample_tensor):
        """Test decode returns torch.Tensor."""
        data, shape = codec.encode(sample_tensor)
        reconstructed = codec.decode(data, shape)

        assert isinstance(reconstructed, torch.Tensor)

    def test_different_shapes(self, codec, sample_tensors):
        """Test encoding/decoding tensors with different shapes."""
        for tensor in sample_tensors:
            data, shape = codec.encode(tensor)
            reconstructed = codec.decode(data, shape)

            assert torch.allclose(reconstructed, tensor)
            assert reconstructed.shape == tensor.shape

    def test_data_size(self, codec):
        """Test encoded data has correct size."""
        tensor = torch.randn(10, 20, 30)

        data, shape = codec.encode(tensor)

        # Size should be num_elements * 4 bytes (float32)
        expected_size = 10 * 20 * 30 * 4
        assert len(data) == expected_size

    def test_shape_preservation(self, codec):
        """Test shape is correctly preserved."""
        shapes = [
            (1,),
            (10,),
            (5, 5),
            (2, 3, 4),
            (1, 1, 1, 1),
            (10, 20, 30, 40)
        ]

        for original_shape in shapes:
            tensor = torch.randn(*original_shape)
            data, shape = codec.encode(tensor)

            assert shape == original_shape

    def test_dtype_conversion(self, codec):
        """Test that tensors are converted to float32."""
        # Different dtypes
        tensors = [
            torch.randn(5, 5).double(),  # float64
            torch.randn(5, 5).half(),     # float16
            torch.randint(0, 10, (5, 5)).float(),  # int -> float
        ]

        for tensor in tensors:
            data, shape = codec.encode(tensor)
            reconstructed = codec.decode(data, shape)

            # Reconstructed should be float32
            assert reconstructed.dtype == torch.float32

    def test_gpu_tensor_encoding(self, codec):
        """Test encoding tensor from GPU (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        gpu_tensor = torch.randn(5, 5).cuda()
        data, shape = codec.encode(gpu_tensor)
        reconstructed = codec.decode(data, shape)

        # Should work correctly
        assert torch.allclose(reconstructed.cpu(), gpu_tensor.cpu())

    def test_zero_tensor(self, codec):
        """Test encoding all-zeros tensor."""
        tensor = torch.zeros(10, 10)
        data, shape = codec.encode(tensor)
        reconstructed = codec.decode(data, shape)

        assert torch.allclose(reconstructed, tensor)

    def test_ones_tensor(self, codec):
        """Test encoding all-ones tensor."""
        tensor = torch.ones(10, 10)
        data, shape = codec.encode(tensor)
        reconstructed = codec.decode(data, shape)

        assert torch.allclose(reconstructed, tensor)

    def test_extreme_values(self, codec):
        """Test encoding tensors with extreme values."""
        tensors = [
            torch.tensor([1e-10, 1e10]),
            torch.tensor([-1e10, -1e-10]),
            torch.tensor([float('inf'), float('-inf'), 0.0])
        ]

        for tensor in tensors:
            data, shape = codec.encode(tensor)
            reconstructed = codec.decode(data, shape)

            # Note: inf values should be preserved
            assert torch.equal(reconstructed, tensor.float())

    def test_scalar_tensor(self, codec):
        """Test encoding scalar tensor."""
        tensor = torch.tensor(3.14)
        data, shape = codec.encode(tensor)
        reconstructed = codec.decode(data, shape)

        assert torch.allclose(reconstructed, tensor)
        assert reconstructed.shape == tensor.shape

    def test_empty_tensor(self, codec):
        """Test encoding empty tensor."""
        tensor = torch.tensor([])
        data, shape = codec.encode(tensor)
        reconstructed = codec.decode(data, shape)

        assert reconstructed.shape == tensor.shape
        assert len(data) == 0


class TestCompressedTensorCodec:
    """Tests for CompressedTensorCodec."""

    @pytest.fixture
    def codec(self):
        """Fixture providing a CompressedTensorCodec instance."""
        return CompressedTensorCodec(compression_level=6)

    def test_encode_decode_roundtrip(self, codec, sample_tensor):
        """Test encode-decode roundtrip with compression."""
        data, shape = codec.encode(sample_tensor)
        reconstructed = codec.decode(data, shape)

        assert torch.allclose(reconstructed, sample_tensor)
        assert reconstructed.shape == sample_tensor.shape

    def test_compression_reduces_size(self):
        """Test that compression reduces data size for compressible data."""
        # Create compressible tensor (lots of repeated values)
        tensor = torch.zeros(1000, 1000)
        tensor[:, ::2] = 1.0  # Pattern

        # Compare compressed vs uncompressed
        compressed_codec = CompressedTensorCodec(compression_level=9)
        uncompressed_codec = TensorCodec()

        compressed_data, _ = compressed_codec.encode(tensor)
        uncompressed_data, _ = uncompressed_codec.encode(tensor)

        # Compressed should be smaller
        assert len(compressed_data) < len(uncompressed_data)

    def test_compression_levels(self, sample_tensor):
        """Test different compression levels."""
        levels = [0, 1, 6, 9]
        sizes = []

        for level in levels:
            codec = CompressedTensorCodec(compression_level=level)
            data, shape = codec.encode(sample_tensor)
            sizes.append(len(data))

            # Verify correctness
            reconstructed = codec.decode(data, shape)
            assert torch.allclose(reconstructed, sample_tensor)

        # Higher compression should generally give smaller size
        # (though not guaranteed for random data)

    def test_different_shapes(self, codec, sample_tensors):
        """Test compressed codec with different shapes."""
        for tensor in sample_tensors:
            data, shape = codec.encode(tensor)
            reconstructed = codec.decode(data, shape)

            assert torch.allclose(reconstructed, tensor)

    def test_random_data_compression(self):
        """Test compression with random (incompressible) data."""
        tensor = torch.randn(100, 100)

        compressed_codec = CompressedTensorCodec(compression_level=9)
        uncompressed_codec = TensorCodec()

        compressed_data, _ = compressed_codec.encode(tensor)
        uncompressed_data, _ = uncompressed_codec.encode(tensor)

        # Random data may not compress well
        # But should still work correctly
        reconstructed, _ = compressed_codec.encode(tensor)
        assert isinstance(reconstructed, bytes)


class TestCodecEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_invalid_shape_decode(self):
        """Test decoding with mismatched shape."""
        codec = TensorCodec()

        tensor = torch.randn(2, 3, 4)
        data, _ = codec.encode(tensor)

        # Try to decode with wrong shape
        wrong_shape = (3, 2, 4)  # Same num elements, different shape

        # This should work (reshape), but give different tensor
        reconstructed = codec.decode(data, wrong_shape)
        assert reconstructed.shape == wrong_shape

    def test_invalid_data_size(self):
        """Test decoding with data size not matching shape."""
        codec = TensorCodec()

        # Create fake data
        data = b'\x00' * 100  # 25 float32 values

        # Shape requires more data than available
        shape = (10, 10)  # Would need 400 bytes

        with pytest.raises(Exception):  # Could be ValueError or similar
            codec.decode(data, shape)

    def test_corrupted_compressed_data(self):
        """Test decoding corrupted compressed data."""
        codec = CompressedTensorCodec()

        # Create corrupted data
        corrupted_data = b'\x00\x01\x02\x03'
        shape = (2, 2)

        with pytest.raises(Exception):  # Should raise decompression error
            codec.decode(corrupted_data, shape)

    def test_very_large_tensor(self):
        """Test codec with very large tensor."""
        codec = TensorCodec()

        # 10M elements
        large_tensor = torch.randn(100, 100, 1000)

        data, shape = codec.encode(large_tensor)
        reconstructed = codec.decode(data, shape)

        assert reconstructed.shape == large_tensor.shape
        # Use sampling for large tensor comparison
        assert torch.allclose(reconstructed[0, 0, :100], large_tensor[0, 0, :100])


class TestCodecCompatibility:
    """Tests for compatibility between codec instances."""

    def test_cross_codec_compatibility(self, sample_tensor):
        """Test that different codec instances can decode each other's output."""
        codec1 = TensorCodec()
        codec2 = TensorCodec()

        data, shape = codec1.encode(sample_tensor)
        reconstructed = codec2.decode(data, shape)

        assert torch.allclose(reconstructed, sample_tensor)

    def test_compressed_uncompressed_incompatibility(self, sample_tensor):
        """Test that compressed and uncompressed codecs are incompatible."""
        compressed_codec = CompressedTensorCodec()
        uncompressed_codec = TensorCodec()

        # Encode with compression
        data, shape = compressed_codec.encode(sample_tensor)

        # Try to decode without decompression - should fail
        with pytest.raises(Exception):
            uncompressed_codec.decode(data, shape)


class TestCodecPerformance:
    """Performance-related tests."""

    def test_encode_performance(self, benchmark=None):
        """Test encoding performance (requires pytest-benchmark)."""
        if benchmark is None:
            pytest.skip("pytest-benchmark not available")

        codec = TensorCodec()
        tensor = torch.randn(100, 100, 10)

        result = benchmark(codec.encode, tensor)

    def test_decode_performance(self, benchmark=None):
        """Test decoding performance (requires pytest-benchmark)."""
        if benchmark is None:
            pytest.skip("pytest-benchmark not available")

        codec = TensorCodec()
        tensor = torch.randn(100, 100, 10)
        data, shape = codec.encode(tensor)

        result = benchmark(codec.decode, data, shape)
