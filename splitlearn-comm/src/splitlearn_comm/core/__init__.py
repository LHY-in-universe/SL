"""Core abstractions for splitlearn-comm."""

from .compute_function import ComputeFunction, ModelComputeFunction
from .tensor_codec import TensorCodec, CompressedTensorCodec

__all__ = [
    "ComputeFunction",
    "ModelComputeFunction",
    "TensorCodec",
    "CompressedTensorCodec",
]
