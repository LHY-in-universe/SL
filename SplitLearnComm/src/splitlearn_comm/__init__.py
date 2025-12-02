"""
splitlearn-comm - A gRPC-based communication library for distributed computing

This library provides a simple and efficient way to distribute computation
across multiple machines using gRPC, with full abstraction from specific models.

Example:
    Server:
        >>> from splitlearn_comm import GRPCComputeServer
        >>> from splitlearn_comm.core import ModelComputeFunction
        >>>
        >>> compute_fn = ModelComputeFunction(model, device="cuda")
        >>> server = GRPCComputeServer(compute_fn, port=50051)
        >>> server.start()
        >>> server.wait_for_termination()

    Client:
        >>> from splitlearn_comm import GRPCComputeClient
        >>>
        >>> client = GRPCComputeClient("localhost:50051")
        >>> client.connect()
        >>> output = client.compute(input_tensor)
        >>> client.close()
"""

from .__version__ import __version__

# Core abstractions
from .core import (
    ComputeFunction,
    ModelComputeFunction,
    TensorCodec,
    CompressedTensorCodec,
)

# Async core abstractions
from .core.async_compute_function import (
    AsyncComputeFunction,
    AsyncModelComputeFunction,
    AsyncLambdaComputeFunction,
    AsyncChainComputeFunction,
)

# Server
from .server import (
    ComputeServicer,
    GRPCComputeServer,
    serve,
)

# Async server
from .server.async_grpc_server import (
    AsyncGRPCComputeServer,
    serve_async,
)
from .server.async_servicer import AsyncComputeServicer

# Client
from .client import (
    RetryStrategy,
    ExponentialBackoff,
    FixedDelay,
    GRPCComputeClient,
)

# UI (optional - requires gradio)
try:
    from .ui import ClientUI, ServerMonitoringUI
    _has_ui = True
except ImportError:
    _has_ui = False
    ClientUI = None
    ServerMonitoringUI = None

__all__ = [
    # Version
    "__version__",

    # Core
    "ComputeFunction",
    "ModelComputeFunction",
    "TensorCodec",
    "CompressedTensorCodec",

    # Async Core
    "AsyncComputeFunction",
    "AsyncModelComputeFunction",
    "AsyncLambdaComputeFunction",
    "AsyncChainComputeFunction",

    # Server
    "ComputeServicer",
    "GRPCComputeServer",
    "serve",

    # Async Server
    "AsyncGRPCComputeServer",
    "AsyncComputeServicer",
    "serve_async",

    # Client
    "RetryStrategy",
    "ExponentialBackoff",
    "FixedDelay",
    "GRPCComputeClient",

    # UI (optional)
    "ClientUI",
    "ServerMonitoringUI",
]
