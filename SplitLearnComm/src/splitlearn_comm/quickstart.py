"""
Quickstart API for SplitLearnComm

This module provides simplified, high-level APIs for quick setup of gRPC
clients and servers, with sensible defaults and minimal configuration.

Example:
    Client usage:
        >>> from splitlearn_comm.quickstart import Client
        >>> client = Client("localhost:50051")
        >>> output = client.compute(input_tensor)

    Server usage:
        >>> from splitlearn_comm.quickstart import Server
        >>> server = Server(model=my_model, port=50051)
        >>> server.start()
"""

from typing import Optional, Union
import logging

import torch
import torch.nn as nn

from .client import GRPCComputeClient, RetryStrategy, ExponentialBackoff
from .server import GRPCComputeServer
from .core import ModelComputeFunction

logger = logging.getLogger(__name__)


class Client:
    """
    Simplified gRPC client with automatic connection management.

    This is a high-level wrapper around GRPCComputeClient with:
    - Automatic connection on initialization
    - Sensible default retry strategy
    - Simplified API

    Args:
        server_address: Server address in format "host:port" (e.g., "localhost:50051")
        max_retries: Maximum number of retry attempts (default: 5)
        timeout: Request timeout in seconds (default: 30.0)
        auto_connect: Automatically connect on initialization (default: True)

    Example:
        >>> client = Client("localhost:50051")
        >>> output = client.compute(input_tensor)
        >>> # Client automatically closes on deletion
    """

    def __init__(
        self,
        server_address: str,
        max_retries: int = 5,
        timeout: float = 30.0,
        auto_connect: bool = True
    ):
        # Create default retry strategy
        retry_strategy = ExponentialBackoff(
            max_retries=max_retries,
            initial_delay=1.0,
            max_delay=60.0
        )

        # Create underlying client
        self._client = GRPCComputeClient(
            server_address=server_address,
            retry_strategy=retry_strategy,
            timeout=timeout
        )

        # Auto-connect if requested
        if auto_connect:
            self.connect()

    def connect(self):
        """Connect to the server."""
        self._client.connect()
        logger.info(f"Connected to server at {self._client.server_address}")

    def compute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Send tensor for computation and receive result.

        Args:
            input_tensor: Input tensor to compute

        Returns:
            Output tensor from server

        Raises:
            RuntimeError: If computation fails
        """
        return self._client.compute(input_tensor)

    def close(self):
        """Close the connection."""
        self._client.close()
        logger.info("Client connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically close connection."""
        self.close()
        return False

    def __del__(self):
        """Automatically close connection on deletion."""
        try:
            self.close()
        except:
            pass


class Server:
    """
    Simplified gRPC server with automatic setup.

    This is a high-level wrapper around GRPCComputeServer with:
    - Automatic model wrapping
    - Sensible default configuration
    - Simplified startup

    Args:
        model: PyTorch model to serve (nn.Module)
        port: Port to listen on (default: 50051)
        host: Host address to bind to (default: "0.0.0.0")
        device: Device to run model on (default: "cuda" if available, else "cpu")
        max_workers: Maximum number of worker threads (default: 10)

    Example:
        >>> server = Server(model=my_model, port=50051)
        >>> server.start()
        >>> server.wait_for_termination()

        Or with context manager:
        >>> with Server(model=my_model, port=50051) as server:
        ...     server.wait_for_termination()
    """

    def __init__(
        self,
        model: nn.Module,
        port: int = 50051,
        host: str = "0.0.0.0",
        device: Optional[str] = None,
        max_workers: int = 10
    ):
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Wrap model in compute function
        compute_fn = ModelComputeFunction(model, device=device)

        # Create underlying server
        self._server = GRPCComputeServer(
            compute_fn=compute_fn,
            port=port,
            host=host,
            max_workers=max_workers
        )

        self.host = host
        self.port = port
        self.device = device

    def start(self):
        """Start the server."""
        self._server.start()
        logger.info(f"Server started on {self.host}:{self.port} (device: {self.device})")

    def stop(self, grace: Optional[float] = 5.0):
        """
        Stop the server.

        Args:
            grace: Grace period for shutting down (seconds)
        """
        self._server.stop(grace=grace)
        logger.info("Server stopped")

    def wait_for_termination(self):
        """Wait for server to terminate (blocking)."""
        self._server.wait_for_termination()

    def __enter__(self):
        """Context manager entry - automatically start server."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically stop server."""
        self.stop()
        return False


# Convenience function
def serve(
    model: nn.Module,
    port: int = 50051,
    host: str = "0.0.0.0",
    device: Optional[str] = None
):
    """
    Start a server and wait for termination (blocking).

    This is the simplest way to serve a model:

    Example:
        >>> from splitlearn_comm.quickstart import serve
        >>> serve(my_model, port=50051)  # Blocks until Ctrl+C

    Args:
        model: PyTorch model to serve
        port: Port to listen on
        host: Host address to bind to
        device: Device to run model on (auto-detected if None)
    """
    with Server(model=model, port=port, host=host, device=device) as server:
        try:
            logger.info(f"Server running on {host}:{port}. Press Ctrl+C to stop.")
            server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")


__all__ = [
    "Client",
    "Server",
    "serve",
]
