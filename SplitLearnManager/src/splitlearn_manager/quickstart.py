"""
Quickstart API for SplitLearnManager

This module provides simplified, high-level APIs for quickly setting up
managed clients and servers with integrated model management.

Example:
    Server:
        >>> from splitlearn_manager.quickstart import ManagedServer
        >>> server = ManagedServer("gpt2", port=50051)
        >>> server.start()

    Client:
        >>> from splitlearn_manager.quickstart import ManagedClient
        >>> client = ManagedClient("localhost:50051")
        >>> result = client.infer("Hello, world!")
"""

import asyncio
from typing import Optional, Dict, Any
import logging

import torch

from .config import ModelConfig, ServerConfig
from .server import AsyncManagedServer

logger = logging.getLogger(__name__)


class ManagedServer:
    """
    Simplified managed server for quick setup.

    This provides an easy-to-use interface for:
    - Loading models automatically
    - Starting async managed server
    - Managing lifecycle

    Args:
        model_type: Type of model to serve (e.g., "gpt2", "qwen2")
        model_path: Path or name of the model (default: same as model_type)
        component: Model component to serve ("bottom", "trunk", or "top")
        port: Port to listen on (default: 50051)
        host: Host address (default: "0.0.0.0")
        device: Device to use (default: auto-detect)
        max_models: Maximum number of models to manage (default: 5)
        max_workers: Maximum number of worker threads (default: 1 for single-threaded)

    Example:
        Basic usage:
        >>> server = ManagedServer("gpt2", port=50051)
        >>> server.start()  # Blocking

        With context manager:
        >>> with ManagedServer("gpt2", port=50051) as server:
        ...     server.wait()  # Blocking
    """

    def __init__(
        self,
        model_type: str,
        model_path: Optional[str] = None,
        component: str = "trunk",
        port: int = 50051,
        host: str = "0.0.0.0",
        device: Optional[str] = None,
        max_models: int = 5,
        max_workers: int = 1,
        **model_kwargs
    ):
        self.model_type = model_type
        self.model_path = model_path or model_type
        self.component = component
        self.port = port
        self.host = host
        self.max_models = max_models

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Store additional model kwargs
        self.model_kwargs = model_kwargs

        # Create server config
        self.server_config = ServerConfig(
            host=self.host,
            port=self.port,
            max_models=self.max_models,
            max_workers=max_workers
        )

        # Create async server (will be initialized in start())
        self._async_server: Optional[AsyncManagedServer] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server_task: Optional[asyncio.Task] = None

        logger.info(
            f"ManagedServer initialized: "
            f"model={self.model_type}, "
            f"component={self.component}, "
            f"port={self.port}, "
            f"device={self.device}, "
            f"max_workers={max_workers}"
        )

    def start(self):
        """
        Start the server and load the model.

        This is a blocking call that runs until interrupted.
        """
        try:
            asyncio.run(self._async_start())
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")

    async def _async_start(self):
        """Internal async start method."""
        # Create async server (但不启动 gRPC)
        # 注意：PyTorch 线程配置会在模型加载时自动设置（在 load_from_config 中）
        self._async_server = AsyncManagedServer(config=self.server_config)

        # 先加载模型（在启动 gRPC 服务器之前）
        # component 和其他参数放在 config 字典中
        # 默认 split_points 用于 gpt2
        default_split_points = self.model_kwargs.get("split_points", [2, 10])
        model_config_dict = {
            "component": self.component,
            "split_points": default_split_points,
            "cache_dir": self.model_kwargs.get("cache_dir", "./models"),
            **{k: v for k, v in self.model_kwargs.items() if k not in ["split_points", "cache_dir"]}
        }
        model_config = ModelConfig(
            model_id=f"{self.model_type}_{self.component}",
            model_path=self.model_path,
            model_type=self.model_type,
            device=self.device,
            config=model_config_dict
        )

        logger.info(f"Loading model: {model_config.model_id} (before starting gRPC server)...")
        await self._async_server.load_model(model_config)
        logger.info("✓ Model loaded successfully")

        # 模型加载完成后，再启动 gRPC 服务器
        logger.info("Starting gRPC server...")
        await self._async_server.start()
        logger.info(f"✓ Server running on {self.host}:{self.port}. Press Ctrl+C to stop.")

        # Wait for termination
        await self._async_server.wait_for_termination()

    def start_background(self):
        """
        Start the server in the background (non-blocking).

        Returns:
            asyncio event loop running the server

        Note:
            You need to manage the event loop yourself when using this method.
        """
        if self._loop is None:
            self._loop = asyncio.new_event_loop()

        def run_server():
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._async_start())

        import threading
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()

        logger.info("Server started in background thread")
        return self._loop

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Cleanup if needed
        return False


class ManagedClient:
    """
    Simplified managed client for connecting to ManagedServer.

    This provides an easy-to-use interface for:
    - Connecting to managed servers
    - Sending inference requests
    - Handling responses

    Args:
        server_address: Server address (e.g., "localhost:50051")
        timeout: Request timeout in seconds (default: 30.0)

    Example:
        >>> client = ManagedClient("localhost:50051")
        >>> result = client.compute(input_tensor)
    """

    def __init__(
        self,
        server_address: str,
        timeout: float = 30.0
    ):
        # Import here to avoid circular dependency
        from splitlearn_comm.quickstart import Client as GRPCClient

        self.server_address = server_address
        self.timeout = timeout

        # Create underlying gRPC client
        self._client = GRPCClient(
            server_address=server_address,
            timeout=timeout
        )

        logger.info(f"ManagedClient connected to {server_address}")

    def compute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Send tensor for computation.

        Args:
            input_tensor: Input tensor

        Returns:
            Output tensor from server
        """
        return self._client.compute(input_tensor)

    def close(self):
        """Close the connection."""
        self._client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


# Convenience function
async def serve_model_async(
    model_type: str,
    model_path: Optional[str] = None,
    component: str = "trunk",
    port: int = 50051,
    **kwargs
):
    """
    Async convenience function to serve a model.

    Args:
        model_type: Type of model
        model_path: Path to model
        component: Model component ("bottom", "trunk", or "top")
        port: Port to listen on
        **kwargs: Additional arguments for model configuration

    Example:
        >>> import asyncio
        >>> asyncio.run(serve_model_async("gpt2", port=50051))
    """
    server_config = ServerConfig(host="0.0.0.0", port=port)
    async_server = AsyncManagedServer(config=server_config)

    # Start server
    await async_server.start()

    # Load model
    model_config = ModelConfig(
        model_id=f"{model_type}_{component}",
        model_type=model_type,
        component=component,
        model_name_or_path=model_path or model_type,
        **kwargs
    )

    await async_server.load_model(model_config)
    logger.info(f"Model {model_config.model_id} loaded and serving on port {port}")

    # Wait for termination
    await async_server.wait_for_termination()


__all__ = [
    "ManagedServer",
    "ManagedClient",
    "serve_model_async",
]
