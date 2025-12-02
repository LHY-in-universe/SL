"""
异步 gRPC 计算服务器。

使用 grpc.aio 提供非阻塞的 gRPC 服务。
"""

import asyncio
import logging
from typing import Optional

import grpc.aio

from ..protocol import compute_service_pb2_grpc
from ..core.async_compute_function import AsyncComputeFunction
from .async_servicer import AsyncComputeServicer

logger = logging.getLogger(__name__)


class AsyncGRPCComputeServer:
    """
    异步 gRPC 计算服务器。

    使用 grpc.aio 提供非阻塞的 gRPC 服务，支持高并发。

    主要特性：
    - 异步 I/O：不阻塞事件循环
    - 高并发：可以同时处理多个请求
    - 资源高效：相比线程池，协程开销更小
    """

    def __init__(
        self,
        compute_fn: AsyncComputeFunction,
        host: str = "0.0.0.0",
        port: int = 50051,
        max_message_length: int = 100 * 1024 * 1024,  # 100MB
        options: Optional[list] = None
    ):
        """
        初始化异步 gRPC 服务器。

        Args:
            compute_fn: 异步计算函数
            host: 监听地址
            port: 监听端口
            max_message_length: 最大消息长度（字节）
            options: 额外的 gRPC 选项
        """
        self.compute_fn = compute_fn
        self.host = host
        self.port = port
        self.address = f"{host}:{port}"

        # 创建异步 Servicer
        self.servicer = AsyncComputeServicer(compute_fn=compute_fn)

        # 设置 gRPC 选项
        default_options = [
            ("grpc.max_send_message_length", max_message_length),
            ("grpc.max_receive_message_length", max_message_length),
            ("grpc.keepalive_time_ms", 30000),  # 30 seconds
            ("grpc.keepalive_timeout_ms", 10000),  # 10 seconds
            ("grpc.keepalive_permit_without_calls", True),
            ("grpc.http2.max_pings_without_data", 0),
        ]

        if options:
            default_options.extend(options)

        # 创建异步 gRPC 服务器
        # 注意：grpc.aio.server() 不需要显式的线程池
        self.server = grpc.aio.server(options=default_options)

        # 添加 Servicer 到服务器
        compute_service_pb2_grpc.add_ComputeServiceServicer_to_server(
            self.servicer,
            self.server
        )

        # 添加不安全的端口（在生产环境应该使用 TLS）
        self.server.add_insecure_port(self.address)

        logger.info(
            f"AsyncGRPCComputeServer initialized at {self.address} "
            f"(max_message_length={max_message_length / 1024 / 1024:.1f}MB)"
        )

    async def start(self):
        """
        异步启动服务器。

        这是一个协程，需要在事件循环中运行。
        """
        # 调用计算函数的 setup 方法（如果有）
        await self.compute_fn.setup()

        # 启动 gRPC 服务器
        await self.server.start()

        logger.info(f"AsyncGRPCComputeServer started on {self.address}")

    async def wait_for_termination(self):
        """
        等待服务器终止。

        这是一个阻塞操作，会一直等待直到服务器被停止。
        """
        await self.server.wait_for_termination()

    async def stop(self, grace: Optional[float] = 5.0):
        """
        异步停止服务器。

        Args:
            grace: 优雅关闭的超时时间（秒）
        """
        logger.info(f"Stopping AsyncGRPCComputeServer (grace={grace}s)...")

        # 停止服务器（带超时）
        await self.server.stop(grace)

        # 调用计算函数的 teardown 方法（如果有）
        await self.compute_fn.teardown()

        logger.info("AsyncGRPCComputeServer stopped")

    async def serve(self):
        """
        启动并运行服务器，直到被停止。

        这是一个便捷方法，组合了 start() 和 wait_for_termination()。
        """
        await self.start()
        await self.wait_for_termination()

    async def get_statistics(self) -> dict:
        """
        获取服务器统计信息。

        Returns:
            统计信息字典
        """
        servicer_stats = await self.servicer.get_statistics()
        return {
            "address": self.address,
            "status": "running",
            **servicer_stats,
        }

    def __enter__(self):
        """不支持同步上下文管理器。"""
        raise TypeError(
            "AsyncGRPCComputeServer requires async context manager. "
            "Use 'async with' instead of 'with'."
        )

    async def __aenter__(self):
        """异步上下文管理器入口。"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出。"""
        await self.stop()
        return False


# 便捷函数

async def serve_async(
    compute_fn: AsyncComputeFunction,
    host: str = "0.0.0.0",
    port: int = 50051,
    **kwargs
):
    """
    便捷函数：创建并运行异步 gRPC 服务器。

    Args:
        compute_fn: 异步计算函数
        host: 监听地址
        port: 监听端口
        **kwargs: 传递给 AsyncGRPCComputeServer 的其他参数

    Example:
        ```python
        import asyncio
        from splitlearn_comm.server import serve_async
        from splitlearn_comm.core import AsyncModelComputeFunction

        async def main():
            # 创建计算函数
            model = YourModel()
            compute_fn = AsyncModelComputeFunction(model)

            # 启动服务器
            await serve_async(compute_fn, port=50051)

        asyncio.run(main())
        ```
    """
    server = AsyncGRPCComputeServer(
        compute_fn=compute_fn,
        host=host,
        port=port,
        **kwargs
    )

    await server.serve()


if __name__ == "__main__":
    # 简单的测试示例
    import torch

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x * 2

    from ..core.async_compute_function import AsyncModelComputeFunction

    async def main():
        model = DummyModel()
        compute_fn = AsyncModelComputeFunction(model, device="cpu")

        logger.info("Starting test server...")
        await serve_async(compute_fn, port=50051)

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
