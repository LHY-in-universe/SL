"""
GRPCComputeServer - gRPC 服务器实现
"""

import logging
import signal
import sys
from concurrent import futures
from typing import Optional

import grpc

from ..core import ComputeFunction, TensorCodec
from ..protocol import compute_service_pb2_grpc
from .servicer import ComputeServicer

logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_MAX_MESSAGE_LENGTH = 100 * 1024 * 1024  # 100 MB
DEFAULT_MAX_WORKERS = 10


class GRPCComputeServer:
    """
    gRPC 计算服务器

    提供简单的 API 来启动 gRPC 服务器，支持任意 ComputeFunction。

    Example:
        >>> from splitlearn_comm import GRPCComputeServer
        >>> from splitlearn_comm.core import ModelComputeFunction
        >>>
        >>> # 创建计算函数
        >>> compute_fn = ModelComputeFunction(model, device="cuda")
        >>>
        >>> # 创建并启动服务器
        >>> server = GRPCComputeServer(
        ...     compute_fn=compute_fn,
        ...     host="0.0.0.0",
        ...     port=50051
        ... )
        >>> server.start()
        >>> server.wait_for_termination()
    """

    def __init__(
        self,
        compute_fn: ComputeFunction,
        host: str = "0.0.0.0",
        port: int = 50051,
        max_workers: int = DEFAULT_MAX_WORKERS,
        max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
        codec: Optional[TensorCodec] = None,
        version: str = "1.0.0"
    ):
        """
        Args:
            compute_fn: 计算函数实例
            host: 服务器主机地址
            port: 服务器端口
            max_workers: 最大工作线程数
            max_message_length: 最大消息长度（字节）
            codec: Tensor 编解码器
            version: 服务版本号
        """
        self.compute_fn = compute_fn
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.max_message_length = max_message_length
        self.version = version

        # 创建 Servicer
        self.servicer = ComputeServicer(
            compute_fn=compute_fn,
            codec=codec,
            version=version
        )

        # 创建 gRPC 服务器
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[
                ("grpc.max_send_message_length", max_message_length),
                ("grpc.max_receive_message_length", max_message_length),
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
            ],
        )

        # 添加 Servicer
        compute_service_pb2_grpc.add_ComputeServiceServicer_to_server(
            self.servicer, self.server
        )

        # 服务器地址
        self.server_address = f"{host}:{port}"
        self.server.add_insecure_port(self.server_address)

        # 注册信号处理
        self._setup_signal_handlers()

        logger.info(f"GRPCComputeServer initialized at {self.server_address}")

    def _setup_signal_handlers(self):
        """设置信号处理器，支持优雅关闭"""
        def signal_handler(sig, frame):
            logger.info("Received shutdown signal, stopping server...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def start(self):
        """启动服务器"""
        self.server.start()
        logger.info(f"✓ Server started on {self.server_address}")
        logger.info(f"  Max workers: {self.max_workers}")
        logger.info(f"  Max message length: {self.max_message_length / (1024**2):.1f} MB")
        logger.info(f"  Version: {self.version}")
        logger.info("Waiting for requests... (Press Ctrl+C to stop)")

    def stop(self, grace: int = 5):
        """
        停止服务器

        Args:
            grace: 优雅关闭的等待时间（秒）
        """
        logger.info("Shutting down server...")
        self.servicer.shutdown()
        self.server.stop(grace=grace)
        logger.info("Server stopped")

    def wait_for_termination(self):
        """等待服务器终止"""
        try:
            self.server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("Server interrupted")
            self.stop()

    def __enter__(self):
        """上下文管理器支持"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器清理"""
        self.stop()

    def launch_monitoring_ui(
        self,
        theme: str = "default",
        refresh_interval: int = 2,
        share: bool = False,
        server_port: int = 7861,
        **kwargs
    ):
        """
        启动 Gradio 监控 UI

        这是一个便捷方法，用于快速启动 Gradio 监控仪表板。
        需要安装 gradio: pip install splitlearn-comm[ui]

        Args:
            theme: UI 主题 ("default", "dark", "light")
            refresh_interval: 仪表板刷新间隔（秒）
            share: 是否创建公共 Gradio 链接
            server_port: 监控 UI 端口
            **kwargs: 传递给 demo.launch() 的额外参数

        Example:
            >>> from splitlearn_comm import GRPCComputeServer
            >>> from splitlearn_comm.core import ModelComputeFunction
            >>>
            >>> # 创建并启动服务器
            >>> compute_fn = ModelComputeFunction(model)
            >>> server = GRPCComputeServer(compute_fn, port=50051)
            >>> server.start()
            >>>
            >>> # 启动监控 UI（在单独线程中运行）
            >>> server.launch_monitoring_ui(
            ...     share=False,
            ...     server_port=7861,
            ...     blocking=False
            ... )
            >>>
            >>> # 服务器继续运行
            >>> server.wait_for_termination()

        Raises:
            ImportError: 如果未安装 gradio
        """
        try:
            from ..ui import ServerMonitoringUI
        except ImportError:
            raise ImportError(
                "Gradio UI requires additional dependencies. "
                "Install with: pip install splitlearn-comm[ui]"
            )

        # 创建并启动监控 UI
        ui = ServerMonitoringUI(
            servicer=self.servicer,
            theme=theme,
            refresh_interval=refresh_interval,
        )

        ui.launch(
            share=share,
            server_port=server_port,
            **kwargs
        )


def serve(
    compute_fn: ComputeFunction,
    host: str = "0.0.0.0",
    port: int = 50051,
    **kwargs
):
    """
    便捷函数：创建并启动服务器

    Args:
        compute_fn: 计算函数
        host: 主机地址
        port: 端口
        **kwargs: 其他 GRPCComputeServer 参数

    Example:
        >>> from splitlearn_comm.server import serve
        >>> from splitlearn_comm.core import ModelComputeFunction
        >>>
        >>> compute_fn = ModelComputeFunction(model)
        >>> serve(compute_fn, port=50051)
    """
    server = GRPCComputeServer(
        compute_fn=compute_fn,
        host=host,
        port=port,
        **kwargs
    )
    server.start()
    server.wait_for_termination()
