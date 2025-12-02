"""
GRPCComputeClient - gRPC 客户端实现
"""

import logging
import time
from typing import Optional, Dict, Any

import grpc
import torch

from ..core import TensorCodec
from ..protocol import compute_service_pb2, compute_service_pb2_grpc
from .retry import RetryStrategy, ExponentialBackoff

logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_MAX_MESSAGE_LENGTH = 100 * 1024 * 1024  # 100 MB
DEFAULT_TIMEOUT = 30.0  # 30 seconds


class GRPCComputeClient:
    """
    gRPC 计算客户端

    提供简单的 API 来连接远程计算服务器并执行计算。

    Example:
        >>> from splitlearn_comm import GRPCComputeClient
        >>>
        >>> # 连接服务器
        >>> client = GRPCComputeClient("localhost:50051")
        >>> client.connect()
        >>>
        >>> # 执行计算
        >>> output = client.compute(input_tensor)
        >>>
        >>> # 关闭连接
        >>> client.close()
    """

    def __init__(
        self,
        server_address: str,
        timeout: float = DEFAULT_TIMEOUT,
        max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
        retry_strategy: Optional[RetryStrategy] = None,
        codec: Optional[TensorCodec] = None
    ):
        """
        Args:
            server_address: 服务器地址 (host:port)
            timeout: RPC 超时时间（秒）
            max_message_length: 最大消息长度（字节）
            retry_strategy: 重试策略（默认使用指数退避）
            codec: Tensor 编解码器
        """
        self.server_address = server_address
        self.timeout = timeout
        self.max_message_length = max_message_length
        self.codec = codec or TensorCodec()

        # 重试策略
        self.retry_strategy = retry_strategy or ExponentialBackoff(
            max_retries=3,
            initial_delay=1.0
        )

        # gRPC 连接
        self.channel = None
        self.stub = None

        # 统计信息
        self.request_count = 0
        self.total_network_time = 0.0
        self.total_compute_time = 0.0

    def connect(self) -> bool:
        """
        连接到服务器并进行健康检查

        Returns:
            连接是否成功
        """
        logger.info(f"Connecting to server: {self.server_address}")

        try:
            # 创建 gRPC 通道
            self.channel = grpc.insecure_channel(
                self.server_address,
                options=[
                    ("grpc.max_send_message_length", self.max_message_length),
                    ("grpc.max_receive_message_length", self.max_message_length),
                    ("grpc.keepalive_time_ms", 30000),
                    ("grpc.keepalive_timeout_ms", 10000),
                ],
            )

            # 创建 stub
            self.stub = compute_service_pb2_grpc.ComputeServiceStub(self.channel)

            # 健康检查
            response = self.stub.HealthCheck(
                compute_service_pb2.HealthRequest(),
                timeout=self.timeout
            )

            logger.info(f"✓ Connected to server")
            logger.info(f"  Status: {response.status}")
            logger.info(f"  Version: {response.version}")
            logger.info(f"  Uptime: {response.uptime_seconds:.1f}s")

            return True

        except grpc.RpcError as e:
            logger.error(f"✗ Connection failed: {e.code()}")
            logger.error(f"  Details: {e.details()}")
            return False

        except Exception as e:
            logger.error(f"✗ Unexpected error during connection: {e}")
            return False

    def compute(self, input_tensor: torch.Tensor, model_id: Optional[str] = None) -> torch.Tensor:
        """
        执行远程计算

        Args:
            input_tensor: 输入张量
            model_id: 目标模型 ID (可选)

        Returns:
            输出张量

        Raises:
            grpc.RpcError: RPC 调用失败
        """
        if self.stub is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        def _do_compute():
            self.request_count += 1

            # 1. 编码输入张量
            input_data, input_shape = self.codec.encode(input_tensor)

            # 2. 构建请求
            request_kwargs = {
                "data": input_data,
                "shape": list(input_shape),
                "request_id": self.request_count
            }
            if model_id:
                request_kwargs["model_id"] = model_id
                
            request = compute_service_pb2.ComputeRequest(**request_kwargs)

            # 3. RPC 调用
            start_time = time.time()
            response = self.stub.Compute(request, timeout=self.timeout)
            network_time = (time.time() - start_time) * 1000  # ms

            # 4. 更新统计
            self.total_network_time += network_time
            self.total_compute_time += response.compute_time_ms

            # 5. 解码输出张量
            output_tensor = self.codec.decode(
                data=response.data,
                shape=tuple(response.shape)
            )

            logger.debug(
                f"[Request {self.request_count}] "
                f"Network {network_time:.2f}ms, "
                f"Compute {response.compute_time_ms:.2f}ms"
            )

            return output_tensor

        try:
            # 使用重试策略执行
            return self.retry_strategy.execute(_do_compute)

        except grpc.RpcError as e:
            logger.error(f"RPC Error: {e.code()}")
            logger.error(f"Details: {e.details()}")
            raise

    def health_check(self) -> bool:
        """
        健康检查

        Returns:
            服务器是否健康
        """
        if self.stub is None:
            return False

        try:
            response = self.stub.HealthCheck(
                compute_service_pb2.HealthRequest(),
                timeout=self.timeout
            )
            return response.status == "ok"

        except grpc.RpcError:
            return False

    def get_service_info(self) -> Optional[Dict[str, Any]]:
        """
        获取服务器信息

        Returns:
            服务器信息字典，失败返回 None
        """
        if self.stub is None:
            return None

        try:
            response = self.stub.GetServiceInfo(
                compute_service_pb2.ServiceInfoRequest(),
                timeout=self.timeout
            )

            info = {
                "service_name": response.service_name,
                "version": response.version,
                "device": response.device,
                "uptime_seconds": response.uptime_seconds,
                "total_requests": response.total_requests,
                "custom_info": dict(response.custom_info)
            }

            return info

        except grpc.RpcError as e:
            logger.error(f"Failed to get service info: {e.details()}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取客户端统计信息

        Returns:
            统计信息字典
        """
        avg_network = (
            self.total_network_time / self.request_count
            if self.request_count > 0
            else 0
        )
        avg_compute = (
            self.total_compute_time / self.request_count
            if self.request_count > 0
            else 0
        )

        return {
            "total_requests": self.request_count,
            "avg_network_time_ms": avg_network,
            "avg_compute_time_ms": avg_compute,
            "avg_total_time_ms": avg_network + avg_compute,
        }

    def close(self):
        """关闭连接"""
        if self.channel:
            self.channel.close()
            logger.info(f"Connection closed. Total requests: {self.request_count}")

    def __enter__(self):
        """上下文管理器支持"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器清理"""
        self.close()

    def launch_ui(
        self,
        bottom_model: Any,
        top_model: Any,
        tokenizer: Any,
        theme: str = "default",
        share: bool = False,
        server_port: int = 7860,
        **kwargs
    ):
        """
        启动 Gradio UI 进行交互式文本生成

        这是一个便捷方法，用于快速启动 Gradio 界面进行文本生成。
        需要安装 gradio: pip install splitlearn-comm[ui]

        Args:
            bottom_model: Bottom 模型（在客户端运行）
            top_model: Top 模型（在客户端运行）
            tokenizer: 分词器（用于编码/解码文本）
            theme: UI 主题 ("default", "dark", "light")
            share: 是否创建公共 Gradio 链接
            server_port: 服务器端口
            **kwargs: 传递给 demo.launch() 的额外参数

        Example:
            >>> from splitlearn_comm import GRPCComputeClient
            >>> from transformers import AutoTokenizer
            >>> import torch
            >>>
            >>> # 连接服务器
            >>> client = GRPCComputeClient("localhost:50051")
            >>> client.connect()
            >>>
            >>> # 加载本地模型
            >>> bottom = torch.load("bottom_model.pt")
            >>> top = torch.load("top_model.pt")
            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>>
            >>> # 启动 UI
            >>> client.launch_ui(
            ...     bottom_model=bottom,
            ...     top_model=top,
            ...     tokenizer=tokenizer,
            ...     share=False
            ... )

        Raises:
            ImportError: 如果未安装 gradio
            RuntimeError: 如果客户端未连接
        """
        try:
            from ..ui import ClientUI
        except ImportError:
            raise ImportError(
                "Gradio UI requires additional dependencies. "
                "Install with: pip install splitlearn-comm[ui]"
            )

        if self.stub is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        # 创建并启动 UI
        ui = ClientUI(
            client=self,
            bottom_model=bottom_model,
            top_model=top_model,
            tokenizer=tokenizer,
            theme=theme,
        )

        ui.launch(
            share=share,
            server_port=server_port,
            **kwargs
        )
