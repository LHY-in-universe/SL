"""
GRPCComputeClient - gRPC 客户端实现
"""

import logging
import time
from typing import Optional, Dict, Any, List, Iterator

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

        # 服务端监控数据
        self.server_monitoring_snapshots = []

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
        执行远程计算，仅返回输出张量（向后兼容）。
        如需获取时间拆分信息，请使用 compute_with_timing。
        """
        output, _ = self.compute_with_timing(input_tensor, model_id=model_id)
        return output

    def compute_with_timing(self, input_tensor: torch.Tensor, model_id: Optional[str] = None):
        """
        执行远程计算，返回 (输出张量, timing_dict)

        Args:
            input_tensor: 输入张量
            model_id: 目标模型 ID (可选)

        Returns:
            (输出张量, timing_dict)
            timing_dict 包含：
                - network_total_ms: 往返总耗时（编码+网络+解码+服务端计算）
                - server_compute_ms: 服务端计算时间（来自响应 compute_time_ms，如缺失则为 None）
                - network_overhead_ms: network_total_ms - server_compute_ms（若 server_compute_ms 缺失则为 None）

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
            network_time_ms = (time.time() - start_time) * 1000  # ms

            # 4. 更新统计
            self.total_network_time += network_time_ms
            compute_time = response.compute_time_ms if response.HasField("compute_time_ms") else 0.0
            self.total_compute_time += compute_time

            # 5. 接收并存储服务端监控数据
            if response.HasField("monitoring_data"):
                monitoring_data = response.monitoring_data
                snapshot_dict = {
                    "timestamp": monitoring_data.timestamp,
                    "cpu_percent": monitoring_data.cpu_percent,
                    "memory_mb": monitoring_data.memory_mb,
                    "memory_percent": monitoring_data.memory_percent,
                    "gpu_available": monitoring_data.gpu_available,
                    "gpu_utilization": monitoring_data.gpu_utilization if monitoring_data.HasField("gpu_utilization") else None,
                    "gpu_memory_used_mb": monitoring_data.gpu_memory_used_mb if monitoring_data.HasField("gpu_memory_used_mb") else None,
                    "gpu_memory_total_mb": monitoring_data.gpu_memory_total_mb if monitoring_data.HasField("gpu_memory_total_mb") else None,
                }
                self.server_monitoring_snapshots.append(snapshot_dict)

            # 6. 解码输出张量
            output_tensor = self.codec.decode(
                data=response.data,
                shape=tuple(response.shape)
            )

            server_compute_ms = response.compute_time_ms if response.HasField("compute_time_ms") else None
            network_overhead_ms = None
            if server_compute_ms is not None:
                network_overhead_ms = max(network_time_ms - server_compute_ms, 0.0)

            logger.debug(
                f"[Request {self.request_count}] "
                f"Network {network_time_ms:.2f}ms, "
                f"Compute {server_compute_ms:.2f}ms" if server_compute_ms is not None else f"Network {network_time_ms:.2f}ms"
            )

            timing = {
                "network_total_ms": network_time_ms,
                "server_compute_ms": server_compute_ms,
                "network_overhead_ms": network_overhead_ms,
            }

            return output_tensor, timing

        try:
            # 使用重试策略执行
            return self.retry_strategy.execute(_do_compute)

        except grpc.RpcError as e:
            logger.error(f"RPC Error: {e.code()}")
            logger.error(f"Details: {e.details()}")
            raise

    def compute_with_cache(
        self,
        input_tensor: torch.Tensor,
        past_key_values: Optional[tuple] = None,
        use_cache: bool = False,
        model_id: Optional[str] = None,
        training_mode: bool = False,
        forward_id: Optional[str] = None
    ):
        """
        执行远程计算，支持 KV-cache（用于快速推理）

        Args:
            input_tensor: 输入张量
            past_key_values: 过去的 Key-Value cache（来自前一次生成）
            use_cache: 是否返回 present_key_values
            model_id: 目标模型 ID (可选)
            training_mode: 是否为训练模式
            forward_id: 前向传播 ID（训练模式使用）

        Returns:
            如果 use_cache=True:
                (output_tensor, present_key_values, timing_dict)
            否则:
                (output_tensor, None, timing_dict)

        Raises:
            grpc.RpcError: RPC 调用失败
        """
        if self.stub is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        def _do_compute():
            self.request_count += 1

            # 1. 编码输入张量
            input_data, input_shape = self.codec.encode(input_tensor)

            # 2. 编码 past_key_values（如果有）
            kv_entries = []
            if past_key_values is not None:
                from splitlearn_comm.core.kv_cache_codec import KVCacheCodec
                kv_codec = KVCacheCodec(tensor_codec=self.codec)
                kv_entries = kv_codec.encode(past_key_values)
                logger.debug(f"Encoded {len(kv_entries)} past_key_values layers")

            # 3. 构建请求
            request_kwargs = {
                "data": input_data,
                "shape": list(input_shape),
                "request_id": self.request_count,
                "use_cache": use_cache,
            }

            if model_id:
                request_kwargs["model_id"] = model_id

            if training_mode:
                request_kwargs["training_mode"] = training_mode
                if forward_id:
                    request_kwargs["forward_id"] = forward_id

            request = compute_service_pb2.ComputeRequest(**request_kwargs)

            # 添加 past_key_values 到请求
            if kv_entries:
                request.past_key_values.extend(kv_entries)

            # 4. RPC 调用
            start_time = time.time()
            response = self.stub.Compute(request, timeout=self.timeout)
            network_time_ms = (time.time() - start_time) * 1000  # ms

            # 5. 更新统计
            self.total_network_time += network_time_ms
            compute_time = response.compute_time_ms if response.HasField("compute_time_ms") else 0.0
            self.total_compute_time += compute_time

            # 6. 解码输出张量
            output_tensor = self.codec.decode(
                data=response.data,
                shape=tuple(response.shape)
            )

            # 7. 解码 present_key_values（如果有）
            present_key_values = None
            if len(response.present_key_values) > 0:
                from splitlearn_comm.core.kv_cache_codec import KVCacheCodec
                kv_codec = KVCacheCodec(tensor_codec=self.codec)
                present_key_values = kv_codec.decode(response.present_key_values)
                logger.debug(f"Decoded {len(present_key_values)} present_key_values layers")

            server_compute_ms = response.compute_time_ms if response.HasField("compute_time_ms") else None
            network_overhead_ms = None
            if server_compute_ms is not None:
                network_overhead_ms = max(network_time_ms - server_compute_ms, 0.0)

            timing = {
                "network_total_ms": network_time_ms,
                "server_compute_ms": server_compute_ms,
                "network_overhead_ms": network_overhead_ms,
            }

            return output_tensor, present_key_values, timing

        try:
            # 使用重试策略执行
            return self.retry_strategy.execute(_do_compute)

        except grpc.RpcError as e:
            logger.error(f"RPC Error: {e.code()}")
            logger.error(f"Details: {e.details()}")
            raise

    def compute_backward(
        self,
        grad_output: torch.Tensor,
        forward_id: str,
        model_id: Optional[str] = None
    ) -> torch.Tensor:
        """
        执行反向传播

        Args:
            grad_output: 输出梯度张量（∂loss/∂output）
            forward_id: 前向传播 ID（用于检索缓存的激活值）
            model_id: 目标模型 ID (可选)

        Returns:
            输入梯度张量（∂loss/∂input）

        Raises:
            grpc.RpcError: RPC 调用失败
            RuntimeError: 客户端未连接或服务端训练模式未启用
        """
        if self.stub is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        def _do_backward():
            # 1. 编码梯度张量
            grad_output_data, grad_output_shape = self.codec.encode(grad_output)

            # 2. 构建反向传播请求
            request_kwargs = {
                "grad_output_data": grad_output_data,
                "grad_output_shape": list(grad_output_shape),
                "forward_id": forward_id
            }
            if model_id:
                request_kwargs["model_id"] = model_id

            request = compute_service_pb2.BackwardRequest(**request_kwargs)

            # 3. RPC 调用
            start_time = time.time()
            response = self.stub.ComputeBackward(request, timeout=self.timeout)
            network_time = (time.time() - start_time) * 1000  # ms

            # 4. 更新统计
            self.total_network_time += network_time
            if response.HasField("backward_time_ms"):
                self.total_compute_time += response.backward_time_ms

            # 5. 接收并存储服务端监控数据
            if response.HasField("monitoring_data"):
                monitoring_data = response.monitoring_data
                snapshot_dict = {
                    "timestamp": monitoring_data.timestamp,
                    "cpu_percent": monitoring_data.cpu_percent,
                    "memory_mb": monitoring_data.memory_mb,
                    "memory_percent": monitoring_data.memory_percent,
                    "gpu_available": monitoring_data.gpu_available,
                    "gpu_utilization": monitoring_data.gpu_utilization if monitoring_data.HasField("gpu_utilization") else None,
                    "gpu_memory_used_mb": monitoring_data.gpu_memory_used_mb if monitoring_data.HasField("gpu_memory_used_mb") else None,
                    "gpu_memory_total_mb": monitoring_data.gpu_memory_total_mb if monitoring_data.HasField("gpu_memory_total_mb") else None,
                }
                self.server_monitoring_snapshots.append(snapshot_dict)

            # 6. 解码输入梯度
            grad_input = self.codec.decode(
                data=response.grad_input_data,
                shape=tuple(response.grad_input_shape)
            )

            logger.debug(
                f"[Backward forward_id={forward_id}] "
                f"Network {network_time:.2f}ms, "
                f"Compute {response.backward_time_ms:.2f}ms"
            )

            return grad_input

        try:
            # 使用重试策略执行
            return self.retry_strategy.execute(_do_backward)

        except grpc.RpcError as e:
            logger.error(f"RPC Error in ComputeBackward: {e.code()}")
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

    def get_server_monitoring_data(self) -> List[Dict[str, Any]]:
        """
        获取累积的服务端监控数据

        Returns:
            服务端监控快照列表
        """
        return self.server_monitoring_snapshots.copy()

    def clear_server_monitoring_data(self):
        """清空服务端监控数据"""
        self.server_monitoring_snapshots.clear()

    def stream_monitoring(self, sampling_interval: float = 1.0) -> Iterator[Dict[str, Any]]:
        """
        订阅服务端实时监控流

        Args:
            sampling_interval: 采样间隔秒数（服务端限制 0.1~5.0）

        Yields:
            监控快照字典
        """
        if self.stub is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        interval = sampling_interval if sampling_interval and sampling_interval > 0 else 0
        # 具体限流由服务端控制，这里只做简单校验
        request = compute_service_pb2.MonitoringStreamRequest(
            sampling_interval=interval
        )

        try:
            for resp in self.stub.StreamMonitoring(request):
                snap = resp.snapshot
                yield {
                    "timestamp": snap.timestamp,
                    "cpu_percent": snap.cpu_percent,
                    "memory_mb": snap.memory_mb,
                    "memory_percent": snap.memory_percent,
                    "gpu_available": snap.gpu_available,
                    "gpu_utilization": snap.gpu_utilization if snap.HasField("gpu_utilization") else None,
                    "gpu_memory_used_mb": snap.gpu_memory_used_mb if snap.HasField("gpu_memory_used_mb") else None,
                    "gpu_memory_total_mb": snap.gpu_memory_total_mb if snap.HasField("gpu_memory_total_mb") else None,
                }
        except grpc.RpcError as e:
            logger.warning(f"StreamMonitoring RPC ended: {e.code()} {e.details()}")
        except Exception as e:
            logger.warning(f"StreamMonitoring error: {e}")

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
