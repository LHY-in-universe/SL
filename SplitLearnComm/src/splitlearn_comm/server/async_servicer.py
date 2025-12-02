"""
异步 gRPC Compute Servicer。

实现了 ComputeService 的异步版本，使用 grpc.aio。
"""

import asyncio
import logging
from typing import Optional

import grpc.aio

from ..protocol import compute_service_pb2, compute_service_pb2_grpc
from ..core.tensor_codec import TensorCodec
from ..core.async_compute_function import AsyncComputeFunction

logger = logging.getLogger(__name__)


class AsyncComputeServicer(compute_service_pb2_grpc.ComputeServiceServicer):
    """
    异步 ComputeService Servicer。

    实现了异步的 Compute 和 HealthCheck RPC 方法。
    使用 grpc.aio 提供非阻塞的 gRPC 服务。
    """

    def __init__(
        self,
        compute_fn: AsyncComputeFunction,
        codec: Optional[TensorCodec] = None
    ):
        """
        初始化异步 Servicer。

        Args:
            compute_fn: 异步计算函数
            codec: 张量编解码器（默认使用 TensorCodec）
        """
        self.compute_fn = compute_fn
        self.codec = codec or TensorCodec()

        # 统计信息
        self.total_requests = 0
        self.failed_requests = 0
        self.lock = asyncio.Lock()

        logger.info("AsyncComputeServicer initialized")

    async def Compute(
        self,
        request: compute_service_pb2.ComputeRequest,
        context: grpc.aio.ServicerContext
    ) -> compute_service_pb2.ComputeResponse:
        """
        异步处理计算请求。

        Args:
            request: 计算请求
            context: gRPC 上下文

        Returns:
            计算响应

        Raises:
            grpc.RpcError: 如果计算失败
        """
        # 增加请求计数
        async with self.lock:
            self.total_requests += 1

        try:
            # 解码输入张量
            input_tensor = self.codec.decode(
                data=request.data,
                shape=tuple(request.shape),
                dtype=request.dtype if request.dtype else "float32"
            )

            logger.debug(
                f"Received compute request: shape={input_tensor.shape}, "
                f"dtype={input_tensor.dtype}"
            )

            # 异步执行计算
            output_tensor = await self.compute_fn.compute(input_tensor)

            logger.debug(
                f"Compute completed: output_shape={output_tensor.shape}"
            )

            # 编码输出张量
            output_data, output_shape = self.codec.encode(output_tensor)

            # 返回响应
            return compute_service_pb2.ComputeResponse(
                data=output_data,
                shape=list(output_shape),
                dtype=str(output_tensor.dtype).replace("torch.", "")
            )

        except Exception as e:
            logger.error(f"Compute error: {e}", exc_info=True)

            # 增加失败计数
            async with self.lock:
                self.failed_requests += 1

            # 设置 gRPC 错误状态
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Computation failed: {str(e)}"
            )

    async def HealthCheck(
        self,
        request: compute_service_pb2.HealthRequest,
        context: grpc.aio.ServicerContext
    ) -> compute_service_pb2.HealthResponse:
        """
        异步健康检查。

        Args:
            request: 健康检查请求
            context: gRPC 上下文

        Returns:
            健康检查响应
        """
        try:
            # 获取统计信息
            async with self.lock:
                total = self.total_requests
                failed = self.failed_requests

            return compute_service_pb2.HealthResponse(
                status="healthy",
                message=f"Service is running (requests: {total}, failed: {failed})"
            )

        except Exception as e:
            logger.error(f"HealthCheck error: {e}")
            return compute_service_pb2.HealthResponse(
                status="unhealthy",
                message=f"Health check failed: {str(e)}"
            )

    async def GetServiceInfo(
        self,
        request: compute_service_pb2.ServiceInfoRequest,
        context: grpc.aio.ServicerContext
    ) -> compute_service_pb2.ServiceInfoResponse:
        """
        异步获取服务信息。

        Args:
            request: 服务信息请求
            context: gRPC 上下文

        Returns:
            服务信息响应
        """
        try:
            # 获取计算函数信息
            compute_info = self.compute_fn.get_info()

            # 获取统计信息
            async with self.lock:
                total = self.total_requests
                failed = self.failed_requests

            info_str = (
                f"AsyncComputeService - "
                f"Type: {compute_info.get('type', 'unknown')}, "
                f"Requests: {total}, "
                f"Failed: {failed}, "
                f"Success Rate: {(total - failed) / total * 100:.2f}% "
                if total > 0 else "No requests yet"
            )

            return compute_service_pb2.ServiceInfoResponse(
                service_name="AsyncComputeService",
                version="2.0.0-async",
                info=info_str
            )

        except Exception as e:
            logger.error(f"GetServiceInfo error: {e}")
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Failed to get service info: {str(e)}"
            )

    async def get_statistics(self) -> dict:
        """
        获取服务统计信息。

        Returns:
            统计信息字典
        """
        async with self.lock:
            return {
                "total_requests": self.total_requests,
                "failed_requests": self.failed_requests,
                "success_rate": (
                    (self.total_requests - self.failed_requests)
                    / self.total_requests * 100
                    if self.total_requests > 0 else 0.0
                ),
            }
