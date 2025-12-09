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
        codec: Optional[TensorCodec] = None,
        enable_training: bool = False,
        activation_cache = None
    ):
        """
        初始化异步 Servicer。

        Args:
            compute_fn: 异步计算函数
            codec: 张量编解码器（默认使用 TensorCodec）
            enable_training: 是否启用训练模式（支持反向传播）
            activation_cache: 激活缓存（用于训练模式）
        """
        self.compute_fn = compute_fn
        self.codec = codec or TensorCodec()
        self.enable_training = enable_training
        self.activation_cache = activation_cache

        # 统计信息
        self.total_requests = 0
        self.failed_requests = 0
        self.lock = asyncio.Lock()

        logger.info(f"AsyncComputeServicer initialized (training={enable_training})")

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
            # Note: TensorCodec.decode always uses float32, no dtype parameter needed
            input_tensor = self.codec.decode(
                data=request.data,
                shape=tuple(request.shape)
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
            # Note: ComputeResponse does not have dtype field, always uses float32
            return compute_service_pb2.ComputeResponse(
                data=output_data,
                shape=list(output_shape)
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

    async def ComputeBackward(
        self,
        request: compute_service_pb2.BackwardRequest,
        context: grpc.aio.ServicerContext
    ) -> compute_service_pb2.BackwardResponse:
        """
        异步反向传播。

        Args:
            request: 反向传播请求
            context: gRPC 上下文

        Returns:
            反向传播响应

        Raises:
            grpc.RpcError: 如果反向传播失败
        """
        # 检查训练模式
        if not self.enable_training or self.activation_cache is None:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Training mode not enabled or activation cache not available"
            )

        try:
            # 解码梯度
            grad_output = self.codec.decode(
                request.grad_output_data,
                tuple(request.grad_output_shape)
            )

            logger.debug(
                f"Received backward request: forward_id={request.forward_id}, "
                f"grad_shape={grad_output.shape}"
            )

            # 检索激活值
            try:
                cached_activation = self.activation_cache.retrieve(request.forward_id)
            except KeyError as e:
                logger.error(f"Activation cache miss: {e}")
                await context.abort(grpc.StatusCode.NOT_FOUND, str(e))

            # 执行反向传播
            cached_activation.backward(grad_output, retain_graph=False)
            grad_input = cached_activation.grad

            logger.debug(
                f"Backward completed: grad_input_shape={grad_input.shape}"
            )

            # 编码返回梯度
            grad_input_data, grad_input_shape = self.codec.encode(grad_input)

            return compute_service_pb2.BackwardResponse(
                grad_input_data=grad_input_data,
                grad_input_shape=list(grad_input_shape),
                forward_id=request.forward_id
            )

        except Exception as e:
            logger.error(f"Backward error: {e}", exc_info=True)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

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
