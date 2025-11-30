"""
ComputeServicer - gRPC 服务端 Servicer 实现
"""

import logging
import time
from typing import Optional, List, Dict, Any
from collections import deque
from datetime import datetime

import grpc

from ..core import ComputeFunction, TensorCodec
from ..protocol import compute_service_pb2, compute_service_pb2_grpc

logger = logging.getLogger(__name__)


class ComputeServicer(compute_service_pb2_grpc.ComputeServiceServicer):
    """
    ComputeService 的 gRPC Servicer 实现

    使用 ComputeFunction 抽象，完全解耦模型依赖。

    Example:
        >>> compute_fn = ModelComputeFunction(model, device="cuda")
        >>> servicer = ComputeServicer(compute_fn)
        >>> # servicer 将被添加到 gRPC 服务器
    """

    def __init__(
        self,
        compute_fn: ComputeFunction,
        codec: Optional[TensorCodec] = None,
        version: str = "1.0.0",
        history_size: int = 100
    ):
        """
        Args:
            compute_fn: 计算函数实例
            codec: Tensor 编解码器（默认使用 TensorCodec）
            version: 服务版本号
            history_size: 保留的请求历史记录数量
        """
        self.compute_fn = compute_fn
        self.codec = codec or TensorCodec()
        self.version = version

        # 统计信息
        self.total_requests = 0
        self.total_compute_time = 0.0
        self.server_start_time = time.time()
        self.failed_requests = 0

        # 请求历史（用于监控 UI）
        self.request_history: deque = deque(maxlen=history_size)

        # 初始化计算函数
        try:
            self.compute_fn.setup()
            logger.info("ComputeFunction setup completed")
        except Exception as e:
            logger.warning(f"ComputeFunction setup failed: {e}")

        # 获取服务信息
        self.service_info = self.compute_fn.get_info()
        logger.info(f"ComputeServicer initialized: {self.service_info}")

    def Compute(self, request, context):
        """执行计算"""
        self.total_requests += 1
        start_time = time.time()
        success = False
        compute_time = 0.0
        error_msg = None

        try:
            # 1. 解码输入张量
            input_tensor = self.codec.decode(
                data=request.data,
                shape=tuple(request.shape)
            )

            logger.debug(
                f"[Request {self.total_requests}] Input shape: {input_tensor.shape}"
            )

            # 2. 执行计算
            output_tensor = self.compute_fn.compute(input_tensor)

            # 3. 编码输出张量
            output_data, output_shape = self.codec.encode(output_tensor)

            # 4. 计算耗时
            compute_time = (time.time() - start_time) * 1000  # ms
            self.total_compute_time += compute_time
            success = True

            # 5. 构建响应
            response = compute_service_pb2.ComputeResponse(
                data=output_data,
                shape=list(output_shape),
                compute_time_ms=compute_time
            )

            # 如果请求包含 request_id，回传
            if request.HasField("request_id"):
                response.request_id = request.request_id

            logger.debug(
                f"[Request {self.total_requests}] "
                f"Output shape: {output_shape}, "
                f"Time: {compute_time:.2f}ms"
            )

            return response

        except Exception as e:
            self.failed_requests += 1
            success = False
            error_msg = str(e)
            compute_time = (time.time() - start_time) * 1000  # ms

            logger.error(f"Error in Compute: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return compute_service_pb2.ComputeResponse()

        finally:
            # 记录请求历史（用于监控 UI）
            self.request_history.append({
                "timestamp": datetime.now(),
                "request_id": self.total_requests,
                "success": success,
                "compute_time_ms": compute_time,
                "error": error_msg
            })

    def HealthCheck(self, request, context):
        """健康检查"""
        try:
            uptime = time.time() - self.server_start_time

            response = compute_service_pb2.HealthResponse(
                status="ok",
                message="Service is healthy",
                version=self.version,
                uptime_seconds=uptime
            )

            logger.debug("Health check passed")
            return response

        except Exception as e:
            logger.error(f"Error in HealthCheck: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return compute_service_pb2.HealthResponse(
                status="unhealthy",
                message=str(e)
            )

    def GetServiceInfo(self, request, context):
        """获取服务信息"""
        try:
            uptime = time.time() - self.server_start_time

            # 构建响应
            response = compute_service_pb2.ServiceInfoResponse(
                service_name=self.service_info.get("name", "ComputeService"),
                version=self.version,
                device=self.service_info.get("device", "unknown"),
                uptime_seconds=uptime,
                total_requests=self.total_requests
            )

            # 添加自定义信息
            for key, value in self.service_info.items():
                if key not in ["name", "device"]:
                    response.custom_info[key] = str(value)

            logger.debug(
                f"Service info requested: {self.total_requests} total requests"
            )
            return response

        except Exception as e:
            logger.error(f"Error in GetServiceInfo: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return compute_service_pb2.ServiceInfoResponse()

    def get_metrics(self) -> Dict[str, Any]:
        """
        获取服务器指标（用于监控 UI）

        Returns:
            包含服务器指标的字典
        """
        uptime = time.time() - self.server_start_time
        success_requests = self.total_requests - self.failed_requests
        success_rate = success_requests / self.total_requests if self.total_requests > 0 else 0.0
        avg_compute_time = self.total_compute_time / self.total_requests if self.total_requests > 0 else 0.0

        return {
            "total_requests": self.total_requests,
            "success_requests": success_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "avg_compute_time_ms": avg_compute_time,
            "uptime_seconds": uptime,
            "request_history": list(self.request_history)
        }

    def shutdown(self):
        """关闭服务，清理资源"""
        try:
            self.compute_fn.teardown()
            logger.info("ComputeServicer shutdown completed")
            logger.info(f"Total requests processed: {self.total_requests}")
            if self.total_requests > 0:
                avg_time = self.total_compute_time / self.total_requests
                logger.info(f"Average compute time: {avg_time:.2f}ms")
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
