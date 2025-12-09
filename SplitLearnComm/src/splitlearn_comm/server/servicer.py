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
from ..monitoring import MetricsManager, LogManager, LogLevel, MonitoringConfig
from .activation_cache import ActivationCache

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
        history_size: int = 100,
        monitoring_config: Optional[MonitoringConfig] = None,
        enable_resource_monitoring: bool = True,
        enable_training: bool = False,
        activation_cache_size: int = 100,
        activation_cache_ttl: float = 60.0
    ):
        """
        Args:
            compute_fn: 计算函数实例
            codec: Tensor 编解码器（默认使用 TensorCodec）
            version: 服务版本号
            history_size: 保留的请求历史记录数量
            monitoring_config: 监控配置（默认使用 MonitoringConfig）
            enable_resource_monitoring: 是否启用资源监控（CPU/GPU/内存）
            enable_training: 是否启用训练模式（激活值缓存和反向传播）
            activation_cache_size: 激活值缓存最大条目数
            activation_cache_ttl: 激活值缓存存活时间（秒）
        """
        self.compute_fn = compute_fn
        self.codec = codec or TensorCodec()
        self.version = version
        self.enable_training = enable_training

        # 统计信息
        self.total_requests = 0
        self.total_backward_requests = 0
        self.total_compute_time = 0.0
        self.total_backward_time = 0.0
        self.server_start_time = time.time()
        self.failed_requests = 0
        self.failed_backward_requests = 0

        # 请求历史（用于监控 UI）
        self.request_history: deque = deque(maxlen=history_size)

        # 激活值缓存（用于训练）
        self.activation_cache = None
        if enable_training:
            self.activation_cache = ActivationCache(
                max_size=activation_cache_size,
                ttl_seconds=activation_cache_ttl
            )
            logger.info(f"Training mode enabled with activation cache")

        # 监控管理器
        config = monitoring_config or MonitoringConfig()
        self.metrics_manager = MetricsManager(max_history_size=config.max_history_size)
        self.log_manager = LogManager(max_logs=config.max_log_size)

        # 初始化资源监控（如果启用）
        self.server_monitor = None
        if enable_resource_monitoring:
            try:
                from splitlearn_monitor import ServerMonitor
                self.server_monitor = ServerMonitor(
                    server_name="compute_server",
                    sampling_interval=0.1,
                    enable_gpu=True,
                    auto_start=True
                )
                logger.info("ServerMonitor initialized and started")
                self.log_manager.add_log(
                    LogLevel.INFO,
                    "ServerMonitor initialized and started"
                )
            except ImportError:
                logger.warning("splitlearn_monitor not available, resource monitoring disabled")
                self.log_manager.add_log(
                    LogLevel.WARNING,
                    "splitlearn_monitor not available, resource monitoring disabled"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize ServerMonitor: {e}")
                self.log_manager.add_log(
                    LogLevel.WARNING,
                    f"Failed to initialize ServerMonitor: {e}"
                )

        # 初始化计算函数
        try:
            self.compute_fn.setup()
            logger.info("ComputeFunction setup completed")
            self.log_manager.add_log(
                LogLevel.INFO,
                "ComputeFunction setup completed"
            )
        except Exception as e:
            logger.warning(f"ComputeFunction setup failed: {e}")
            self.log_manager.add_log(
                LogLevel.WARNING,
                f"ComputeFunction setup failed: {e}"
            )

        # 获取服务信息
        self.service_info = self.compute_fn.get_info()
        logger.info(f"ComputeServicer initialized: {self.service_info}")
        self.log_manager.add_log(
            LogLevel.INFO,
            f"ComputeServicer initialized: {self.service_info}"
        )

    def _snapshot_to_proto(self, snapshot):
        """将监控快照转换为 protobuf 对象"""
        if snapshot is None:
            return None

        monitoring_data = compute_service_pb2.MonitoringSnapshot(
            timestamp=snapshot.timestamp,
            cpu_percent=snapshot.cpu_percent,
            memory_mb=snapshot.memory_mb,
            memory_percent=snapshot.memory_percent,
            gpu_available=snapshot.gpu_available
        )

        if snapshot.gpu_available and snapshot.gpu_utilization is not None:
            monitoring_data.gpu_utilization = snapshot.gpu_utilization
            if snapshot.gpu_memory_used_mb is not None:
                monitoring_data.gpu_memory_used_mb = snapshot.gpu_memory_used_mb
            if snapshot.gpu_memory_total_mb is not None:
                monitoring_data.gpu_memory_total_mb = snapshot.gpu_memory_total_mb

        return monitoring_data

    def Compute(self, request, context):
        """执行计算（前向传播）"""
        self.total_requests += 1
        request_id = self.total_requests
        start_time = time.time()
        success = False
        compute_time = 0.0
        error_msg = None
        forward_id = None

        try:
            # 1. 解码输入张量
            input_tensor = self.codec.decode(
                data=request.data,
                shape=tuple(request.shape)
            )

            logger.debug(
                f"[Request {request_id}] Input shape: {input_tensor.shape}"
            )
            self.log_manager.add_log(
                LogLevel.DEBUG,
                f"Received compute request {request_id}, shape={input_tensor.shape}"
            )

            # 2. 检查是否为训练模式
            training_mode = request.HasField("training_mode") and request.training_mode

            # 3. 执行计算
            if training_mode and self.activation_cache is not None:
                # 训练模式：需要启用梯度并缓存激活值
                input_tensor.requires_grad_(True)
                output_tensor = self.compute_fn.compute(input_tensor)

                # 生成 forward_id 并缓存输入激活
                import uuid
                forward_id = request.forward_id if request.HasField("forward_id") else str(uuid.uuid4())
                self.activation_cache.store(forward_id, input_tensor)

                logger.debug(f"[Request {request_id}] Cached activation with forward_id={forward_id}")
            else:
                # 推理模式：无需梯度
                output_tensor = self.compute_fn.compute(input_tensor)

            # 4. 编码输出张量
            output_data, output_shape = self.codec.encode(output_tensor)

            # 5. 计算耗时
            compute_time = (time.time() - start_time) * 1000  # ms
            self.total_compute_time += compute_time
            success = True

            # 记录延迟到 MetricsManager
            self.metrics_manager.record_latency(compute_time / 1000.0)  # 转换为秒

            # 6. 构建响应
            response = compute_service_pb2.ComputeResponse(
                data=output_data,
                shape=list(output_shape),
                compute_time_ms=compute_time
            )

            # 如果请求包含 request_id，回传
            if request.HasField("request_id"):
                response.request_id = request.request_id

            # 如果是训练模式，返回 forward_id
            if training_mode and forward_id:
                response.forward_id = forward_id

            # 6. 附加服务端监控数据（如果可用）
            if self.server_monitor:
                try:
                    snapshot = self.server_monitor.system_monitor.get_current_snapshot()
                    monitoring_data = self._snapshot_to_proto(snapshot)
                    if monitoring_data:
                        response.monitoring_data.CopyFrom(monitoring_data)
                except Exception as e:
                    logger.warning(f"Failed to attach monitoring data: {e}")

            logger.debug(
                f"[Request {request_id}] "
                f"Output shape: {output_shape}, "
                f"Time: {compute_time:.2f}ms"
            )
            self.log_manager.add_log(
                LogLevel.INFO,
                f"Request {request_id} completed successfully in {compute_time:.2f}ms"
            )

            return response

        except Exception as e:
            self.failed_requests += 1
            success = False
            error_msg = str(e)
            compute_time = (time.time() - start_time) * 1000  # ms

            logger.error(f"Error in Compute: {e}", exc_info=True)
            self.log_manager.add_log(
                LogLevel.ERROR,
                f"Request {request_id} failed: {error_msg}"
            )

            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return compute_service_pb2.ComputeResponse()

        finally:
            # 记录请求历史（用于监控 UI）
            self.request_history.append({
                "timestamp": datetime.now(),
                "request_id": request_id,
                "success": success,
                "compute_time_ms": compute_time,
                "error": error_msg
            })

    def ComputeBackward(self, request, context):
        """执行反向传播"""
        # 检查是否启用训练模式
        if not self.enable_training or self.activation_cache is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Training mode not enabled on server")
            logger.error("ComputeBackward called but training mode not enabled")
            return compute_service_pb2.BackwardResponse()

        self.total_backward_requests += 1
        request_id = self.total_backward_requests
        start_time = time.time()
        success = False
        backward_time = 0.0
        error_msg = None

        try:
            # 1. 解码梯度张量
            grad_output = self.codec.decode(
                data=request.grad_output_data,
                shape=tuple(request.grad_output_shape)
            )

            forward_id = request.forward_id

            logger.debug(
                f"[Backward {request_id}] forward_id={forward_id}, "
                f"grad_output shape: {grad_output.shape}"
            )
            self.log_manager.add_log(
                LogLevel.DEBUG,
                f"Received backward request {request_id}, forward_id={forward_id}"
            )

            # 2. 检索缓存的激活值
            try:
                cached_activation = self.activation_cache.retrieve(forward_id)
            except KeyError as e:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(str(e))
                logger.error(f"[Backward {request_id}] {e}")
                self.failed_backward_requests += 1
                return compute_service_pb2.BackwardResponse()

            # 3. 执行反向传播
            # cached_activation 已经 requires_grad=True
            # 调用 backward 将计算 cached_activation 的梯度
            cached_activation.backward(grad_output, retain_graph=False)

            # 4. 提取输入梯度
            grad_input = cached_activation.grad

            if grad_input is None:
                raise RuntimeError("Gradient not computed for cached activation")

            # 5. 编码输入梯度
            grad_input_data, grad_input_shape = self.codec.encode(grad_input)

            # 6. 计算耗时
            backward_time = (time.time() - start_time) * 1000  # ms
            self.total_backward_time += backward_time
            success = True

            # 7. 构建响应
            response = compute_service_pb2.BackwardResponse(
                grad_input_data=grad_input_data,
                grad_input_shape=list(grad_input_shape),
                forward_id=forward_id,
                backward_time_ms=backward_time
            )

            # 8. 附加监控数据（如果可用）
            if self.server_monitor:
                try:
                    snapshot = self.server_monitor.system_monitor.get_current_snapshot()
                    monitoring_data = self._snapshot_to_proto(snapshot)
                    if monitoring_data:
                        response.monitoring_data.CopyFrom(monitoring_data)
                except Exception as e:
                    logger.warning(f"Failed to attach monitoring data: {e}")

            logger.debug(
                f"[Backward {request_id}] "
                f"grad_input shape: {grad_input_shape}, "
                f"Time: {backward_time:.2f}ms"
            )
            self.log_manager.add_log(
                LogLevel.INFO,
                f"Backward request {request_id} completed successfully in {backward_time:.2f}ms"
            )

            return response

        except Exception as e:
            self.failed_backward_requests += 1
            success = False
            error_msg = str(e)
            backward_time = (time.time() - start_time) * 1000  # ms

            logger.error(f"Error in ComputeBackward: {e}", exc_info=True)
            self.log_manager.add_log(
                LogLevel.ERROR,
                f"Backward request {request_id} failed: {error_msg}"
            )

            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return compute_service_pb2.BackwardResponse()

        finally:
            # 记录请求历史
            self.request_history.append({
                "timestamp": datetime.now(),
                "request_id": f"backward_{request_id}",
                "success": success,
                "compute_time_ms": backward_time,
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

    def StreamMonitoring(self, request, context):
        """流式推送监控快照"""
        if not self.server_monitor:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Resource monitoring is not enabled on server")
            return

        # 默认 1.0s，限定 0.1~5.0s
        interval = 1.0
        try:
            if request.HasField("sampling_interval") and request.sampling_interval > 0:
                interval = request.sampling_interval
        except Exception:
            pass
        interval = max(0.1, min(interval, 5.0))

        while context.is_active():
            try:
                snapshot = self.server_monitor.system_monitor.get_current_snapshot()
                monitoring_data = self._snapshot_to_proto(snapshot)
                if monitoring_data:
                    yield compute_service_pb2.MonitoringSnapshotResponse(
                        snapshot=monitoring_data
                    )
            except Exception as e:
                logger.warning(f"StreamMonitoring error: {e}")
                break
            time.sleep(interval)

    def get_metrics(self) -> Dict[str, Any]:
        """
        获取服务器指标（用于监控 UI）

        Returns:
            包含服务器指标的字典，包括：
            - 基本统计：请求数、成功率、平均时间
            - 详细延迟统计：P50/P95/P99、min/max、标准差
            - 延迟分布数据
            - 吞吐量历史
            - 请求历史
        """
        uptime = time.time() - self.server_start_time
        success_requests = self.total_requests - self.failed_requests
        success_rate = success_requests / self.total_requests if self.total_requests > 0 else 0.0
        avg_compute_time = self.total_compute_time / self.total_requests if self.total_requests > 0 else 0.0

        # 获取详细的延迟统计
        latency_stats = self.metrics_manager.get_latency_stats()
        latency_distribution = self.metrics_manager.get_latency_distribution()
        latency_history = self.metrics_manager.get_latency_history(limit=50)
        throughput_history = self.metrics_manager.get_throughput_history(limit=50)

        return {
            # 基本统计
            "total_requests": self.total_requests,
            "success_requests": success_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "avg_compute_time_ms": avg_compute_time,
            "uptime_seconds": uptime,

            # 详细延迟统计 (来自 MetricsManager)
            "latency_stats": latency_stats,

            # 延迟分布 (用于直方图)
            "latency_distribution": {
                "bin_edges": latency_distribution[0],
                "counts": latency_distribution[1]
            },

            # 延迟历史 (用于趋势图)
            "latency_history": latency_history,

            # 吞吐量历史 (用于吞吐量图)
            "throughput_history": throughput_history,

            # 当前吞吐量
            "current_rps": self.metrics_manager.get_current_throughput(),

            # 请求历史（保持向后兼容）
            "request_history": list(self.request_history)
        }

    def get_logs(
        self,
        level_filter: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        获取日志记录（用于监控 UI）

        Args:
            level_filter: 日志级别过滤器 (e.g., "ERROR", "INFO", "ALL")
            limit: 返回的最大日志数量

        Returns:
            日志列表
        """
        return self.log_manager.get_logs(
            level_filter=level_filter,
            limit=limit
        )

    def shutdown(self):
        """关闭服务，清理资源"""
        try:
            # 停止资源监控（如果启用）
            if self.server_monitor:
                self.server_monitor.stop()
                logger.info("ServerMonitor stopped")

            self.compute_fn.teardown()
            logger.info("ComputeServicer shutdown completed")
            logger.info(f"Total requests processed: {self.total_requests}")
            if self.total_requests > 0:
                avg_time = self.total_compute_time / self.total_requests
                logger.info(f"Average compute time: {avg_time:.2f}ms")
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
