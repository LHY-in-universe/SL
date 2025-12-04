"""
异步托管 gRPC 服务器，集成模型生命周期管理。

这是完整的异步解决方案，集成了：
- AsyncModelManager：异步模型管理
- AsyncGRPCComputeServer：异步 gRPC 服务器
- 资源监控和健康检查
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any

import torch

from splitlearn_comm import AsyncGRPCComputeServer
from splitlearn_comm.core import AsyncComputeFunction

from ..config import ModelConfig, ServerConfig
from ..core.async_model_manager import AsyncModelManager
from ..core import ResourceManager
from ..monitoring import MetricsCollector, HealthChecker
from ..routing import ModelRouter

logger = logging.getLogger(__name__)


class AsyncManagedComputeFunction(AsyncComputeFunction):
    """
    异步计算函数，通过 ModelManager 路由请求。

    集成了路由、模型管理和指标收集。
    """

    def __init__(
        self,
        model_manager: AsyncModelManager,
        router: ModelRouter,
        resource_manager: ResourceManager,
        metrics: Optional[MetricsCollector] = None,
        executor: Optional[Any] = None
    ):
        """
        初始化异步托管计算函数。

        Args:
            model_manager: 异步模型管理器实例
            router: 模型路由器实例
            resource_manager: 资源管理器实例
            metrics: 指标收集器（可选）
            executor: 线程池执行器（可选，用于模型推理）
        """
        self.model_manager = model_manager
        self.router = router
        self.resource_manager = resource_manager
        self.metrics = metrics
        self.executor = executor

    async def compute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        使用托管模型执行异步计算。

        Args:
            input_tensor: 输入张量

        Returns:
            输出张量

        Raises:
            RuntimeError: 如果没有可用模型或计算失败
        """
        start_time = time.time()

        # 路由到一个模型
        model_id = self.router.route_to_model()

        if not model_id:
            raise RuntimeError("No models available")

        # 获取托管模型
        managed_model = await self.model_manager.get_model(model_id)

        if not managed_model:
            raise RuntimeError(f"Model {model_id} not found or still loading")

        # 执行计算
        try:
            # 在 executor 中执行推理（避免阻塞事件循环）
            loop = asyncio.get_event_loop()

            def _sync_inference():
                with torch.no_grad():
                    return managed_model.model(input_tensor)

            # 使用指定的 executor，如果没有则使用默认的
            output = await loop.run_in_executor(self.executor, _sync_inference)

            # 记录成功指标
            if self.metrics:
                duration = time.time() - start_time
                self.metrics.record_inference_request(model_id, duration, True)

            return output

        except Exception as e:
            # 记录失败指标
            if self.metrics:
                duration = time.time() - start_time
                self.metrics.record_inference_request(model_id, duration, False)

            logger.error(f"Computation failed for model {model_id}: {e}")
            raise RuntimeError(f"Computation failed: {e}")

    def get_info(self) -> Dict:
        """获取服务信息（同步）。"""
        # 注意：这是同步方法，不应该调用异步的 list_models
        # 在实际使用中，应该缓存这些信息或使用其他方式
        usage = self.resource_manager.get_current_usage()

        return {
            "name": "AsyncManagedComputeService",
            "type": "async",
            "custom_info": {
                "cpu_percent": str(usage.cpu_percent),
                "memory_mb": str(usage.memory_mb),
            }
        }


class AsyncManagedServer:
    """
    异步托管服务器。

    集成了所有组件：
    - AsyncModelManager：模型生命周期管理
    - AsyncGRPCComputeServer：gRPC 服务
    - ModelRouter：请求路由
    - MetricsCollector：指标收集
    - HealthChecker：健康检查
    - ResourceManager：资源监控

    主要特性：
    - 完全异步：解决锁阻塞问题
    - 高并发：可同时处理多个请求
    - 资源高效：协程开销小
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        """
        初始化异步托管服务器。

        Args:
            config: 服务器配置（可选）
        """
        self.config = config or ServerConfig()

        # 核心组件
        self.resource_manager = ResourceManager()
        self.model_manager = AsyncModelManager(
            resource_manager=self.resource_manager,
            max_models=self.config.max_models,
            max_workers=self.config.max_workers
        )
        self.router = ModelRouter(self.model_manager)

        # 监控组件
        self.metrics = MetricsCollector() if self.config.enable_monitoring else None
        self.health_checker = HealthChecker(
            model_manager=self.model_manager,
            resource_manager=self.resource_manager
        )

        # 创建异步计算函数
        # 传递 model_manager 的 executor 用于推理
        self.compute_fn = AsyncManagedComputeFunction(
            model_manager=self.model_manager,
            router=self.router,
            resource_manager=self.resource_manager,
            metrics=self.metrics,
            executor=self.model_manager.executor
        )

        # 创建异步 gRPC 服务器
        # max_message_length 从 config 字典获取，如果没有则使用默认值 100MB
        max_message_length = self.config.config.get("max_message_length", 100 * 1024 * 1024)
        self.grpc_server = AsyncGRPCComputeServer(
            compute_fn=self.compute_fn,
            host=self.config.host,
            port=self.config.port,
            max_message_length=max_message_length
        )

        # 监控任务
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False

        logger.info(
            f"AsyncManagedServer initialized "
            f"(host={self.config.host}, port={self.config.port})"
        )

    async def load_model(self, config: ModelConfig) -> bool:
        """
        异步加载模型。

        Args:
            config: 模型配置

        Returns:
            True 表示加载成功
        """
        return await self.model_manager.load_model(config)

    async def unload_model(self, model_id: str) -> bool:
        """
        异步卸载模型。

        Args:
            model_id: 模型标识符

        Returns:
            True 表示卸载成功
        """
        return await self.model_manager.unload_model(model_id)

    async def list_models(self) -> list:
        """
        列出所有已加载的模型。

        Returns:
            模型信息列表
        """
        return await self.model_manager.list_models()

    async def start(self):
        """
        异步启动服务器。

        启动所有组件：
        - gRPC 服务器
        - 监控任务
        - 指标收集
        """
        logger.info("Starting AsyncManagedServer...")

        # 标记为运行状态
        self.running = True

        # 启动 gRPC 服务器
        await self.grpc_server.start()

        # 启动监控任务
        if self.config.health_check_interval > 0:
            self.monitoring_task = asyncio.create_task(
                self._monitoring_loop()
            )
            logger.info("Monitoring task started")

        logger.info(
            f"AsyncManagedServer started on "
            f"{self.config.host}:{self.config.port}"
        )

    async def stop(self, grace: float = 5.0):
        """
        异步停止服务器。

        Args:
            grace: 优雅关闭超时（秒）
        """
        logger.info("Stopping AsyncManagedServer...")

        # 标记为停止状态
        self.running = False

        # 停止监控任务
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                logger.info("Monitoring task cancelled")

        # 停止 gRPC 服务器
        await self.grpc_server.stop(grace)

        # 关闭模型管理器
        await self.model_manager.shutdown()

        logger.info("AsyncManagedServer stopped")

    async def wait_for_termination(self):
        """
        等待服务器终止。

        这会一直阻塞，直到服务器被停止。
        """
        await self.grpc_server.wait_for_termination()

    async def serve(self):
        """
        启动并运行服务器，直到被停止。

        这是一个便捷方法，组合了 start() 和 wait_for_termination()。
        """
        await self.start()
        await self.wait_for_termination()

    async def _monitoring_loop(self):
        """
        异步监控循环。

        定期检查：
        - 资源使用情况
        - 模型健康状态
        - 收集指标
        """
        logger.info(
            f"Monitoring loop started "
            f"(interval={self.config.health_check_interval}s)"
        )

        while self.running:
            try:
                # 记录资源使用
                self.resource_manager.log_resource_usage()

                # 执行健康检查（跳过模型检查，因为它是异步的）
                # 只检查资源，模型检查在 get_status() 中处理
                try:
                    health_status = self.health_checker.check_health()
                    # 如果健康检查返回字典，提取状态
                    if isinstance(health_status, dict):
                        status = health_status.get("status", "unknown")
                        if status != "healthy":
                            message = health_status.get("message", "Unknown issue")
                            logger.warning(f"Health check warning: {message}")
                except Exception as e:
                    logger.warning(f"Health check error: {e}")

                # 收集指标（如果启用）
                if self.metrics:
                    try:
                        stats = await self.model_manager.get_statistics()
                        # 更新模型数量
                        if isinstance(stats, dict):
                            models_count = stats.get("total_models", 0)
                            self.metrics.update_models_loaded(models_count)
                    except Exception as e:
                        logger.debug(f"Failed to update metrics: {e}")

                # 等待下一次检查
                await asyncio.sleep(self.config.health_check_interval)

            except asyncio.CancelledError:
                logger.info("Monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(self.config.health_check_interval)

    async def get_status(self) -> Dict:
        """
        获取服务器状态。

        Returns:
            状态信息字典
        """
        models = await self.model_manager.list_models()
        stats = await self.model_manager.get_statistics()
        health = self.health_checker.check_health()
        grpc_stats = await self.grpc_server.get_statistics()

        # 处理健康检查结果（可能是字典）
        if isinstance(health, dict):
            health_status = health.get("status", "unknown")
            health_message = health.get("message", "No message")
        else:
            health_status = getattr(health, "status", "unknown")
            health_message = getattr(health, "message", "No message")

        return {
            "running": self.running,
            "models": models,
            "statistics": stats,
            "health": {
                "status": health_status,
                "message": health_message,
            },
            "grpc": grpc_stats,
            "config": {
                "host": self.config.host,
                "port": self.config.port,
                "max_models": self.config.max_models,
            }
        }

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
    config: Optional[ServerConfig] = None,
    models: Optional[list[ModelConfig]] = None
):
    """
    便捷函数：创建并运行异步托管服务器。

    Args:
        config: 服务器配置（可选）
        models: 要预加载的模型配置列表（可选）

    Example:
        ```python
        import asyncio
        from splitlearn_manager.server import serve_async
        from splitlearn_manager.config import ServerConfig, ModelConfig

        async def main():
            config = ServerConfig(host="0.0.0.0", port=50051)
            model_configs = [
                ModelConfig(model_id="model1", ...)
            ]

            await serve_async(config, models=model_configs)

        asyncio.run(main())
        ```
    """
    server = AsyncManagedServer(config=config)

    # 预加载模型（如果提供）
    if models:
        logger.info(f"Pre-loading {len(models)} models...")
        for model_config in models:
            try:
                await server.load_model(model_config)
                logger.info(f"Pre-loaded model {model_config.model_id}")
            except Exception as e:
                logger.error(f"Failed to pre-load model {model_config.model_id}: {e}")

    # 启动并运行
    await server.serve()


if __name__ == "__main__":
    # 简单的测试示例
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    async def main():
        config = ServerConfig(
            host="0.0.0.0",
            port=50051,
            max_models=3
        )

        logger.info("Starting test AsyncManagedServer...")
        await serve_async(config)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
