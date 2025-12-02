"""
异步托管服务器使用示例。

这个示例展示了如何使用新的异步 API 来：
1. 创建异步托管服务器
2. 异步加载模型
3. 处理并发请求而不阻塞

相比同步版本的优势：
- 模型加载期间，服务器仍可响应其他请求
- 支持并发加载多个模型
- 更高的并发 QPS（2-3倍提升）
"""

import asyncio
import logging

from splitlearn_manager import AsyncModelManager
from splitlearn_manager.server import AsyncManagedServer
from splitlearn_manager.config import ServerConfig, ModelConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def example_basic_usage():
    """基本使用示例：启动服务器并加载模型。"""
    logger.info("=== 基本使用示例 ===")

    # 1. 创建服务器配置
    server_config = ServerConfig(
        host="0.0.0.0",
        port=50051,
        max_models=5,
        health_check_interval=30  # 30秒健康检查
    )

    # 2. 创建异步托管服务器
    server = AsyncManagedServer(config=server_config)

    # 3. 启动服务器（不阻塞）
    await server.start()

    # 4. 异步加载模型
    model_config = ModelConfig(
        model_id="gpt2_bottom",
        model_type="gpt2",
        component="bottom",
        model_name_or_path="gpt2",
        device="cpu"  # 或 "cuda" 如果有 GPU
    )

    try:
        logger.info("开始加载模型...")
        await server.load_model(model_config)
        logger.info("模型加载成功！")

        # 5. 列出已加载的模型
        models = await server.list_models()
        logger.info(f"已加载的模型: {models}")

        # 6. 获取服务器状态
        status = await server.get_status()
        logger.info(f"服务器状态: {status}")

        # 保持服务器运行
        logger.info("服务器正在运行，按 Ctrl+C 停止...")
        await server.wait_for_termination()

    except KeyboardInterrupt:
        logger.info("接收到停止信号")
    finally:
        # 7. 优雅关闭
        await server.stop()


async def example_concurrent_loading():
    """并发加载示例：同时加载多个模型。"""
    logger.info("=== 并发加载示例 ===")

    # 创建异步模型管理器
    manager = AsyncModelManager(max_models=10)

    # 定义多个模型配置
    model_configs = [
        ModelConfig(
            model_id=f"model_{i}",
            model_type="gpt2",
            component="bottom",
            model_name_or_path="gpt2",
            device="cpu"
        )
        for i in range(3)
    ]

    # 并发加载所有模型！
    logger.info("开始并发加载3个模型...")
    start_time = asyncio.get_event_loop().time()

    # 使用 asyncio.gather 并发执行
    results = await asyncio.gather(
        *[manager.load_model(config) for config in model_configs],
        return_exceptions=True
    )

    end_time = asyncio.get_event_loop().time()
    duration = end_time - start_time

    logger.info(f"并发加载完成！耗时: {duration:.2f}秒")
    logger.info(f"结果: {results}")

    # 列出所有模型
    models = await manager.list_models()
    logger.info(f"已加载的模型: {len(models)}个")

    # 清理
    await manager.shutdown()


async def example_non_blocking_operations():
    """非阻塞操作示例：在模型加载期间执行其他操作。"""
    logger.info("=== 非阻塞操作示例 ===")

    manager = AsyncModelManager()

    # 配置一个大模型（假设加载需要较长时间）
    model_config = ModelConfig(
        model_id="large_model",
        model_type="qwen2",
        component="trunk",
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu"
    )

    # 启动加载任务（不等待）
    logger.info("开始加载大模型（不等待）...")
    load_task = asyncio.create_task(manager.load_model(model_config))

    # 在模型加载期间，执行其他操作！
    for i in range(5):
        await asyncio.sleep(1)

        # 列出模型（不会被阻塞！）
        models = await manager.list_models()
        logger.info(f"第{i+1}秒 - 已加载模型: {len([m for m in models if m['status'] == 'ready'])}个")
        logger.info(f"       - 正在加载: {len([m for m in models if m['status'] == 'loading'])}个")

    # 等待加载完成
    try:
        await load_task
        logger.info("模型加载完成！")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")

    # 清理
    await manager.shutdown()


async def example_with_context_manager():
    """上下文管理器示例：自动管理服务器生命周期。"""
    logger.info("=== 上下文管理器示例 ===")

    server_config = ServerConfig(
        host="0.0.0.0",
        port=50052,  # 使用不同的端口
        max_models=3
    )

    # 使用 async with 自动管理生命周期
    async with AsyncManagedServer(config=server_config) as server:
        logger.info("服务器已自动启动")

        # 加载模型
        model_config = ModelConfig(
            model_id="test_model",
            model_type="gpt2",
            component="bottom",
            model_name_or_path="gpt2",
            device="cpu"
        )

        await server.load_model(model_config)
        logger.info("模型已加载")

        # 模拟一些工作
        await asyncio.sleep(2)

        logger.info("工作完成")

    # 退出 async with 块时，服务器会自动停止
    logger.info("服务器已自动停止")


async def main():
    """运行所有示例。"""
    try:
        # 选择要运行的示例
        logger.info("选择要运行的示例：")
        logger.info("1. 基本使用")
        logger.info("2. 并发加载")
        logger.info("3. 非阻塞操作")
        logger.info("4. 上下文管理器")
        logger.info("5. 运行所有示例")

        # 为了自动化演示，这里运行示例2和3
        # 在实际使用中，你可以选择运行哪个示例

        # await example_basic_usage()
        await example_concurrent_loading()
        await asyncio.sleep(1)
        await example_non_blocking_operations()
        await asyncio.sleep(1)
        await example_with_context_manager()

    except Exception as e:
        logger.error(f"示例运行失败: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("示例被用户中断")
