"""
异步模型生命周期管理。

这个模块提供了 ModelManager 的异步版本，解决了同步版本中的锁阻塞问题。
主要改进：
1. 使用 asyncio.Lock 替代 threading.Lock
2. 模型加载在锁外执行（使用 ThreadPoolExecutor）
3. 占位符机制防止并发加载同一模型
4. 短时间持锁，仅在修改共享状态时

这使得在模型加载期间，其他操作（如 list_models）不会被阻塞。
"""

import asyncio
import logging
from typing import Dict, Optional, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import torch.nn as nn

from ..config import ModelConfig
from .model_loader import ModelLoader
from .resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """模型状态枚举。"""
    LOADING = "loading"
    READY = "ready"
    UNLOADING = "unloading"
    FAILED = "failed"


class LoadingPlaceholder:
    """
    模型加载占位符。

    在模型加载过程中，用此占位符标记模型正在加载，
    防止重复加载同一模型。
    """

    def __init__(self, model_id: str, config: ModelConfig):
        self.model_id = model_id
        self.config = config
        self.status = ModelStatus.LOADING
        self.loaded_at = datetime.now()


class AsyncManagedModel:
    """
    异步托管模型包装器，包含元数据。

    Attributes:
        model_id: 唯一模型标识符
        model: PyTorch 模型
        config: 模型配置
        status: 当前状态
        loaded_at: 加载时间戳
        request_count: 已处理的请求总数
        last_used: 最后请求时间戳
        lock: 用于保护计数器的细粒度锁
    """

    def __init__(self, model_id: str, model: nn.Module, config: ModelConfig):
        self.model_id = model_id
        self.model = model
        self.config = config
        self.status = ModelStatus.READY
        self.loaded_at = datetime.now()
        self.request_count = 0
        self.last_used = datetime.now()
        # 使用 asyncio.Lock 保护计数器
        self.lock = asyncio.Lock()

    def get_info(self) -> Dict:
        """获取模型信息。"""
        return {
            "model_id": self.model_id,
            "status": self.status.value,
            "loaded_at": self.loaded_at.isoformat(),
            "last_used": self.last_used.isoformat(),
            "request_count": self.request_count,
            "device": self.config.device,
            "model_type": self.config.model_type,
        }

    async def increment_request_count(self):
        """异步增加请求计数器并更新 last_used。"""
        async with self.lock:
            self.request_count += 1
            self.last_used = datetime.now()


class AsyncModelManager:
    """
    异步模型管理器：管理模型生命周期（加载、卸载和监控）。

    主要特性：
    - 按需异步加载/卸载模型
    - 资源管理
    - 模型状态跟踪
    - 异步安全操作
    - 解决锁阻塞问题：模型加载在锁外执行

    关键设计：
    1. 使用占位符机制：在开始加载前，在锁内放置占位符
    2. 锁外加载：使用 ThreadPoolExecutor 在后台加载模型
    3. 短时间持锁：只在修改 self.models 字典时持有锁
    4. 这样 list_models() 等操作在模型加载期间不会被阻塞
    """

    def __init__(
        self,
        resource_manager: Optional[ResourceManager] = None,
        max_models: int = 5,
        max_workers: int = 4
    ):
        """
        初始化异步模型管理器。

        Args:
            resource_manager: 资源管理器实例
            max_models: 最大加载模型数
            max_workers: 线程池最大工作线程数（用于模型加载）
        """
        self.models: Dict[str, AsyncManagedModel | LoadingPlaceholder] = {}
        self.resource_manager = resource_manager or ResourceManager()
        self.max_models = max_models

        # 使用 asyncio.Lock 而非 threading.Lock
        self.lock = asyncio.Lock()

        # 用于 CPU/IO 密集型操作的线程池
        # PyTorch 模型加载主要是 IO（读文件）和 GPU 操作
        # GPU 操作会释放 GIL，因此使用 ThreadPoolExecutor 是合适的
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self.loader = ModelLoader()

        logger.info(
            f"AsyncModelManager initialized "
            f"(max_models={max_models}, max_workers={max_workers})"
        )

    async def load_model(self, config: ModelConfig) -> bool:
        """
        异步加载模型。

        此方法实现了关键的锁优化：
        1. 短暂持锁：检查状态并添加占位符（<1ms）
        2. 锁外加载：在 ThreadPoolExecutor 中加载模型（几秒到几分钟）
        3. 短暂持锁：更新为真实模型（<1ms）

        这样，在模型加载期间，其他操作不会被阻塞。

        Args:
            config: 模型配置

        Returns:
            True 表示加载成功

        Raises:
            ValueError: 如果 model_id 已存在
            RuntimeError: 如果加载失败
        """
        # 在锁外验证配置（快速操作）
        config.validate()

        # ============ 第一步：短暂持锁，添加占位符 ============
        async with self.lock:
            # 检查是否已加载
            if config.model_id in self.models:
                existing = self.models[config.model_id]
                if isinstance(existing, LoadingPlaceholder):
                    raise ValueError(
                        f"Model {config.model_id} is already loading"
                    )
                else:
                    raise ValueError(
                        f"Model {config.model_id} already loaded"
                    )

            # 检查是否需要卸载 LRU 模型
            if len(self.models) >= self.max_models:
                logger.info(
                    f"Max models ({self.max_models}) reached, "
                    f"will unload LRU model"
                )
                await self._unload_lru_model_unsafe()

            # 检查资源
            if config.max_memory_mb:
                if not self.resource_manager.check_available_resources(
                    required_memory_mb=config.max_memory_mb,
                    required_gpu=config.device.startswith("cuda")
                ):
                    raise RuntimeError(
                        "Insufficient resources to load model"
                    )

            # 在 models dict 中放置占位符
            # 这样其他并发的 load_model 调用会看到这个模型正在加载
            placeholder = LoadingPlaceholder(
                model_id=config.model_id,
                config=config
            )
            self.models[config.model_id] = placeholder

            logger.info(f"Placeholder set for model {config.model_id}")

        # ============ 第二步：锁外加载模型（耗时操作） ============
        logger.info(f"Loading model {config.model_id} in background...")

        try:
            # 在 ThreadPoolExecutor 中异步执行模型加载
            # 这是一个阻塞操作，但不会阻塞事件循环
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                self.executor,
                self.loader.load_from_config,
                config
            )

            # ============ 第三步：短暂持锁，更新为真实模型 ============
            async with self.lock:
                # 创建托管模型
                managed_model = AsyncManagedModel(
                    model_id=config.model_id,
                    model=model,
                    config=config
                )

                # 替换占位符
                self.models[config.model_id] = managed_model

                logger.info(
                    f"Successfully loaded model {config.model_id} "
                    f"on {config.device}"
                )

            # 在锁外记录资源使用（可能需要一些时间）
            self.resource_manager.log_resource_usage()

            return True

        except Exception as e:
            # 加载失败，清理占位符
            logger.error(f"Failed to load model {config.model_id}: {e}")

            async with self.lock:
                if config.model_id in self.models:
                    del self.models[config.model_id]

            raise RuntimeError(f"Model loading failed: {e}")

    async def unload_model(self, model_id: str) -> bool:
        """
        异步卸载模型。

        Args:
            model_id: 模型标识符

        Returns:
            True 表示卸载成功

        Raises:
            KeyError: 如果模型未找到
        """
        async with self.lock:
            if model_id not in self.models:
                raise KeyError(f"Model {model_id} not found")

            model_entry = self.models[model_id]

            # 不能卸载正在加载的模型
            if isinstance(model_entry, LoadingPlaceholder):
                raise RuntimeError(
                    f"Cannot unload model {model_id} while it's loading"
                )

            logger.info(f"Unloading model {model_id}...")

            managed_model = model_entry
            managed_model.status = ModelStatus.UNLOADING

            try:
                # 删除模型以释放内存
                del managed_model.model

                # 从注册表中移除
                del self.models[model_id]

                # 对 GPU 内存强制垃圾回收
                if managed_model.config.device.startswith("cuda"):
                    import torch
                    torch.cuda.empty_cache()

                logger.info(f"Successfully unloaded model {model_id}")

                # 在锁外记录资源使用
                # 注意：这里我们仍在锁内，但可以考虑移到锁外
                self.resource_manager.log_resource_usage()

                return True

            except Exception as e:
                logger.error(f"Error unloading model {model_id}: {e}")
                managed_model.status = ModelStatus.FAILED
                return False

    async def get_model(self, model_id: str) -> Optional[AsyncManagedModel]:
        """
        异步获取托管模型。

        注意：此方法不需要锁，因为：
        1. 字典读取在 Python 中是原子的
        2. 我们只是获取引用，不修改字典
        3. 计数器更新使用模型自己的细粒度锁

        Args:
            model_id: 模型标识符

        Returns:
            AsyncManagedModel 或 None（如果未找到或正在加载）
        """
        model_entry = self.models.get(model_id)

        # 如果是占位符，返回 None
        if isinstance(model_entry, LoadingPlaceholder):
            logger.debug(f"Model {model_id} is still loading")
            return None

        if model_entry:
            await model_entry.increment_request_count()

        return model_entry

    async def list_models(self) -> List[Dict]:
        """
        列出所有已加载的模型及其信息。

        关键改进：此方法现在非常快（<1ms），即使有模型正在加载。

        Returns:
            模型信息字典列表
        """
        async with self.lock:
            result = []
            for model_id, model_entry in self.models.items():
                if isinstance(model_entry, LoadingPlaceholder):
                    # 为占位符返回基本信息
                    result.append({
                        "model_id": model_id,
                        "status": ModelStatus.LOADING.value,
                        "loaded_at": model_entry.loaded_at.isoformat(),
                        "device": model_entry.config.device,
                        "model_type": model_entry.config.model_type,
                    })
                else:
                    result.append(model_entry.get_info())
            return result

    async def get_model_info(self, model_id: str) -> Dict:
        """
        获取模型的详细信息。

        Args:
            model_id: 模型标识符

        Returns:
            模型信息字典

        Raises:
            KeyError: 如果模型未找到
        """
        if model_id not in self.models:
            raise KeyError(f"Model {model_id} not found")

        model_entry = self.models[model_id]

        if isinstance(model_entry, LoadingPlaceholder):
            return {
                "model_id": model_id,
                "status": ModelStatus.LOADING.value,
                "loaded_at": model_entry.loaded_at.isoformat(),
                "device": model_entry.config.device,
                "model_type": model_entry.config.model_type,
            }

        managed_model = model_entry
        info = managed_model.get_info()

        # 添加额外信息
        loader_info = self.loader.get_model_info(
            managed_model.model,
            managed_model.config.device
        )
        info.update(loader_info)

        return info

    async def reload_model(self, model_id: str) -> bool:
        """
        重新加载模型（先卸载再加载）。

        Args:
            model_id: 模型标识符

        Returns:
            True 表示重新加载成功
        """
        if model_id not in self.models:
            raise KeyError(f"Model {model_id} not found")

        logger.info(f"Reloading model {model_id}...")

        # 保存配置
        model_entry = self.models[model_id]
        if isinstance(model_entry, LoadingPlaceholder):
            config = model_entry.config
        else:
            config = model_entry.config

        # 卸载
        await self.unload_model(model_id)

        # 加载
        return await self.load_model(config)

    async def _unload_lru_model_unsafe(self) -> None:
        """
        卸载最近最少使用的模型。

        注意：此方法假设调用者已持有锁（unsafe）。
        """
        if not self.models:
            return

        # 找到 LRU 模型（跳过占位符）
        real_models = {
            k: v for k, v in self.models.items()
            if not isinstance(v, LoadingPlaceholder)
        }

        if not real_models:
            logger.warning("All models are loading placeholders, cannot unload LRU")
            return

        lru_model_id = min(
            real_models.keys(),
            key=lambda k: real_models[k].last_used
        )

        logger.info(f"Unloading LRU model: {lru_model_id}")

        # 注意：我们已经持有锁，所以直接操作
        managed_model = self.models[lru_model_id]
        managed_model.status = ModelStatus.UNLOADING

        try:
            del managed_model.model
            del self.models[lru_model_id]

            if managed_model.config.device.startswith("cuda"):
                import torch
                torch.cuda.empty_cache()

            logger.info(f"Successfully unloaded LRU model {lru_model_id}")

        except Exception as e:
            logger.error(f"Error unloading LRU model {lru_model_id}: {e}")

    async def get_statistics(self) -> Dict:
        """
        获取总体统计信息。

        Returns:
            统计信息字典
        """
        async with self.lock:
            # 过滤掉占位符
            real_models = {
                k: v for k, v in self.models.items()
                if not isinstance(v, LoadingPlaceholder)
            }

            total_requests = sum(
                model.request_count for model in real_models.values()
            )

            return {
                "total_models": len(real_models),
                "loading_models": len(self.models) - len(real_models),
                "max_models": self.max_models,
                "total_requests": total_requests,
                "models": {
                    model_id: {
                        "request_count": model.request_count,
                        "status": model.status.value,
                    }
                    for model_id, model in real_models.items()
                }
            }

    async def shutdown(self) -> None:
        """关闭管理器并卸载所有模型。"""
        logger.info("Shutting down AsyncModelManager...")

        async with self.lock:
            model_ids = list(self.models.keys())

        for model_id in model_ids:
            model_entry = self.models.get(model_id)

            # 跳过占位符
            if isinstance(model_entry, LoadingPlaceholder):
                logger.warning(
                    f"Model {model_id} is still loading during shutdown, "
                    f"removing placeholder"
                )
                async with self.lock:
                    if model_id in self.models:
                        del self.models[model_id]
                continue

            try:
                await self.unload_model(model_id)
            except Exception as e:
                logger.error(
                    f"Error unloading {model_id} during shutdown: {e}"
                )

        # 关闭线程池
        self.executor.shutdown(wait=True)

        logger.info("AsyncModelManager shutdown complete")
