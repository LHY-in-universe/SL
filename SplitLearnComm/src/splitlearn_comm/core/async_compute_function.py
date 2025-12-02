"""
异步计算函数抽象基类。

这个模块定义了异步计算函数的接口，用于在 gRPC 服务器中执行异步计算。
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


class AsyncComputeFunction(ABC):
    """
    异步计算函数抽象基类。

    所有需要在异步 gRPC 服务器中使用的计算函数都应该继承此类。
    """

    @abstractmethod
    async def compute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        异步执行计算。

        Args:
            input_tensor: 输入张量

        Returns:
            输出张量
        """
        pass

    async def setup(self):
        """
        异步初始化（可选）。

        在服务器启动时调用，可用于加载模型等耗时操作。
        """
        pass

    async def teardown(self):
        """
        异步清理（可选）。

        在服务器关闭时调用，可用于释放资源。
        """
        pass

    def get_info(self) -> Dict:
        """
        获取服务信息（同步）。

        Returns:
            服务信息字典
        """
        return {
            "type": self.__class__.__name__,
            "async": True,
        }


class AsyncModelComputeFunction(AsyncComputeFunction):
    """
    异步模型计算函数。

    包装一个 PyTorch 模型，提供异步推理接口。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        executor: Optional[asyncio.Executor] = None
    ):
        """
        初始化异步模型计算函数。

        Args:
            model: PyTorch 模型
            device: 设备（"cuda" 或 "cpu"）
            executor: 可选的 Executor，用于运行同步代码
        """
        self.model = model
        self.device = device
        self.executor = executor
        self.model.eval()  # 设置为评估模式

        logger.info(
            f"AsyncModelComputeFunction initialized "
            f"(device={device}, model={type(model).__name__})"
        )

    async def compute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        异步执行模型推理。

        注意：PyTorch 推理本身是同步的，但我们使用 run_in_executor
        在后台线程中执行，避免阻塞事件循环。

        Args:
            input_tensor: 输入张量

        Returns:
            输出张量
        """
        loop = asyncio.get_event_loop()

        def _sync_compute():
            """同步计算函数，在 executor 中运行。"""
            with torch.no_grad():
                # 将输入移动到正确的设备
                input_on_device = input_tensor.to(self.device)

                # 执行推理
                output = self.model(input_on_device)

                # 将输出移动回 CPU（如果需要）
                # 这样可以避免 GPU 内存一直被占用
                if self.device.startswith("cuda"):
                    output = output.cpu()

                return output

        # 在 executor 中异步执行
        output_tensor = await loop.run_in_executor(
            self.executor,
            _sync_compute
        )

        return output_tensor

    def get_info(self) -> Dict:
        """获取模型信息。"""
        info = super().get_info()
        info.update({
            "model_type": type(self.model).__name__,
            "device": self.device,
        })
        return info


class AsyncLambdaComputeFunction(AsyncComputeFunction):
    """
    异步 Lambda 计算函数。

    允许使用自定义的异步函数作为计算函数。
    """

    def __init__(self, func):
        """
        初始化 Lambda 计算函数。

        Args:
            func: 异步函数，签名为 async def func(input_tensor) -> output_tensor
        """
        self.func = func

    async def compute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """执行自定义异步函数。"""
        return await self.func(input_tensor)


class AsyncChainComputeFunction(AsyncComputeFunction):
    """
    异步链式计算函数。

    将多个计算函数链接起来，按顺序执行。
    """

    def __init__(self, functions: list):
        """
        初始化链式计算函数。

        Args:
            functions: AsyncComputeFunction 列表
        """
        self.functions = functions

    async def compute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """按顺序执行所有计算函数。"""
        output = input_tensor
        for func in self.functions:
            output = await func.compute(output)
        return output

    async def setup(self):
        """初始化所有计算函数。"""
        for func in self.functions:
            await func.setup()

    async def teardown(self):
        """清理所有计算函数。"""
        for func in self.functions:
            await func.teardown()

    def get_info(self) -> Dict:
        """获取链式信息。"""
        return {
            "type": "AsyncChainComputeFunction",
            "functions": [f.get_info() for f in self.functions],
            "async": True,
        }
