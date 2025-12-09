"""
ComputeFunction - 计算函数抽象接口

这个抽象类完全解耦了模型依赖，允许用户传入任意计算函数。
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch


class ComputeFunction(ABC):
    """
    计算函数抽象基类

    用户需要实现 compute() 方法来定义具体的计算逻辑。
    这个设计使得通信库完全独立于模型实现。

    Example:
        >>> class MyCompute(ComputeFunction):
        ...     def __init__(self, model):
        ...         self.model = model
        ...
        ...     def compute(self, input_tensor):
        ...         with torch.no_grad():
        ...             return self.model(input_tensor)
        ...
        >>> compute_fn = MyCompute(my_model)
        >>> output = compute_fn.compute(input_tensor)
    """

    @abstractmethod
    def compute(
        self,
        input_tensor: torch.Tensor,
        past_key_values: Optional[tuple] = None,
        use_cache: bool = False,
        **kwargs
    ):
        """
        执行计算，返回输出张量（可能包含 KV-cache）

        Args:
            input_tensor: 输入张量
            past_key_values: 过去的 Key-Value cache (可选)
            use_cache: 是否返回 present_key_values (可选)
            **kwargs: 其他参数

        Returns:
            如果 use_cache=True:
                (output_tensor, present_key_values)
            否则:
                output_tensor

        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("Subclasses must implement compute()")

    def get_info(self) -> Dict[str, Any]:
        """
        返回服务信息（可选实现）

        Returns:
            包含服务信息的字典，例如：
            {
                "name": "MyComputeFunction",
                "device": "cuda",
                "model_type": "gpt2",
                ...
            }
        """
        return {"name": self.__class__.__name__}

    def setup(self) -> None:
        """
        初始化设置（可选实现）

        在服务器启动时调用，可用于预加载模型、预热等操作。
        """
        pass

    def teardown(self) -> None:
        """
        清理资源（可选实现）

        在服务器关闭时调用，可用于释放资源、保存状态等操作。
        """
        pass


class ModelComputeFunction(ComputeFunction):
    """
    基于 PyTorch 模型的计算函数实现

    这是一个通用的包装器，可以将任意 PyTorch 模型转换为 ComputeFunction。

    Example:
        >>> model = torch.nn.Sequential(
        ...     torch.nn.Linear(768, 768),
        ...     torch.nn.ReLU()
        ... )
        >>> compute_fn = ModelComputeFunction(model, device="cuda")
        >>> output = compute_fn.compute(input_tensor)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        model_name: Optional[str] = None
    ):
        """
        Args:
            model: PyTorch 模型
            device: 设备 ('cpu', 'cuda', 'mps', 等)
            model_name: 模型名称（用于日志和信息）
        """
        self.model = model.to(device).eval()
        self.device = device
        self.model_name = model_name or model.__class__.__name__

    def compute(
        self,
        input_tensor: torch.Tensor,
        past_key_values: Optional[tuple] = None,
        use_cache: bool = False,
        **kwargs
    ):
        """执行模型前向传播（支持 KV-cache）"""
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            # 尝试传递 KV-cache 参数（如果模型支持）
            import inspect
            sig = inspect.signature(self.model.forward)

            # 检查模型是否支持 past_key_values 和 use_cache
            model_kwargs = {}
            if 'past_key_values' in sig.parameters and past_key_values is not None:
                model_kwargs['past_key_values'] = past_key_values
            if 'use_cache' in sig.parameters:
                model_kwargs['use_cache'] = use_cache

            # 调用模型
            output = self.model(input_tensor, **model_kwargs)

            return output

    def get_info(self) -> Dict[str, Any]:
        """返回模型信息"""
        return {
            "name": self.model_name,
            "device": self.device,
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }
