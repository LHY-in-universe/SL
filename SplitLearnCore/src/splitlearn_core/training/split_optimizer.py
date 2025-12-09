"""
分布式优化器
用于管理 Bottom/Top（客户端）和 Trunk（服务端）的优化器
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """优化器配置"""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    grad_clip_norm: Optional[float] = 1.0  # 梯度裁剪


class SplitOptimizer:
    """
    分布式训练的优化器管理器

    管理客户端的 Bottom 和 Top 模型优化器。
    Trunk 模型在服务端有独立的优化器（在 servicer 中管理）。

    Example:
        >>> from splitlearn_core import ModelFactory
        >>> from splitlearn_core.training import SplitLoraConfig, apply_lora_to_model, SplitOptimizer
        >>>
        >>> # 创建模型并应用 LoRA
        >>> bottom, trunk, top = ModelFactory.create_split_models(...)
        >>> lora_config = SplitLoraConfig.for_qwen3_vl()
        >>> bottom = apply_lora_to_model(bottom, lora_config)
        >>> top = apply_lora_to_model(top, lora_config)
        >>>
        >>> # 创建优化器
        >>> optimizer = SplitOptimizer(bottom, top)
        >>>
        >>> # 训练循环
        >>> for batch in dataloader:
        ...     # 前向+反向传播
        ...     loss = train_step(batch)
        ...
        ...     # 更新参数
        ...     optimizer.step()
        ...     optimizer.zero_grad()
    """

    def __init__(
        self,
        bottom_model: Optional[nn.Module] = None,
        top_model: Optional[nn.Module] = None,
        config: Optional[OptimizerConfig] = None,
        separate_lr: bool = False,
        bottom_lr: Optional[float] = None,
        top_lr: Optional[float] = None,
    ):
        """
        Args:
            bottom_model: Bottom 模型（客户端）
            top_model: Top 模型（客户端）
            config: 优化器配置
            separate_lr: 是否为 Bottom 和 Top 使用不同学习率
            bottom_lr: Bottom 学习率（如果 separate_lr=True）
            top_lr: Top 学习率（如果 separate_lr=True）
        """
        self.bottom_model = bottom_model
        self.top_model = top_model
        self.config = config or OptimizerConfig()

        # 创建优化器
        self.optimizer_bottom = None
        self.optimizer_top = None

        if bottom_model is not None:
            bottom_params = self._get_trainable_params(bottom_model)
            lr = bottom_lr if separate_lr and bottom_lr else self.config.learning_rate
            self.optimizer_bottom = AdamW(
                bottom_params,
                lr=lr,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas,
                eps=self.config.eps
            )
            logger.info(
                f"Created Bottom optimizer: "
                f"lr={lr}, trainable_params={len(bottom_params)}"
            )

        if top_model is not None:
            top_params = self._get_trainable_params(top_model)
            lr = top_lr if separate_lr and top_lr else self.config.learning_rate
            self.optimizer_top = AdamW(
                top_params,
                lr=lr,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas,
                eps=self.config.eps
            )
            logger.info(
                f"Created Top optimizer: "
                f"lr={lr}, trainable_params={len(top_params)}"
            )

        # 统计信息
        self.steps = 0
        self.total_grad_norm_bottom = 0.0
        self.total_grad_norm_top = 0.0

    @staticmethod
    def _get_trainable_params(model: nn.Module) -> List[torch.nn.Parameter]:
        """获取可训练参数列表"""
        return [p for p in model.parameters() if p.requires_grad]

    def zero_grad(self):
        """清零所有梯度"""
        if self.optimizer_bottom is not None:
            self.optimizer_bottom.zero_grad()
        if self.optimizer_top is not None:
            self.optimizer_top.zero_grad()

    def step(self):
        """
        执行优化器步骤

        包括：
        1. 梯度裁剪（如果启用）
        2. 优化器更新
        3. 统计信息收集
        """
        # 梯度裁剪
        if self.config.grad_clip_norm is not None:
            if self.bottom_model is not None and self.optimizer_bottom is not None:
                grad_norm_bottom = torch.nn.utils.clip_grad_norm_(
                    self.bottom_model.parameters(),
                    self.config.grad_clip_norm
                )
                self.total_grad_norm_bottom += grad_norm_bottom.item()
                logger.debug(f"Bottom grad norm: {grad_norm_bottom:.4f}")

            if self.top_model is not None and self.optimizer_top is not None:
                grad_norm_top = torch.nn.utils.clip_grad_norm_(
                    self.top_model.parameters(),
                    self.config.grad_clip_norm
                )
                self.total_grad_norm_top += grad_norm_top.item()
                logger.debug(f"Top grad norm: {grad_norm_top:.4f}")

        # 优化器步骤
        if self.optimizer_bottom is not None:
            self.optimizer_bottom.step()
        if self.optimizer_top is not None:
            self.optimizer_top.step()

        self.steps += 1

    def get_lr(self) -> Dict[str, float]:
        """
        获取当前学习率

        Returns:
            字典包含 'bottom' 和 'top' 的学习率
        """
        lr_dict = {}
        if self.optimizer_bottom is not None:
            lr_dict['bottom'] = self.optimizer_bottom.param_groups[0]['lr']
        if self.optimizer_top is not None:
            lr_dict['top'] = self.optimizer_top.param_groups[0]['lr']
        return lr_dict

    def set_lr(self, lr: float, component: Optional[str] = None):
        """
        设置学习率

        Args:
            lr: 新的学习率
            component: 要设置的组件 ('bottom', 'top', None 表示全部)
        """
        if component is None or component == 'bottom':
            if self.optimizer_bottom is not None:
                for param_group in self.optimizer_bottom.param_groups:
                    param_group['lr'] = lr
                logger.info(f"Set Bottom LR to {lr}")

        if component is None or component == 'top':
            if self.optimizer_top is not None:
                for param_group in self.optimizer_top.param_groups:
                    param_group['lr'] = lr
                logger.info(f"Set Top LR to {lr}")

    def state_dict(self) -> Dict[str, Any]:
        """
        保存优化器状态

        Returns:
            状态字典
        """
        state = {
            'steps': self.steps,
            'total_grad_norm_bottom': self.total_grad_norm_bottom,
            'total_grad_norm_top': self.total_grad_norm_top,
        }

        if self.optimizer_bottom is not None:
            state['optimizer_bottom'] = self.optimizer_bottom.state_dict()
        if self.optimizer_top is not None:
            state['optimizer_top'] = self.optimizer_top.state_dict()

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        加载优化器状态

        Args:
            state_dict: 状态字典
        """
        self.steps = state_dict.get('steps', 0)
        self.total_grad_norm_bottom = state_dict.get('total_grad_norm_bottom', 0.0)
        self.total_grad_norm_top = state_dict.get('total_grad_norm_top', 0.0)

        if 'optimizer_bottom' in state_dict and self.optimizer_bottom is not None:
            self.optimizer_bottom.load_state_dict(state_dict['optimizer_bottom'])
            logger.info("Loaded Bottom optimizer state")

        if 'optimizer_top' in state_dict and self.optimizer_top is not None:
            self.optimizer_top.load_state_dict(state_dict['optimizer_top'])
            logger.info("Loaded Top optimizer state")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取优化器统计信息

        Returns:
            统计信息字典
        """
        stats = {
            'steps': self.steps,
            'learning_rates': self.get_lr(),
        }

        if self.steps > 0:
            stats['avg_grad_norm_bottom'] = self.total_grad_norm_bottom / self.steps
            stats['avg_grad_norm_top'] = self.total_grad_norm_top / self.steps

        return stats

    def __repr__(self) -> str:
        return (
            f"SplitOptimizer("
            f"steps={self.steps}, "
            f"lr_bottom={self.get_lr().get('bottom', 'N/A')}, "
            f"lr_top={self.get_lr().get('top', 'N/A')})"
        )


def create_scheduler(
    optimizer: SplitOptimizer,
    scheduler_type: str = 'cosine',
    num_training_steps: int = 10000,
    num_warmup_steps: int = 100,
) -> Dict[str, Any]:
    """
    为 SplitOptimizer 创建学习率调度器

    Args:
        optimizer: SplitOptimizer 实例
        scheduler_type: 调度器类型 ('cosine', 'linear', 'constant')
        num_training_steps: 总训练步数
        num_warmup_steps: 预热步数

    Returns:
        包含调度器的字典

    Example:
        >>> optimizer = SplitOptimizer(bottom, top)
        >>> schedulers = create_scheduler(optimizer, 'cosine', num_training_steps=10000)
        >>>
        >>> # 训练循环中
        >>> optimizer.step()
        >>> for scheduler in schedulers.values():
        ...     scheduler.step()
    """
    try:
        from transformers import get_scheduler
    except ImportError:
        logger.warning(
            "transformers not available, using torch.optim schedulers. "
            "Install transformers for more scheduler options."
        )
        from torch.optim.lr_scheduler import CosineAnnealingLR

        schedulers = {}
        if optimizer.optimizer_bottom is not None:
            schedulers['bottom'] = CosineAnnealingLR(
                optimizer.optimizer_bottom,
                T_max=num_training_steps
            )
        if optimizer.optimizer_top is not None:
            schedulers['top'] = CosineAnnealingLR(
                optimizer.optimizer_top,
                T_max=num_training_steps
            )
        return schedulers

    # 使用 transformers 的调度器
    schedulers = {}

    if optimizer.optimizer_bottom is not None:
        schedulers['bottom'] = get_scheduler(
            scheduler_type,
            optimizer=optimizer.optimizer_bottom,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        logger.info(f"Created {scheduler_type} scheduler for Bottom")

    if optimizer.optimizer_top is not None:
        schedulers['top'] = get_scheduler(
            scheduler_type,
            optimizer=optimizer.optimizer_top,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        logger.info(f"Created {scheduler_type} scheduler for Top")

    return schedulers
