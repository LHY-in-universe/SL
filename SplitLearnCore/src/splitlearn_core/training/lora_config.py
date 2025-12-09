"""
LoRA 配置和应用工具
用于在分布式训练中为 Bottom/Trunk/Top 模型应用 LoRA 适配器
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# 检查 PEFT 是否可用
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT library not available. LoRA functionality will be disabled.")


@dataclass
class SplitLoraConfig:
    """
    分布式训练的 LoRA 配置

    针对不同模型架构提供预设的 target_modules

    Example:
        >>> # GPT-2
        >>> config = SplitLoraConfig.for_gpt2(rank=8)
        >>>
        >>> # Qwen3-VL
        >>> config = SplitLoraConfig.for_qwen3_vl(rank=16)
        >>>
        >>> # 自定义
        >>> config = SplitLoraConfig(
        ...     rank=8,
        ...     target_modules=["q_proj", "v_proj"]
        ... )
    """

    # LoRA 参数
    rank: int = 8
    """LoRA 秩（低秩矩阵的维度），通常 4-16"""

    alpha: int = 16
    """LoRA 缩放因子，通常设为 rank 的 2 倍"""

    dropout: float = 0.1
    """LoRA 层的 dropout 概率"""

    target_modules: Optional[List[str]] = None
    """要应用 LoRA 的模块名称列表"""

    bias: str = "none"
    """Bias 训练策略: "none", "all", "lora_only" """

    task_type: str = "CAUSAL_LM"
    """任务类型，用于 PEFT"""

    # 训练参数
    learning_rate: float = 1e-4
    """学习率"""

    weight_decay: float = 0.01
    """权重衰减"""

    # 元数据
    modules_to_save: Optional[List[str]] = None
    """额外需要保存的模块（非 LoRA 但需要训练）"""

    @classmethod
    def for_gpt2(cls, rank: int = 8, alpha: Optional[int] = None) -> 'SplitLoraConfig':
        """
        GPT-2 模型的 LoRA 配置

        Args:
            rank: LoRA 秩
            alpha: LoRA alpha（默认为 rank * 2）
        """
        return cls(
            rank=rank,
            alpha=alpha or rank * 2,
            target_modules=[
                "c_attn",   # QKV 投影（合并的）
                "c_proj",   # 注意力输出投影
                "c_fc",     # MLP 第一层
            ],
        )

    @classmethod
    def for_qwen2(cls, rank: int = 8, alpha: Optional[int] = None) -> 'SplitLoraConfig':
        """
        Qwen2 模型的 LoRA 配置

        Args:
            rank: LoRA 秩
            alpha: LoRA alpha（默认为 rank * 2）
        """
        return cls(
            rank=rank,
            alpha=alpha or rank * 2,
            target_modules=[
                "q_proj",      # Query 投影
                "k_proj",      # Key 投影
                "v_proj",      # Value 投影
                "o_proj",      # 输出投影
                "gate_proj",   # MLP gate
                "up_proj",     # MLP up
                "down_proj",   # MLP down
            ],
        )

    @classmethod
    def for_qwen3_vl(cls, rank: int = 8, alpha: Optional[int] = None) -> 'SplitLoraConfig':
        """
        Qwen3-VL 模型的 LoRA 配置

        Args:
            rank: LoRA 秩
            alpha: LoRA alpha（默认为 rank * 2）
        """
        return cls(
            rank=rank,
            alpha=alpha or rank * 2,
            target_modules=[
                "q_proj",      # Query 投影
                "k_proj",      # Key 投影
                "v_proj",      # Value 投影
                "o_proj",      # 输出投影
                "gate_proj",   # MLP gate
                "up_proj",     # MLP up
                "down_proj",   # MLP down
            ],
        )

    @classmethod
    def for_gemma(cls, rank: int = 8, alpha: Optional[int] = None) -> 'SplitLoraConfig':
        """
        Gemma 模型的 LoRA 配置

        Args:
            rank: LoRA 秩
            alpha: LoRA alpha（默认为 rank * 2）
        """
        return cls(
            rank=rank,
            alpha=alpha or rank * 2,
            target_modules=[
                "q_proj",      # Query 投影
                "k_proj",      # Key 投影
                "v_proj",      # Value 投影
                "o_proj",      # 输出投影
                "gate_proj",   # MLP gate
                "up_proj",     # MLP up
                "down_proj",   # MLP down
            ],
        )

    def to_peft_config(self) -> 'LoraConfig':
        """
        转换为 PEFT LoraConfig

        Returns:
            PEFT LoraConfig 对象

        Raises:
            ImportError: 如果 PEFT 未安装
        """
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT library is required for LoRA. "
                "Install with: pip install peft"
            )

        # 映射 task_type 字符串到 PEFT TaskType
        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
            "TOKEN_CLS": TaskType.TOKEN_CLS,
            "SEQ_CLS": TaskType.SEQ_CLS,
        }

        task_type = task_type_map.get(self.task_type, TaskType.CAUSAL_LM)

        return LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            modules_to_save=self.modules_to_save,
        )

    def get_optimizer_params(self) -> Dict[str, Any]:
        """
        获取优化器参数

        Returns:
            优化器参数字典
        """
        return {
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay,
        }


def apply_lora_to_model(
    model: nn.Module,
    lora_config: SplitLoraConfig,
    freeze_base: bool = True
) -> nn.Module:
    """
    为模型应用 LoRA 适配器并冻结基础参数

    Args:
        model: 要应用 LoRA 的模型
        lora_config: LoRA 配置
        freeze_base: 是否冻结基础模型参数（默认 True）

    Returns:
        应用了 LoRA 的模型

    Raises:
        ImportError: 如果 PEFT 未安装

    Example:
        >>> from splitlearn_core import ModelFactory
        >>> from splitlearn_core.training import SplitLoraConfig, apply_lora_to_model
        >>>
        >>> # 创建分割模型
        >>> bottom, trunk, top = ModelFactory.create_split_models(
        ...     model_type='qwen3_vl',
        ...     split_point_1=0,
        ...     split_point_2=14
        ... )
        >>>
        >>> # 应用 LoRA
        >>> lora_config = SplitLoraConfig.for_qwen3_vl(rank=8)
        >>> bottom_lora = apply_lora_to_model(bottom, lora_config)
        >>> trunk_lora = apply_lora_to_model(trunk, lora_config)
        >>> top_lora = apply_lora_to_model(top, lora_config)
        >>>
        >>> # 验证参数
        >>> trainable = sum(p.numel() for p in bottom_lora.parameters() if p.requires_grad)
        >>> total = sum(p.numel() for p in bottom_lora.parameters())
        >>> print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT library is required for LoRA. "
            "Install with: pip install peft"
        )

    # 1. 如果需要，先冻结所有基础参数
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False
        logger.info("Froze all base model parameters")

    # 2. 应用 LoRA
    peft_config = lora_config.to_peft_config()
    model_with_lora = get_peft_model(model, peft_config)

    # 3. 统计参数
    trainable_params = sum(
        p.numel() for p in model_with_lora.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model_with_lora.parameters())

    logger.info(
        f"Applied LoRA: rank={lora_config.rank}, alpha={lora_config.alpha}"
    )
    logger.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({trainable_params/total_params*100:.2f}%)"
    )

    # 4. 列出 LoRA 模块
    lora_modules = [
        name for name, param in model_with_lora.named_parameters()
        if param.requires_grad and 'lora' in name.lower()
    ]
    logger.debug(f"LoRA modules: {lora_modules[:5]}...")  # 只显示前 5 个

    return model_with_lora


def print_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    """
    打印模型的可训练参数统计

    Args:
        model: 模型

    Returns:
        统计信息字典

    Example:
        >>> stats = print_trainable_parameters(model_with_lora)
        >>> print(f"Trainable: {stats['trainable']:,}")
    """
    trainable_params = 0
    all_params = 0

    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            logger.debug(f"  Trainable: {name} - {param.numel():,} params")

    percentage = 100 * trainable_params / all_params if all_params > 0 else 0

    logger.info(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_params:,} || "
        f"Trainable%: {percentage:.2f}%"
    )

    return {
        'trainable': trainable_params,
        'total': all_params,
        'percentage': percentage
    }


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    合并 LoRA 权重到基础模型

    用于推理时加速：将 LoRA 适配器权重合并到基础模型中，
    避免推理时的额外计算开销。

    Args:
        model: 应用了 LoRA 的模型

    Returns:
        合并后的模型

    Warning:
        合并后无法继续训练 LoRA 适配器

    Example:
        >>> # 训练完成后
        >>> model_merged = merge_lora_weights(model_with_lora)
        >>> # 现在可以更快地进行推理
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT library required")

    if not isinstance(model, PeftModel):
        logger.warning("Model is not a PeftModel, skipping merge")
        return model

    logger.info("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    logger.info("LoRA weights merged successfully")

    return merged_model
