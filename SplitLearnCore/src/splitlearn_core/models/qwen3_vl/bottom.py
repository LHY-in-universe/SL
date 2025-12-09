"""
Qwen3-VL Bottom Model (Vision Front)

职责：仅包含视觉编码器（ViT + patch 合并），输出视觉 token 序列。
实现与 Qwen2-VL 拆分保持一致。
"""
from typing import Optional

import torch
import torch.nn as nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLVisionModel,
)

from ...core import BaseBottomModel
from ...registry import ModelRegistry


@ModelRegistry.register("qwen3_vl", "bottom")
class Qwen3VLBottomModel(BaseBottomModel):
    """
    视觉前端：
      - 仅包含官方视觉塔（patch_embed + vision blocks + patch merger）
      - 不处理文本 token，也不做投影/拼接
    """

    def __init__(self, config: Qwen3VLConfig):
        # 视觉端没有 decoder 层，这里 end_layer=0 以兼容基类
        super().__init__(config, end_layer=0)
        self.visual = Qwen3VLVisionModel(config.vision_config)

    # --- BaseBottomModel 抽象实现（视觉端不使用这些接口，但需满足类型要求） ---
    def get_embeddings(self) -> nn.Module:
        # 视觉端不使用 token embedding
        return nn.Identity()

    def apply_position_encoding(
        self, inputs_embeds: torch.Tensor, position_ids: Optional[torch.LongTensor]
    ) -> torch.Tensor:
        return inputs_embeds

    def get_transformer_blocks(self) -> nn.ModuleList:
        return nn.ModuleList()

    def prepare_attention_mask(
        self, attention_mask: Optional[torch.Tensor], hidden_states: torch.Tensor
    ) -> Optional[torch.Tensor]:
        return None

    def get_layer_name_pattern(self) -> str:
        return r"\.layers\.[0-9]+"

    # --- 自定义前向：接收像素张量并返回视觉特征序列 ---
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: 图像/视频张量，形状遵循官方 Qwen-VL 预处理输出
            grid_thw:      grid 信息（T,H,W），用于 3D RoPE
        Returns:
            视觉特征序列，形状 [num_vision_tokens, hidden_size]
        """
        return self.visual(pixel_values, grid_thw=grid_thw)

    # --- 权重加载 ---
    @classmethod
    def from_pretrained_split(
        cls,
        full_state_dict: dict,
        config: Qwen3VLConfig,
        **kwargs,
    ):
        """
        仅提取 `visual.*` 权重加载到视觉前端。
        """
        model = cls(config)
        visual_weights = {
            k.replace("model.visual.", "", 1): v
            for k, v in full_state_dict.items()
            if k.startswith("model.visual.")
        }
        missing, unexpected = model.visual.load_state_dict(visual_weights, strict=False)
        if missing:
            print(f"    Warning: Missing keys in Qwen3VLBottomModel: {missing}")
        if unexpected:
            print(f"    Warning: Unexpected keys in Qwen3VLBottomModel: {unexpected}")
        return model

