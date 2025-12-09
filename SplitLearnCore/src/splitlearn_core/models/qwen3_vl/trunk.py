"""
Qwen3-VL Trunk Model (中段 Transformer)

职责：承载中间若干 decoder 层（从 split_point_1 到 split_point_2），不含视觉塔/LM head。
实现与 Qwen2-VL 拆分保持一致。
"""
from typing import Optional

import torch
import torch.nn as nn
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextConfig,
    Qwen3VLTextRotaryEmbedding,
)

from ...utils.param_mapper import ParamMapper
from ...core import BaseTrunkModel
from ...registry import ModelRegistry


@ModelRegistry.register("qwen3_vl", "trunk")
class Qwen3VLTrunkModel(BaseTrunkModel):
    def __init__(self, config: Qwen3VLConfig, start_layer: int = 0, end_layer: int = 1):
        super().__init__(config, start_layer, end_layer)
        if isinstance(config.text_config, Qwen3VLTextConfig):
            text_cfg = config.text_config
        elif hasattr(config.text_config, "to_dict"):
            text_cfg = Qwen3VLTextConfig(**config.text_config.to_dict())
        elif isinstance(config.text_config, dict):
            text_cfg = Qwen3VLTextConfig(**config.text_config)
        else:
            text_cfg = Qwen3VLTextConfig()
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(text_cfg, layer_idx=i) for i in range(self.num_layers)]
        )
        # 添加 rotary_emb 用于生成 position_embeddings
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=text_cfg)
        self._fix_attention_implementation()
        self.apply(self._init_weights)

    def get_transformer_blocks(self) -> nn.ModuleList:
        return self.layers

    def prepare_attention_mask(
        self, attention_mask: Optional[torch.Tensor], hidden_states: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        dtype = hidden_states.dtype

        if attention_mask.dim() == 2:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
                diagonal=1,
            )
            expanded_mask = attention_mask[:, None, None, :].to(dtype=dtype)
            expanded_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
            attention_mask = expanded_mask + causal_mask.unsqueeze(0)
        return attention_mask

    def get_layer_name_pattern(self) -> str:
        return r"\.language_model\.layers\.[0-9]+"

    @classmethod
    def from_pretrained_split(
        cls,
        full_state_dict: dict,
        config: Qwen3VLConfig,
        start_layer: int = 0,
        end_layer: int = 1,
    ):
        model = cls(config, start_layer=start_layer, end_layer=end_layer)
        trunk_dict = ParamMapper.filter_and_remap_state_dict(
            full_state_dict,
            model_type="qwen3_vl",
            include_embedding=False,
            include_lm_head=False,
            include_final_norm=False,
            layer_start=start_layer,
            layer_end=end_layer,
            remap_layers=True,
        )
        # 去掉 'model.' 前缀
        cleaned = {k.replace("model.language_model.", "", 1) if k.startswith("model.language_model.") else k.replace("model.", "", 1): v for k, v in trunk_dict.items()}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"    Warning: Missing keys in Qwen3VLTrunkModel: {missing}")
        if unexpected:
            print(f"    Warning: Unexpected keys in Qwen3VLTrunkModel: {unexpected}")
        return model

    def _fix_attention_implementation(self):
        blocks = self.get_transformer_blocks()
        for block in blocks:
            if hasattr(block, "self_attn") and hasattr(block.self_attn, "config"):
                if not hasattr(block.self_attn.config, "_attn_implementation") or \
                   block.self_attn.config._attn_implementation is None:
                    block.self_attn.config._attn_implementation = "eager"

