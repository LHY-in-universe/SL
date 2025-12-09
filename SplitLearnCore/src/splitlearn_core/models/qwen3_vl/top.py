"""
Qwen3-VL Top Model（末端 + LM Head）

职责：承载末段 decoder 层 + 最终 norm + lm_head，不包含视觉塔。
实现与 Qwen2-VL 拆分保持一致。
"""
from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextConfig,
    Qwen3VLTextRotaryEmbedding,
)

from ...utils.param_mapper import ParamMapper
from ...core import BaseTopModel
from ...registry import ModelRegistry


@ModelRegistry.register("qwen3_vl", "top")
class Qwen3VLTopModel(BaseTopModel):
    def __init__(self, config: Qwen3VLConfig, start_layer: int = 1):
        super().__init__(config, start_layer=start_layer)
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
        self.norm = Qwen3VLTextRMSNorm(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps)
        # 添加 rotary_emb 用于生成 position_embeddings
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=text_cfg)
        # vocab_size 在 text_config 中
        vocab_size = getattr(text_cfg, 'vocab_size', None) or getattr(config, 'vocab_size', None)
        if vocab_size is None:
            raise ValueError("Cannot find vocab_size in config or text_config")
        self.lm_head = nn.Linear(text_cfg.hidden_size, vocab_size, bias=False)
        self._fix_attention_implementation()
        # 保存 text_config 用于 _init_weights
        self._text_config = text_cfg
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        重写 _init_weights，从 text_config 获取 initializer_range
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            initializer_range = getattr(self._text_config, 'initializer_range', 0.02)
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

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

    def get_final_norm(self) -> nn.Module:
        return self.norm

    def get_lm_head(self) -> nn.Module:
        return self.lm_head

    @classmethod
    def from_pretrained_split(
        cls,
        full_state_dict: dict,
        config: Qwen3VLConfig,
        start_layer: int = 1,
    ):
        model = cls(config, start_layer=start_layer)
        top_dict = ParamMapper.filter_and_remap_state_dict(
            full_state_dict,
            model_type="qwen3_vl",
            include_embedding=False,
            include_lm_head=True,
            include_final_norm=True,
            layer_start=start_layer,
            layer_end=None,
            remap_layers=True,
        )
        cleaned = {}
        for k, v in top_dict.items():
            if k.startswith("model.language_model."):
                cleaned[k.replace("model.language_model.", "", 1)] = v
            elif k.startswith("model."):
                cleaned[k.replace("model.", "", 1)] = v
            else:
                cleaned[k] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"    Warning: Missing keys in Qwen3VLTopModel: {missing}")
        if unexpected:
            print(f"    Warning: Unexpected keys in Qwen3VLTopModel: {unexpected}")
        return model

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple] = None,
        use_cache: Optional[bool] = None,
        position_embeddings: Optional[tuple] = None,
    ) -> CausalLMOutputWithPast:
        attention_mask = self.prepare_attention_mask(attention_mask, hidden_states)
        
        # 生成 position_embeddings（如果未提供）
        if position_embeddings is None:
            batch_size, seq_len = hidden_states.shape[:2]
            device = hidden_states.device
            
            # 创建 position_ids（Qwen3VL 需要 3D position_ids: [3, batch_size, seq_len]）
            # 对于纯文本，使用简单的递增序列
            position_ids = torch.arange(0, seq_len, device=device, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len]
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)  # [3, batch_size, seq_len]
            
            # 使用 rotary_emb 生成 position_embeddings
            # rotary_emb.forward(x, position_ids) 返回 position_embeddings
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        presents = () if use_cache else None
        for i, block in enumerate(self.get_transformer_blocks()):
            layer_past = past_key_values[i] if past_key_values is not None else None
            outputs = block(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=layer_past,
                use_cache=use_cache,
            )
            # Qwen3VLTextDecoderLayer 返回单个张量或元组
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
                if use_cache and len(outputs) > 1:
                    presents = presents + (outputs[1],)
            else:
                hidden_states = outputs

        hidden_states = self.get_final_norm()(hidden_states)
        logits = self.get_lm_head()(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents if use_cache else None,
        )

    def _fix_attention_implementation(self):
        blocks = self.get_transformer_blocks()
        for block in blocks:
            if hasattr(block, "self_attn") and hasattr(block.self_attn, "config"):
                if not hasattr(block.self_attn.config, "_attn_implementation") or \
                   block.self_attn.config._attn_implementation is None:
                    block.self_attn.config._attn_implementation = "eager"

