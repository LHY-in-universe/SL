"""
Qwen2-VL Top Model（末端 + LM Head）

职责：承载末段 decoder 层 + 最终 norm + lm_head，不包含视觉塔。
"""
from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDecoderLayer, Qwen2RMSNorm

from ...utils.param_mapper import ParamMapper
from ...core import BaseTopModel
from ...registry import ModelRegistry


@ModelRegistry.register("qwen2_vl", "top")
class Qwen2VLTopModel(BaseTopModel):
    def __init__(self, config: Qwen2VLConfig, start_layer: int = 1):
        super().__init__(config, start_layer=start_layer)
        self.layers = nn.ModuleList(
            [Qwen2VLDecoderLayer(config, layer_idx=i) for i in range(self.num_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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
        return r"\.layers\.[0-9]+"

    def get_final_norm(self) -> nn.Module:
        return self.norm

    def get_lm_head(self) -> nn.Module:
        return self.lm_head

    @classmethod
    def from_pretrained_split(
        cls,
        full_state_dict: dict,
        config: Qwen2VLConfig,
        start_layer: int = 1,
    ):
        model = cls(config, start_layer=start_layer)
        top_dict = ParamMapper.filter_and_remap_state_dict(
            full_state_dict,
            model_type="qwen2_vl",
            include_embedding=False,
            include_lm_head=True,
            include_final_norm=True,
            layer_start=start_layer,
            layer_end=None,
            remap_layers=True,
        )
        cleaned = {}
        for k, v in top_dict.items():
            if k.startswith("model."):
                cleaned[k.replace("model.", "", 1)] = v
            else:
                cleaned[k] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"    Warning: Missing keys in Qwen2VLTopModel: {missing}")
        if unexpected:
            print(f"    Warning: Unexpected keys in Qwen2VLTopModel: {unexpected}")
        return model

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple] = None,
        use_cache: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        attention_mask = self.prepare_attention_mask(attention_mask, hidden_states)
        presents = () if use_cache else None
        for i, block in enumerate(self.get_transformer_blocks()):
            layer_past = past_key_values[i] if past_key_values is not None else None
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                layer_past=layer_past,
                use_cache=use_cache,
            )
            hidden_states = outputs[0]
            if use_cache:
                presents = presents + (outputs[1],)

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

