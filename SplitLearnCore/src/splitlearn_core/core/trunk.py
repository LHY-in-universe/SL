"""
Base abstract class for Trunk models

Trunk models handle: Middle M transformer layers (server-side)
"""
from typing import Optional
import torch
import torch.nn as nn

from .base import BaseSplitModel


class BaseTrunkModel(BaseSplitModel):
    """
    Abstract base class for Trunk split models

    Trunk models are responsible for:
    1. Receiving intermediate hidden states from Bottom
    2. Processing through middle M transformer layers
    3. Outputting hidden states for Top

    Trunk models are typically deployed on the server side.
    """

    def __init__(self, config, start_layer: int, end_layer: int):
        """
        Initialize Trunk model

        Args:
            config: Model configuration
            start_layer: Starting layer index in original model
            end_layer: Ending layer index (exclusive)
        """
        super().__init__(config, start_layer, end_layer)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple] = None,
        use_cache: Optional[bool] = None,
        position_embeddings: Optional[tuple] = None,
    ):
        """
        Forward pass through Trunk model

        Args:
            hidden_states: Input from Bottom [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            past_key_values: Tuple of past key value states
            use_cache: Whether to return the key/value attentions
            position_embeddings: Optional tuple of (cos, sin) for RoPE

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]: 
                - Output hidden states [batch_size, seq_len, hidden_size]
                - (Optional) Present key/value states if use_cache=True
        """
        # Prepare attention mask
        attention_mask = self.prepare_attention_mask(attention_mask, hidden_states)

        # 生成 position_embeddings（如果未提供且需要）
        if position_embeddings is None and len(self.get_transformer_blocks()) > 0:
            # 检查第一个 block 是否需要 position_embeddings
            block = self.get_transformer_blocks()[0]
            if hasattr(block, 'forward'):
                import inspect
                sig = inspect.signature(block.forward)
                if 'position_embeddings' in sig.parameters:
                    # 需要生成 position_embeddings
                    batch_size, seq_len = hidden_states.shape[:2]
                    device = hidden_states.device
                    
                    # 创建 position_ids（Qwen3VL 需要 3D position_ids: [3, batch_size, seq_len]）
                    position_ids = torch.arange(0, seq_len, device=device, dtype=torch.long)
                    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len]
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)  # [3, batch_size, seq_len]
                    
                    # 使用 rotary_emb 生成 position_embeddings（如果存在）
                    if hasattr(self, 'rotary_emb'):
                        position_embeddings = self.rotary_emb(hidden_states, position_ids)
                    else:
                        # 如果没有 rotary_emb，尝试从 block 获取
                        # 这种情况不应该发生，因为我们在 __init__ 中已经添加了 rotary_emb
                        raise RuntimeError("rotary_emb not found in trunk model")

        # Pass through transformer blocks
        presents = () if use_cache else None
        
        for i, block in enumerate(self.get_transformer_blocks()):
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            # 检查 block 是否需要 position_embeddings
            import inspect
            sig = inspect.signature(block.forward)
            if 'position_embeddings' in sig.parameters:
                outputs = block(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=layer_past,
                    use_cache=use_cache,
                )
            else:
                # 旧版本接口，使用 layer_past
                outputs = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    layer_past=layer_past,
                    use_cache=use_cache,
                )
            
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
                if use_cache and len(outputs) > 1:
                    presents = presents + (outputs[1],)
            else:
                hidden_states = outputs

        if use_cache:
            return hidden_states, presents
        return hidden_states
