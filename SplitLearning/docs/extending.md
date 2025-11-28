# 扩展 SplitLearn 以支持新模型

本指南解释了如何向 SplitLearn 添加对新模型架构的支持。

## 概览

要添加新的模型架构，你需要：

1. 创建三个模型类（Bottom, Trunk, Top）
2. 实现所需的抽象方法
3. 使用 ModelRegistry 注册模型
4. 实现从预训练模型加载参数的功能

## 分步指南

### 1. 创建模型文件

在 `src/splitlearn/models/` 下为你的模型创建一个新目录：

```
src/splitlearn/models/
  └── your_model/
      ├── __init__.py
      ├── bottom.py
      ├── trunk.py
      └── top.py
```

### 2. 实现 Bottom 模型

```python
# src/splitlearn/models/your_model/bottom.py

from typing import Optional
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from splitlearn.core import BaseBottomModel
from splitlearn.registry import ModelRegistry
from splitlearn.utils import ParamMapper


@ModelRegistry.register('your_model', 'bottom')
class YourBottomModel(BaseBottomModel):
    """YourModel 架构的 Bottom 模型。"""

    def __init__(self, config: PretrainedConfig, end_layer: int):
        super().__init__(config, end_layer)

        # 初始化 embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )

        # 初始化 transformer 层
        self.layers = nn.ModuleList([
            YourTransformerLayer(config) for _ in range(end_layer)
        ])

        # 其他组件 (layer norm, dropout 等)
        self.norm = nn.LayerNorm(config.hidden_size)

    def get_embeddings(self):
        """返回 token embedding 层。"""
        return self.embed_tokens

    def apply_position_encoding(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor
    ):
        """对 embeddings 应用位置编码。"""
        position_embeds = self.position_embeddings(position_ids)
        return inputs_embeds + position_embeds

    def get_transformer_blocks(self):
        """返回 transformer 层列表。"""
        return self.layers

    def prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        hidden_states: torch.Tensor
    ):
        """为你的模型格式准备 attention mask。"""
        if attention_mask is None:
            return None

        # 实现你的 attention mask 准备逻辑
        # 这取决于你的模型架构
        # 示例: 扩展 mask 到 [batch, 1, seq_len, seq_len]
        return attention_mask

    def get_layer_name_pattern(self):
        """返回 state dict 中层名称的正则表达式模式。"""
        # 常见模式:
        # GPT-2: r'\\.h\\.[0-9]+'
        # Qwen2/LLaMA: r'\\.layers\\.[0-9]+'
        # BERT: r'\\.layer\\.[0-9]+'
        return r'\\.layers\\.[0-9]+'

    @classmethod
    def from_pretrained_split(
        cls,
        full_state_dict: dict,
        config: PretrainedConfig,
        end_layer: int
    ):
        """从完整模型的 state dict 创建 bottom 模型。"""
        model = cls(config, end_layer)

        param_mapper = ParamMapper(
            layer_pattern=model.get_layer_name_pattern(),
            start_layer=0,
            end_layer=end_layer
        )

        filtered_state_dict = param_mapper.filter_and_remap_state_dict(
            full_state_dict,
            include_embeddings=True,
            include_lm_head=False
        )

        model.load_state_dict(filtered_state_dict, strict=False)
        return model
```

### 3. 实现 Trunk 模型

```python
# src/splitlearn/models/your_model/trunk.py

from splitlearn.core import BaseTrunkModel
from splitlearn.registry import ModelRegistry

@ModelRegistry.register('your_model', 'trunk')
class YourTrunkModel(BaseTrunkModel):
    """中间层的 Trunk 模型。"""

    def __init__(self, config: PretrainedConfig, start_layer: int, end_layer: int):
        super().__init__(config, start_layer, end_layer)

        num_layers = end_layer - start_layer
        self.layers = nn.ModuleList([
            YourTransformerLayer(config) for _ in range(num_layers)
        ])

    def get_transformer_blocks(self):
        return self.layers

    def prepare_attention_mask(self, attention_mask, hidden_states):
        # 与 bottom 模型实现相同
        return attention_mask

    def get_layer_name_pattern(self):
        return r'\\.layers\\.[0-9]+'

    @classmethod
    def from_pretrained_split(cls, full_state_dict, config, start_layer, end_layer):
        model = cls(config, start_layer, end_layer)

        param_mapper = ParamMapper(
            layer_pattern=model.get_layer_name_pattern(),
            start_layer=start_layer,
            end_layer=end_layer
        )

        filtered_state_dict = param_mapper.filter_and_remap_state_dict(
            full_state_dict,
            include_embeddings=False,
            include_lm_head=False
        )

        model.load_state_dict(filtered_state_dict, strict=False)
        return model
```

### 4. 实现 Top 模型

```python
# src/splitlearn/models/your_model/top.py

from splitlearn.core import BaseTopModel
from splitlearn.registry import ModelRegistry

@ModelRegistry.register('your_model', 'top')
class YourTopModel(BaseTopModel):
    """最后几层 + LM Head 的 Top 模型。"""

    def __init__(self, config: PretrainedConfig, start_layer: int):
        super().__init__(config, start_layer)

        num_layers = config.num_hidden_layers - start_layer
        self.layers = nn.ModuleList([
            YourTransformerLayer(config) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_transformer_blocks(self):
        return self.layers

    def get_lm_head(self):
        return self.lm_head

    def prepare_attention_mask(self, attention_mask, hidden_states):
        return attention_mask

    def get_layer_name_pattern(self):
        return r'\\.layers\\.[0-9]+'

    @classmethod
    def from_pretrained_split(cls, full_state_dict, config, start_layer):
        model = cls(config, start_layer)

        param_mapper = ParamMapper(
            layer_pattern=model.get_layer_name_pattern(),
            start_layer=start_layer,
            end_layer=config.num_hidden_layers
        )

        filtered_state_dict = param_mapper.filter_and_remap_state_dict(
            full_state_dict,
            include_embeddings=False,
            include_lm_head=True
        )

        model.load_state_dict(filtered_state_dict, strict=False)
        return model
```

### 5. 创建包 Init 文件

```python
# src/splitlearn/models/your_model/__init__.py

from .bottom import YourBottomModel
from .trunk import YourTrunkModel
from .top import YourTopModel

__all__ = ['YourBottomModel', 'YourTrunkModel', 'YourTopModel']
```

### 6. 在主包中注册

将你的模型添加到 `src/splitlearn/models/__init__.py`:

```python
from . import your_model  # 这会触发注册
```

## 关键注意事项

### 层名称模式

不同的模型使用不同的层命名约定：

- **GPT-2**: `transformer.h.0`, `transformer.h.1`, ... → 模式: `r'\\.h\\.[0-9]+'`
- **Qwen2/LLaMA**: `model.layers.0`, `model.layers.1`, ... → 模式: `r'\\.layers\\.[0-9]+'`
- **BERT**: `encoder.layer.0`, `encoder.layer.1`, ... → 模式: `r'\\.layer\\.[0-9]+'`

ParamMapper 使用该模式来识别和重映射层索引。

### Attention Masks

不同的架构需要不同的 attention mask 格式：

- **Causal (GPT-2, GPT-J)**: 下三角掩码, 形状 `[batch, 1, seq_len, seq_len]`
- **Bidirectional (BERT)**: 简单掩码, 形状 `[batch, seq_len]`
- **Prefix (T5)**: 编码器-解码器的自定义掩码

根据你的模型要求实现 `prepare_attention_mask()`。

### 位置编码 (Position Encoding)

不同的位置编码方法：

- **Learned Embeddings** (GPT-2): `nn.Embedding(max_positions, hidden_size)`
- **Sinusoidal** (Original Transformer): 固定的 sin/cos 函数
- **RoPE** (LLaMA, GPT-NeoX): 旋转位置编码 (Rotary position embeddings)
- **ALiBi** (BLOOM): 注意力偏差 (Attention bias)

实现 `apply_position_encoding()` 以匹配你的模型方法。

## 测试你的实现

```python
from splitlearn import ModelFactory

# 使用你的模型进行测试
bottom, trunk, top = ModelFactory.create_split_models(
    model_type='your_model',
    model_name_or_path='path/to/pretrained',
    split_point_1=4,
    split_point_2=8,
    device='cpu'
)

# 验证输出是否与完整模型匹配
from transformers import AutoModelForCausalLM, AutoTokenizer

full_model = AutoModelForCausalLM.from_pretrained('path/to/pretrained')
tokenizer = AutoTokenizer.from_pretrained('path/to/pretrained')

input_ids = tokenizer.encode("Test input", return_tensors="pt")

# 拆分模型
with torch.no_grad():
    h1 = bottom(input_ids)
    h2 = trunk(h1)
    split_output = top(h2)

# 完整模型
with torch.no_grad():
    full_output = full_model(input_ids)

# 比较
diff = torch.abs(split_output.logits - full_output.logits).max()
print(f"最大差异: {diff:.6f}")
assert diff < 1e-4, "输出应该匹配!"
```

## 示例

请参阅现有实现以供参考：
- `src/splitlearn/models/gpt2/` - GPT-2 实现
- `src/splitlearn/models/qwen2/` - Qwen2 实现

## 常见陷阱

1. **错误的层模式**: 确保你的正则表达式匹配你的模型 state dict 的键
2. **缺少组件**: 不要忘记 layer norms, dropout 层等
3. **错误的 attention mask 格式**: 检查你的模型预期的 attention mask 形状
4. **位置编码顺序**: 有些模型在层之前应用位置编码，有些在之后
5. **参数共享**: 处理绑定权重（例如，embedding-LM head 权重绑定）

## 获取帮助

如果遇到问题：
1. 检查现有实现 (GPT-2, Qwen2)
2. 打印完整模型的 state dict 键: `print(full_model.state_dict().keys())`
3. 在 `load_state_dict()` 中使用 `strict=False` 并检查警告
4. 比较每一步的输出形状
