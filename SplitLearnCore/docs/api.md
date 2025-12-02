# API 参考

## ModelFactory

### `ModelFactory.create_split_models()`

从预训练模型创建拆分模型（bottom, trunk, top）。

```python
def create_split_models(
    model_type: str,
    model_name_or_path: str,
    split_point_1: int,
    split_point_2: int,
    device: str = 'cpu'
) -> Tuple[BaseBottomModel, BaseTrunkModel, BaseTopModel]
```

**参数:**
- `model_type` (str): 模型类型 ('gpt2', 'qwen2' 等)
- `model_name_or_path` (str): HuggingFace 模型名称或本地路径
- `split_point_1` (int): Bottom 模型的结束层（不包含）
- `split_point_2` (int): Trunk 模型的结束层（不包含）
- `device` (str): 加载模型的设备 ('cpu', 'cuda', 'mps')

**返回:**
- (bottom_model, trunk_model, top_model) 的元组

**示例:**
```python
from splitlearn import ModelFactory

bottom, trunk, top = ModelFactory.create_split_models(
    model_type='gpt2',
    model_name_or_path='gpt2',
    split_point_1=2,
    split_point_2=10,
    device='cpu'
)
```

## 基类 (Base Classes)

### `BaseSplitModel`

所有拆分模型的抽象基类。

```python
class BaseSplitModel(nn.Module):
    def __init__(self, config: PretrainedConfig)
```

**方法:**
- `forward()`: 前向传播（必须由子类实现）

### `BaseBottomModel`

Bottom 模型的基类（Embeddings + 前 N 层）。

```python
class BaseBottomModel(BaseSplitModel):
    def __init__(self, config: PretrainedConfig, end_layer: int)
```

**抽象方法:**
- `get_embeddings()`: 返回 token embedding 层
- `apply_position_encoding()`: 应用位置编码
- `get_transformer_blocks()`: 返回 transformer 层列表
- `prepare_attention_mask()`: 准备 attention mask
- `get_layer_name_pattern()`: 返回层名称的正则表达式
- `from_pretrained_split()`: 从预训练加载的类方法

**前向传播:**
```python
def forward(
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None
) -> torch.Tensor
```

### `BaseTrunkModel`

Trunk 模型的基类（中间 M 层）。

```python
class BaseTrunkModel(BaseSplitModel):
    def __init__(self, config: PretrainedConfig, start_layer: int, end_layer: int)
```

**抽象方法:**
- `get_transformer_blocks()`: 返回 transformer 层列表
- `prepare_attention_mask()`: 准备 attention mask
- `get_layer_name_pattern()`: 返回层名称的正则表达式
- `from_pretrained_split()`: 从预训练加载的类方法

**前向传播:**
```python
def forward(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor
```

### `BaseTopModel`

Top 模型的基类（最后 K 层 + LM Head）。

```python
class BaseTopModel(BaseSplitModel):
    def __init__(self, config: PretrainedConfig, start_layer: int)
```

**抽象方法:**
- `get_transformer_blocks()`: 返回 transformer 层列表
- `get_lm_head()`: 返回语言模型 head
- `prepare_attention_mask()`: 准备 attention mask
- `get_layer_name_pattern()`: 返回层名称的正则表达式
- `from_pretrained_split()`: 从预训练加载的类方法

**前向传播:**
```python
def forward(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> CausalLMOutputWithCrossAttentions
```

## ModelRegistry

模型实现的注册表。

### `ModelRegistry.register()`

用于注册模型类的装饰器。

```python
@ModelRegistry.register('model_type', 'model_part')
class YourModel(BaseBottomModel):
    ...
```

**参数:**
- `model_type` (str): 类型标识符 (例如 'gpt2', 'qwen2')
- `model_part` (str): 部分标识符 ('bottom', 'trunk', 'top')

### `ModelRegistry.get_model_class()`

检索已注册的模型类。

```python
model_class = ModelRegistry.get_model_class('gpt2', 'bottom')
```

## ParamMapper

用于过滤和重映射模型参数的工具。

```python
class ParamMapper:
    def __init__(
        self,
        layer_pattern: str,
        start_layer: int,
        end_layer: int
    )
```

**方法:**

### `filter_and_remap_state_dict()`

过滤并重映射拆分模型的 state dict。

```python
def filter_and_remap_state_dict(
    self,
    state_dict: dict,
    include_embeddings: bool = False,
    include_lm_head: bool = False
) -> dict
```

**参数:**
- `state_dict` (dict): 完整模型的 state dict
- `include_embeddings` (bool): 是否包含 embedding 参数
- `include_lm_head` (bool): 是否包含 LM head 参数

**返回:**
- 过滤并重映射后的 state dict

## 模型特定类 (Model-Specific Classes)

### GPT-2 Models

```python
from splitlearn.models.gpt2 import (
    GPT2BottomModel,
    GPT2TrunkModel,
    GPT2TopModel
)
```

### Qwen2 Models

```python
from splitlearn.models.qwen2 import (
    Qwen2BottomModel,
    Qwen2TrunkModel,
    Qwen2TopModel
)
```

所有特定于模型的类都遵循基类接口，可以直接使用或通过 ModelFactory 使用。
