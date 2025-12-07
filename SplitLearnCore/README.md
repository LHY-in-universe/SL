# SplitLearn

A Python library for physically splitting large language models into distributed components.

## Features

- 🔧 **Model Splitting**: Split any transformer model into Bottom, Trunk, and Top parts
- 🎯 **Multi-Architecture**: Built-in support for GPT-2, Gemma, Qwen2, easily extensible
- 📦 **Minimal Dependencies**: Only requires PyTorch and Transformers
- 🚀 **Easy to Use**: Simple API with sensible defaults
- 🔌 **Extensible**: Abstract base classes for adding new architectures

## Installation / 环境

- 已验证环境：Python 3.11.12、torch 2.9.1、transformers 4.57.3、grpcio 1.69.0（如使用 Comm）。
- 推荐方式：源码 + PYTHONPATH 或 pip 开发模式安装。

Framework Python 示例：
```bash
export PYTHONPATH=/Users/lhy/Desktop/Git/SL/SplitLearnCore/src:${PYTHONPATH:-}
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -c "import splitlearn_core; print('import ok')"
```

虚拟环境示例：
```bash
python3 -m venv venv
venv/bin/pip install torch==2.6.0 transformers==4.57.3
venv/bin/pip install -e /Users/lhy/Desktop/Git/SL/SplitLearnCore
```

## Quick Start

```python
from splitlearn_core import ModelFactory
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Split a GPT-2 model
bottom, trunk, top = ModelFactory.create_split_models(
    model_type='gpt2',
    model_name_or_path='gpt2',
    split_point_1=2,   # Bottom: layers 0-1
    split_point_2=10,  # Trunk: layers 2-9, Top: layers 10-11
    device='cpu'
)

# Use the split models for inference
input_text = "Hello world"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Forward pass through split models
hidden_1 = bottom(input_ids)
hidden_2 = trunk(hidden_1)
output = top(hidden_2)

# Get predictions
predicted_token = output.logits.argmax(dim=-1)
```

### Load a full Hugging Face causal LM (one-liner)

```python
from splitlearn_core.quickstart import load_full_model
import torch

model, tokenizer = load_full_model(
    model_name_or_path="gpt2",  # any HF causal LM
    device="cpu",               # auto-detect if None
    low_cpu_mem_usage=True      # reduce peak memory
)

inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model(**inputs)
print(out.logits.shape)
```

## Supported Models

| Model | Model Type | Variants |
|-------|------------|----------|
| GPT-2 | `'gpt2'` | gpt2, gpt2-medium, gpt2-large, gpt2-xl |
| Gemma | `'gemma'` | gemma-2b, gemma-7b |
| Qwen2 | `'qwen2'` | Qwen2-0.5B, Qwen2-1.5B, Qwen2-7B, etc. |

## How It Works

SplitLearn splits a transformer model into three parts:

1. **Bottom Model** (Client): Token embeddings + First N layers
2. **Trunk Model** (Server): Middle M layers
3. **Top Model** (Client): Last K layers + LM head

```
┌─────────────────┐
│  Input Tokens   │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Bottom  │  (Embeddings + Layers 0-N)
    └────┬────┘
         │ hidden_states_1
    ┌────▼────┐
    │  Trunk  │  (Layers N-M)
    └────┬────┘
         │ hidden_states_2
    ┌────▼────┐
    │   Top   │  (Layers M-End + LM Head)
    └────┬────┘
         │
    ┌────▼────┐
    │  Logits │
    └─────────┘
```

## Advanced Usage

### Direct Model Creation

```python
from splitlearn_core.models.gpt2 import GPT2BottomModel
from transformers import GPT2Config, GPT2LMHeadModel

# Load full model
config = GPT2Config.from_pretrained('gpt2')
full_model = GPT2LMHeadModel.from_pretrained('gpt2')
state_dict = full_model.state_dict()

# Create bottom model from full model weights
bottom = GPT2BottomModel.from_pretrained_split(
    full_state_dict=state_dict,
    config=config,
    end_layer=2
)
```

### Extending to New Architectures

```python
from splitlearn_core.core import BaseBottomModel
from splitlearn_core.registry import ModelRegistry

@ModelRegistry.register('my_model', 'bottom')
class MyBottomModel(BaseBottomModel):
    def __init__(self, config, end_layer):
        super().__init__(config, end_layer)
        # Initialize your model layers

    def get_embeddings(self):
        return self.embed_tokens

    def apply_position_encoding(self, inputs_embeds, position_ids):
        # Apply your position encoding
        return inputs_embeds + self.position_embeddings

    def get_transformer_blocks(self):
        return self.layers

    def prepare_attention_mask(self, attention_mask, hidden_states):
        # Prepare attention mask for your architecture
        return attention_mask

    def get_layer_name_pattern(self):
        return r'\.layers\.[0-9]+'

    @classmethod
    def from_pretrained_split(cls, full_state_dict, config, end_layer):
        # Load weights from full model
        pass
```

## Documentation

See [docs/](docs/) for detailed documentation:

- [API Reference](docs/api.md)
- [Extending to New Models](docs/extending.md)

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- NumPy >= 1.24.0

## Development

```bash
# Clone repository
git clone https://github.com/yourusername/splitlearn.git
cd splitlearn

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/

# Lint code
flake8 src/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Minimal smoke test (CPU, tiny 模型)

```bash
export PYTHONPATH=/Users/lhy/Desktop/Git/SL/SplitLearnCore/src:${PYTHONPATH:-}
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 - <<'PY'
import torch
from splitlearn_core.quickstart import load_full_model

model, tok = load_full_model('sshleifer/tiny-gpt2', device='cpu', dtype=torch.float32, low_cpu_mem_usage=True)
inputs = tok("hello", return_tensors="pt")
with torch.inference_mode():
    out = model(**inputs)
print("logits:", out.logits.shape)
PY
```

## 简易 API 概览

- `splitlearn_core.quickstart.load_full_model(model_name_or_path, device="cpu", dtype=None, low_cpu_mem_usage=True)`: 一次性加载完整 HF 因果语言模型，返回 `(model, tokenizer)`。
- `splitlearn_core.ModelFactory.create_split_models(model_type, model_name_or_path, split_point_1, split_point_2, device="cpu")`: 创建 Bottom/Trunk/Top 拆分模型。
- 底层模型类（示例）：`splitlearn_core.models.gpt2.GPT2BottomModel / TrunkModel / TopModel`，用于自定义或细粒度控制。

## Citation

If you use SplitLearn in your research, please cite:

```bibtex
@software{splitlearn2024,
  title = {SplitLearn: A Library for Physically Splitting Large Language Models},
  author = {SplitLearn Contributors},
  year = {2024},
  url = {https://github.com/yourusername/splitlearn}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This library was extracted from the [PhysicalSplitGPT2](https://github.com/yourusername/PhysicalSplitGPT2) project.
