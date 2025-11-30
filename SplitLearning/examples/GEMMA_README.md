# Gemma 模型分割示例

本文档说明如何使用 SplitLearn 库分割 Google Gemma 模型。

## 支持的 Gemma 模型

- **Gemma 2B** (`google/gemma-2b`): 18 层，约 2B 参数
- **Gemma 7B** (`google/gemma-7b`): 28 层，约 7B 参数

## 架构特点

Gemma 模型采用类似 LLaMA/Qwen2 的架构：

- **RoPE (Rotary Position Embeddings)**: 旋转位置编码在各注意力层内部应用
- **RMSNorm**: 用于归一化的 Root Mean Square Layer Normalization
- **SwiGLU**: 激活函数
- **多头注意力机制**: 支持分组查询注意力（GQA）
- **Decoder-only 架构**: 自回归语言模型

## 使用方法

### 基本示例

```python
from splitlearn import ModelFactory
from transformers import AutoTokenizer
import torch

# 配置
model_type = 'gemma'
model_name = 'google/gemma-2b'
split_point_1 = 6   # Bottom: 层 0-5
split_point_2 = 12  # Trunk: 层 6-11, Top: 层 12-17
device = 'cpu'

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 分割模型
bottom, trunk, top = ModelFactory.create_split_models(
    model_type=model_type,
    model_name_or_path=model_name,
    split_point_1=split_point_1,
    split_point_2=split_point_2,
    device=device
)

# 准备输入
input_text = "The future of AI is"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 通过分割模型进行推理
with torch.no_grad():
    hidden_1 = bottom(input_ids)    # Bottom 输出
    hidden_2 = trunk(hidden_1)       # Trunk 输出
    output = top(hidden_2)           # Top 输出 (logits)

# 获取预测
predicted_token = output.logits[0, -1].argmax(dim=-1)
print(tokenizer.decode(predicted_token))
```

### 推荐的分割点

根据模型大小选择合适的分割点：

#### Gemma 2B (18 层)
- **均衡分割**: `split_point_1=6, split_point_2=12`
  - Bottom: 6 层 (客户端)
  - Trunk: 6 层 (服务器)
  - Top: 6 层 (客户端)

- **服务器重载**: `split_point_1=4, split_point_2=14`
  - Bottom: 4 层
  - Trunk: 10 层
  - Top: 4 层

#### Gemma 7B (28 层)
- **均衡分割**: `split_point_1=9, split_point_2=19`
  - Bottom: 9 层
  - Trunk: 10 层
  - Top: 9 层

- **服务器重载**: `split_point_1=6, split_point_2=22`
  - Bottom: 6 层
  - Trunk: 16 层
  - Top: 6 层

## 模型访问

**重要**: Gemma 模型需要在 Hugging Face 上接受许可协议才能访问。

1. 访问 https://huggingface.co/google/gemma-2b
2. 接受许可协议
3. 使用您的 Hugging Face token 进行身份验证：

```bash
huggingface-cli login
```

或在代码中：

```python
from huggingface_hub import login
login(token="your_hf_token")
```

## 内存需求

### Gemma 2B
- **FP32**: 约 8 GB
- **FP16**: 约 4 GB
- **INT8**: 约 2 GB

### Gemma 7B
- **FP32**: 约 28 GB
- **FP16**: 约 14 GB
- **INT8**: 约 7 GB

## 高级功能

### 增量加载（低内存模式）

对于大型 Gemma 模型，可以使用增量加载来减少内存峰值：

```python
bottom, trunk, top = ModelFactory.create_split_models(
    model_type='gemma',
    model_name_or_path='google/gemma-7b',
    split_point_1=9,
    split_point_2=19,
    low_memory=True,      # 启用增量加载
    verbose=True,         # 显示详细信息
    device='cpu'
)
```

### 多设备部署

将不同部分部署到不同设备：

```python
device_map = {
    'bottom': 'cpu',      # Bottom 在 CPU
    'trunk': 'cuda:0',    # Trunk 在 GPU 0
    'top': 'cuda:1'       # Top 在 GPU 1
}

bottom, trunk, top = ModelFactory.create_split_models(
    model_type='gemma',
    model_name_or_path='google/gemma-7b',
    split_point_1=9,
    split_point_2=19,
    device='cpu',
    device_map=device_map
)
```

### 自动保存分割模型

```python
bottom, trunk, top = ModelFactory.create_split_models(
    model_type='gemma',
    model_name_or_path='google/gemma-2b',
    split_point_1=6,
    split_point_2=12,
    device='cpu',
    storage_path='./saved_models/gemma-2b',
    auto_save=True
)
```

## 完整示例

运行 [`gemma_example.py`](gemma_example.py) 以查看完整的工作示例：

```bash
cd SplitLearning/examples
python gemma_example.py
```

## 性能提示

1. **使用 FP16**: 在支持的硬件上使用半精度可以减少内存并提高速度
2. **批处理**: 对多个输入进行批处理以提高吞吐量
3. **缓存**: 使用 KV 缓存进行自回归生成
4. **量化**: 考虑使用 INT8 量化进一步减少内存

## 疑难解答

### 模型访问错误
如果遇到 "Repository not found" 或 "401 Unauthorized" 错误：
- 确保已在 Hugging Face 上接受 Gemma 许可协议
- 使用 `huggingface-cli login` 进行身份验证

### 内存不足
如果遇到 OOM 错误：
- 使用 `low_memory=True` 启用增量加载
- 减少批次大小
- 使用较小的模型变体（2B 而不是 7B）
- 考虑使用量化

### transformers 版本
Gemma 支持需要 transformers >= 4.38.0：

```bash
pip install transformers>=4.38.0
```

## 参考

- Gemma 论文: [Gemma: Open Models Based on Gemini Technology](https://arxiv.org/abs/2403.08295)
- Hugging Face 模型页面: https://huggingface.co/google/gemma-2b
- SplitLearn 文档: [../docs/](../docs/)

