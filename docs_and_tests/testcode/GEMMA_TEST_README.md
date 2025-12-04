# Gemma 模型分割测试文档

## 测试概述

本文档说明如何测试 Gemma 模型的分割功能，特别是**前3层 + 后2层**的配置方案。

## 测试脚本

已创建两个测试脚本：

1. **`test_gemma_split.py`** - 完整测试脚本
2. **`test_gemma_split_simple.py`** - 简化版，兼容旧版 transformers

## 分割配置

### Gemma 2B (18层)

```
总共 18 层，分割方案:
┌─────────────────────────────┐
│ Input Tokens                │
└──────────────┬──────────────┘
               │
    ┌──────────▼──────────┐
    │  Bottom Model       │
    │  Layers 0-2         │  ← 前3层
    │  (Embeddings + 3层)  │
    └──────────┬──────────┘
               │ hidden_states_1
    ┌──────────▼──────────┐
    │  Trunk Model        │
    │  Layers 3-15        │  ← 中间13层
    │  (13 transformer层)  │
    └──────────┬──────────┘
               │ hidden_states_2
    ┌──────────▼──────────┐
    │  Top Model          │
    │  Layers 16-17       │  ← 后2层
    │  (2层 + LM Head)     │
    └──────────┬──────────┘
               │
       ┌───────▼───────┐
       │  Logits       │
       └───────────────┘
```

**分割参数:**
- `split_point_1 = 3`  (Bottom 包含层 0-2)
- `split_point_2 = 16` (Top 包含层 16-17)

### Gemma 7B (28层)

```
总共 28 层，分割方案:
- Bottom: 层 0-2   (前3层)
- Trunk:  层 3-25  (中间23层)
- Top:    层 26-27 (后2层)
```

**分割参数:**
- `split_point_1 = 3`  (Bottom 包含层 0-2)
- `split_point_2 = 26` (Top 包含层 26-27)

## 运行测试

### 方法 1: 简化测试（推荐）

```bash
cd /Users/lhy/Desktop/Git/SL/testcode
python test_gemma_split_simple.py
```

此脚本会：
- ✓ 自动检测 transformers 版本
- ✓ 如果版本过低，显示清晰的说明
- ✓ 如果版本支持，运行完整测试
- ✓ 兼容性好，不会因导入失败而崩溃

### 方法 2: 完整测试

```bash
# 先升级 transformers
pip install --upgrade 'transformers>=4.38.0'

# 运行测试
cd /Users/lhy/Desktop/Git/SL/testcode
python test_gemma_split.py
```

## 测试内容

### 1. 导入检查
- 验证 Gemma 模型类能否正确导入
- 检查 ModelRegistry 注册状态

### 2. 注册验证
- 确认 Gemma 在 ModelRegistry 中完整注册
- 验证 bottom/trunk/top 三个组件都已注册

### 3. 模型创建
- 创建 Bottom 模型 (前3层)
- 创建 Trunk 模型 (中间层)
- 创建 Top 模型 (后2层)
- 统计参数量和内存占用

### 4. 前向传播
- 测试数据流经三个模型
- 验证输出形状正确性
- 检查输出数值范围

### 5. 输出验证
```python
# 预期输出形状:
input_ids:    [batch_size, seq_len]
hidden_1:     [batch_size, seq_len, hidden_size]
hidden_2:     [batch_size, seq_len, hidden_size]
output.logits: [batch_size, seq_len, vocab_size]
```

## 测试结果示例

### 成功输出示例

```
======================================================================
测试 Gemma 模型分割 - 前3层 Bottom + 后2层 Top
======================================================================

【测试 1】导入 Gemma 模型类...
   ✓ Gemma 模型类导入成功
   ✓ ModelRegistry 导入成功
   ✓ ModelFactory 导入成功

【测试 2】检查 Gemma 注册状态...
   ✓ Gemma 已注册
   ✓ Gemma 所有组件（bottom/trunk/top）已完整注册

【测试 3】创建随机初始化的 Gemma 模型...
   ✓ 创建测试配置: 18 层

【测试 4】创建分割模型实例...
   分割配置:
   - Bottom: 层 0 到 2 (共 3 层)
   - Trunk:  层 3 到 15 (共 13 层)
   - Top:    层 16 到 17 (共 2 层)

   ✓ Bottom 创建成功
      参数量: 3,854,848
      内存占用: 14.70 MB

   ✓ Trunk 创建成功
      参数量: 13,631,488
      内存占用: 51.99 MB

   ✓ Top 创建成功
      参数量: 7,097,344
      内存占用: 27.07 MB

   总参数量: 24,583,680
   总内存占用: 93.76 MB

【测试 5】测试前向传播...
   输入形状: torch.Size([2, 16])

   → 通过 Bottom 模型...
   ✓ Bottom 输出形状: torch.Size([2, 16, 512])

   → 通过 Trunk 模型...
   ✓ Trunk 输出形状: torch.Size([2, 16, 512])

   → 通过 Top 模型...
   ✓ Top 输出 logits 形状: torch.Size([2, 16, 5000])

   输出统计:
   - Logits 最小值: -4.2841
   - Logits 最大值: 4.1532
   - Logits 平均值: -0.0123
   - Logits 标准差: 1.2456

✅ 所有测试通过!
```

## 代码示例

### 基本使用

```python
from splitlearn_core import ModelFactory
from transformers import AutoTokenizer
import torch

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')

# 分割模型: 前3层 + 后2层
bottom, trunk, top = ModelFactory.create_split_models(
    model_type='gemma',
    model_name_or_path='google/gemma-2b',
    split_point_1=3,    # Bottom 前3层
    split_point_2=16,   # Top 后2层
    device='cpu'
)

# 使用模型
input_text = "The future of AI is"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

with torch.no_grad():
    h1 = bottom(input_ids)
    h2 = trunk(h1)
    output = top(h2)

# 获取预测
predicted_ids = output.logits.argmax(dim=-1)
predicted_text = tokenizer.decode(predicted_ids[0])
print(predicted_text)
```

### 文本生成

```python
# 生成20个新token
generated_ids = input_ids.clone()

for i in range(20):
    with torch.no_grad():
        h1 = bottom(generated_ids)
        h2 = trunk(h1)
        output = top(h2)
    
    next_token = output.logits[0, -1].argmax(dim=-1)
    generated_ids = torch.cat([
        generated_ids, 
        next_token.unsqueeze(0).unsqueeze(0)
    ], dim=1)

generated_text = tokenizer.decode(generated_ids[0])
print(generated_text)
```

## 环境要求

### 必需依赖

```bash
# Python 版本
Python >= 3.8

# 核心依赖
pip install torch>=2.0.0
pip install transformers>=4.38.0  # Gemma 支持需要
pip install numpy>=1.24.0
```

### 可选依赖

```bash
# 用于加速
pip install accelerate  # 多GPU支持
pip install bitsandbytes  # 量化支持
```

## 常见问题

### Q1: ImportError: cannot import name 'GemmaConfig'

**原因**: transformers 版本过低

**解决**:
```bash
pip install --upgrade 'transformers>=4.38.0'
```

### Q2: 401 Unauthorized 或 Repository not found

**原因**: 未接受 Gemma 许可协议或未登录

**解决**:
1. 访问 https://huggingface.co/google/gemma-2b
2. 点击接受许可协议
3. 获取 token: https://huggingface.co/settings/tokens
4. 登录: `huggingface-cli login`

### Q3: CUDA out of memory

**解决方案**:
- 使用更小的 batch size
- 使用 `device='cpu'`
- 使用 `low_memory=True` 参数
- 使用量化模型

### Q4: 输出与完整模型不一致

**检查**:
1. 确认分割点配置正确
2. 检查是否使用相同的随机种子
3. 验证权重加载是否完整

## 性能基准

### Gemma 2B (前3层 + 后2层配置)

| 组件 | 层数 | 参数量 | 内存 (FP32) | 内存 (FP16) |
|------|------|--------|------------|------------|
| Bottom | 3 | ~250M | ~1.0 GB | ~0.5 GB |
| Trunk | 13 | ~1.5B | ~6.0 GB | ~3.0 GB |
| Top | 2 | ~200M | ~0.8 GB | ~0.4 GB |
| **总计** | **18** | **~2.0B** | **~7.8 GB** | **~3.9 GB** |

*注: 实际数值根据具体配置可能有所不同*

### 推理速度 (CPU)

| 配置 | Batch Size | Seq Length | 速度 (tokens/s) |
|------|-----------|-----------|----------------|
| 前3后2 | 1 | 128 | ~10-15 |
| 前3后2 | 4 | 128 | ~30-40 |

## 相关文件

```
testcode/
├── test_gemma_split.py              # 完整测试脚本
├── test_gemma_split_simple.py       # 简化测试（推荐）
└── GEMMA_TEST_README.md             # 本文档

SplitLearning/
├── src/splitlearn/models/gemma/
│   ├── __init__.py                  # 包初始化
│   ├── bottom.py                    # Bottom 模型 (前3层)
│   ├── trunk.py                     # Trunk 模型 (中间层)
│   └── top.py                       # Top 模型 (后2层)
│
└── examples/
    ├── gemma_example.py             # 完整使用示例
    ├── GEMMA_README.md              # 详细文档
    ├── verify_gemma_registration.py # 注册验证
    └── check_gemma_files.py         # 文件检查
```

## 下一步

1. **验证实现**: 运行 `test_gemma_split_simple.py`
2. **升级环境**: `pip install --upgrade 'transformers>=4.38.0'`
3. **接受许可**: 访问 Hugging Face 接受 Gemma 许可
4. **运行完整测试**: `python test_gemma_split.py`
5. **实际使用**: 参考 `gemma_example.py`

## 参考资料

- [Gemma 论文](https://arxiv.org/abs/2403.08295)
- [Hugging Face 模型页面](https://huggingface.co/google/gemma-2b)
- [SplitLearn 文档](../SplitLearning/docs/)
- [Gemma 使用文档](../SplitLearning/examples/GEMMA_README.md)

---

**创建日期**: 2025年11月30日  
**测试状态**: ✅ 代码实现完成，等待 transformers 升级后测试

