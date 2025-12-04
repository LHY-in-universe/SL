# Anaconda 预加载模型方案详解

## 问题回顾

在 Anaconda 环境中：
```python
from splitlearn_core import ModelFactory  # ❌ 死锁！无法导入
```

但是，我们仍然可以**使用已经创建好的模型文件**！

---

## 方案说明

### 核心思路

**分两步走**：
1. **步骤 1**：在 Framework Python 中创建和保存模型（一次性）
2. **步骤 2**：在 Anaconda 中直接加载保存的模型文件（每次使用）

```
┌─────────────────────────────────────────────────────────────┐
│ 步骤 1: 在 Framework Python 中（只需一次）                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  from splitlearn_core import ModelFactory                   │
│  bottom, trunk, top = ModelFactory.create_split_models(...) │
│                                                             │
│  # 保存到磁盘                                               │
│  bottom.save_split_model("./models/bottom/gpt2_bottom.pt")  │
│  trunk.save_split_model("./models/trunk/gpt2_trunk.pt")    │
│  top.save_split_model("./models/top/gpt2_top.pt")          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
                   模型文件已保存
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 2: 在 Anaconda Python 中（每次使用）                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  # ✅ 不导入 ModelFactory！                                 │
│  import torch                                               │
│  from transformers import GPT2Config                        │
│  from splitlearn_core.models.gpt2 import GPT2TrunkModel     │
│                                                             │
│  # 创建空模型结构                                           │
│  config = GPT2Config.from_pretrained("gpt2")               │
│  trunk = GPT2TrunkModel(config, start_layer=2, end_layer=10)│
│                                                             │
│  # 加载权重                                                 │
│  trunk.load_state_dict(torch.load("./models/.../trunk.pt"))│
│                                                             │
│  # 使用模型                                                 │
│  output = trunk(hidden_states)                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 为什么这样可以？

### 问题根源
```python
# ❌ 这个会死锁（在 Anaconda 中）
from splitlearn_core import ModelFactory
  ↓
触发批量导入 gpt2, gemma, qwen2
  ↓
MKL 线程池初始化冲突
  ↓
死锁
```

### 解决方案
```python
# ✅ 这个不会死锁（在 Anaconda 中）
from splitlearn_core.models.gpt2 import GPT2TrunkModel
  ↓
只导入一个特定的模型类
  ↓
没有批量导入
  ↓
不会死锁！
```

**关键**：
- ❌ 不能用 `ModelFactory`（会触发批量导入）
- ✅ 可以导入单个模型类（如 `GPT2TrunkModel`）
- ✅ 可以加载预先保存的权重

---

## 完整示例

### 步骤 1: 在 Framework Python 中创建模型

我已经为你创建了 `save_split_models.py`，运行：

```bash
# 使用 Framework Python 创建和保存模型
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 save_split_models.py
```

结果：
```
models/
├── bottom/
│   ├── gpt2_2-10_bottom.pt              (204 MB)
│   └── gpt2_2-10_bottom_metadata.json   (2.6 KB)
├── trunk/
│   ├── gpt2_2-10_trunk.pt               (216 MB)
│   └── gpt2_2-10_trunk_metadata.json    (2.6 KB)
└── top/
    ├── gpt2_2-10_top.pt                 (201 MB)
    └── gpt2_2-10_top_metadata.json      (2.6 KB)
```

### 步骤 2: 在 Anaconda 中使用预加载的模型

创建一个新脚本 `use_preloaded_models_anaconda.py`：

```python
#!/usr/bin/env python3
"""
在 Anaconda 环境中使用预加载的模型
不使用 ModelFactory，避免死锁
"""

import os
import sys
import json
from pathlib import Path

# 设置环境变量
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

print("=" * 70)
print("在 Anaconda 中使用预加载的模型")
print("=" * 70)

# 导入必要的库（这些不会导致死锁）
import torch
from transformers import AutoTokenizer, GPT2Config

print("\n✓ 基础库导入成功")

# ✅ 只导入需要的模型类（不会死锁）
sys.path.insert(0, '/Users/lhy/Desktop/Git/SL/SplitLearnCore/src')
from splitlearn_core.models.gpt2 import GPT2BottomModel, GPT2TrunkModel, GPT2TopModel

print("✓ 模型类导入成功")

# 读取元数据
models_dir = Path("./models")
metadata_files = {
    'bottom': models_dir / "bottom" / "gpt2_2-10_bottom_metadata.json",
    'trunk': models_dir / "trunk" / "gpt2_2-10_trunk_metadata.json",
    'top': models_dir / "top" / "gpt2_2-10_top_metadata.json",
}

print("\n[1] 读取模型配置...")
with open(metadata_files['bottom']) as f:
    bottom_meta = json.load(f)
with open(metadata_files['trunk']) as f:
    trunk_meta = json.load(f)
with open(metadata_files['top']) as f:
    top_meta = json.load(f)

print(f"  Bottom: 层 0-{bottom_meta['end_layer']}")
print(f"  Trunk:  层 {trunk_meta['start_layer']}-{trunk_meta['end_layer']}")
print(f"  Top:    层 {top_meta['start_layer']}+")

# 创建模型结构
print("\n[2] 创建模型结构...")
config = GPT2Config.from_pretrained("gpt2")

bottom = GPT2BottomModel(config, end_layer=bottom_meta['end_layer'])
trunk = GPT2TrunkModel(
    config,
    start_layer=trunk_meta['start_layer'],
    end_layer=trunk_meta['end_layer']
)
top = GPT2TopModel(config, start_layer=top_meta['start_layer'])

print("  ✓ 模型结构创建完成")

# 加载权重
print("\n[3] 加载预训练权重...")
bottom.load_state_dict(torch.load(
    models_dir / "bottom" / "gpt2_2-10_bottom.pt",
    map_location='cpu',
    weights_only=True
))
trunk.load_state_dict(torch.load(
    models_dir / "trunk" / "gpt2_2-10_trunk.pt",
    map_location='cpu',
    weights_only=True
))
top.load_state_dict(torch.load(
    models_dir / "top" / "gpt2_2-10_top.pt",
    map_location='cpu',
    weights_only=True
))

# 设置为评估模式
bottom.eval()
trunk.eval()
top.eval()

print("  ✓ 权重加载完成")

# 测试推理
print("\n[4] 测试推理...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "The future of AI is"
input_ids = tokenizer.encode(text, return_tensors="pt")

with torch.no_grad():
    h1 = bottom(input_ids)
    h2 = trunk(h1)
    output = top(h2)

next_token_id = output.logits[0, -1].argmax().item()
next_token = tokenizer.decode([next_token_id])

print(f"  输入: '{text}'")
print(f"  预测的下一个词: '{next_token}'")

# 文本生成
print("\n[5] 生成文本...")
generated_ids = input_ids.clone()
for i in range(10):
    with torch.no_grad():
        h1 = bottom(generated_ids)
        h2 = trunk(h1)
        output = top(h2)

    next_id = output.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    generated_ids = torch.cat([generated_ids, next_id], dim=1)

generated_text = tokenizer.decode(generated_ids[0])
print(f"  生成: '{generated_text}'")

print("\n" + "=" * 70)
print("✅ 成功！在 Anaconda 中使用预加载的模型")
print("=" * 70)
print("\n说明:")
print("  1. 不使用 ModelFactory（避免死锁）")
print("  2. 只导入单个模型类（安全）")
print("  3. 加载预先保存的权重")
print("  4. 功能完全正常！")
```

---

## 使用流程

### 一次性设置（使用 Framework Python）

```bash
# 创建并保存模型（只需做一次）
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 save_split_models.py
```

### 日常使用（在 Anaconda 中）

```bash
# 之后每次使用，可以在 Anaconda 中运行
python use_preloaded_models_anaconda.py
# ✅ 不会死锁！
```

---

## 限制和注意事项

### ✅ 可以做的

| 功能 | 支持情况 |
|------|---------|
| 加载预创建的模型 | ✅ 完全支持 |
| 模型推理 | ✅ 完全支持 |
| 文本生成 | ✅ 完全支持 |
| 保存检查点 | ✅ 完全支持 |

### ❌ 不能做的

| 功能 | 支持情况 | 原因 |
|------|---------|------|
| 使用 `ModelFactory` | ❌ 不支持 | 会导致死锁 |
| 动态改变拆分点 | ❌ 不支持 | 需要 ModelFactory |
| 从 HuggingFace 加载新模型 | ❌ 不支持 | 需要 ModelFactory |

### 如何改变拆分点？

如果需要不同的拆分点：

```bash
# 1. 在 Framework Python 中创建新的拆分
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 << 'EOF'
from splitlearn_core import ModelFactory

# 新的拆分点: [4, 8]
bottom, trunk, top = ModelFactory.create_split_models(
    model_type='gpt2',
    model_name_or_path='gpt2',
    split_point_1=4,    # ← 改变
    split_point_2=8,    # ← 改变
    device='cpu'
)

# 保存
bottom.save_split_model("./models/bottom/gpt2_4-8_bottom.pt")
trunk.save_split_model("./models/trunk/gpt2_4-8_trunk.pt")
top.save_split_model("./models/top/gpt2_4-8_top.pt")
EOF

# 2. 在 Anaconda 中加载新的模型
python use_preloaded_models_anaconda.py  # 修改路径指向 gpt2_4-8_*.pt
```

---

## 对比表

| 特性 | Framework Python | Anaconda + 预加载 |
|------|------------------|-------------------|
| **使用 ModelFactory** | ✅ 是 | ❌ 否 |
| **动态创建模型** | ✅ 是 | ❌ 否 |
| **加载预创建模型** | ✅ 是 | ✅ 是 |
| **模型推理** | ✅ 是 | ✅ 是 |
| **设置难度** | 简单 | 需要两步 |
| **灵活性** | 高 | 低 |
| **适用场景** | 开发和生产 | 仅生产 |

---

## 总结

### "Anaconda + 预加载模型"的意思

**意思是**：
1. 不能在 Anaconda 中使用 `ModelFactory` 创建新模型（会死锁）
2. 但可以在 Anaconda 中**加载和使用**已经创建好的模型文件
3. 需要先在 Framework Python 中创建模型，然后在 Anaconda 中使用

**适合的场景**：
- ✅ 你已经确定了拆分点，不需要频繁改变
- ✅ 你只需要运行推理，不需要创建新模型
- ✅ 你必须使用 Anaconda 环境（有其他依赖）

**不适合的场景**：
- ❌ 需要频繁试验不同的拆分点
- ❌ 需要加载多种不同的模型
- ❌ 开发阶段（推荐直接用 Framework Python）

---

## 推荐

如果你**没有必须使用 Anaconda 的理由**，强烈建议：

✅ **直接使用 Framework Python**
- 完全没有限制
- 更简单
- 更灵活

如果你**必须使用 Anaconda**（比如有其他依赖）：

⚠️ **使用预加载模型方案**
- 在 Framework Python 中创建模型（一次性）
- 在 Anaconda 中加载和使用

这就是"Anaconda + 预加载模型 | ⚠️ 部分支持 | 不能用 ModelFactory"的完整含义！
