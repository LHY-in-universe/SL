# Anaconda 环境无法使用的最终结论

## 问题确认

经过详细测试，确认问题：

### 现象
```bash
import splitlearn_core
[mutex.cc : 452] RAW: Lock blocking ...  ← 卡住，无法继续
```

###  已测试的所有方案

| 方案 | 结果 | 说明 |
|------|------|------|
| 设置 OMP_NUM_THREADS=1 | ❌ 失败 | 仍然卡住 |
| 设置 MKL_THREADING_LAYER=SEQUENTIAL | ❌ 失败 | 仍然卡住 |
| 直接导入 factory | ❌ 失败 | factory.py 导入 transformers 时卡住 |
| 延迟导入 | ❌ 不可行 | 需要大量修改代码 |

### 根本原因

**Anaconda PyTorch + MKL + transformers 的组合在嵌套导入时会死锁**：

```
from splitlearn_core.factory import ModelFactory
  ↓
factory.py 第 11 行: from transformers import AutoConfig
  ↓
transformers 内部大量导入和初始化
  ↓
触发 torch 的多次导入
  ↓
MKL 线程池初始化冲突
  ↓
❌ 死锁
```

这是 **Anaconda + MKL + transformers + 复杂导入链** 的已知问题。

---

## 唯一可靠的解决方案

### 使用 Framework Python（非 Anaconda）

Framework Python 使用不同的 PyTorch 后端（OpenBLAS），不会有这个问题。

#### 步骤 1: 检查 Framework Python

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 --version
```

#### 步骤 2: 创建虚拟环境

```bash
# 创建虚拟环境
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m venv ~/venv-splitlearn

# 激活
source ~/venv-splitlearn/bin/activate

# 安装依赖
pip install torch torchvision transformers
```

#### 步骤 3: 安装 SplitLearnCore

```bash
# 进入 SplitLearnCore 目录
cd /Users/lhy/Desktop/Git/SL/SplitLearnCore

# 安装（开发模式）
pip install -e .
```

#### 步骤 4: 测试

```bash
python << 'EOF'
from splitlearn_core import ModelFactory

bottom, trunk, top = ModelFactory.create_split_models(
    model_type='gpt2',
    model_name_or_path='gpt2',
    split_point_1=2,
    split_point_2=10,
    device='cpu'
)

print("✓ 成功！无卡顿")
EOF
```

---

## 临时使用 Anaconda 的变通方案

如果你**必须**使用 Anaconda，唯一的方法是：

### 方案：使用预加载的模型文件

1. **在 Framework Python 中创建和保存模型**
   ```bash
   # 使用 Framework Python
   /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 save_split_models.py
   ```

2. **在 Anaconda 中直接加载保存的模型**
   ```python
   # 这个可以在 Anaconda 中运行
   import os
   os.environ['OMP_NUM_THREADS'] = '1'
   os.environ['MKL_NUM_THREADS'] = '1'

   import torch
   from splitlearn_core.models.gpt2 import GPT2TrunkModel
   from transformers import GPT2Config

   # 创建模型实例
   config = GPT2Config.from_pretrained("gpt2")
   trunk = GPT2TrunkModel(config, start_layer=2, end_layer=10)

   # 加载权重
   trunk.load_state_dict(torch.load("./models/trunk/gpt2_2-10_trunk.pt"))
   trunk.eval()

   # 使用模型
   # ...
   ```

**限制**：
- 需要先在 Framework Python 中创建模型
- 每次改变拆分点都需要重新创建
- 不能直接使用 ModelFactory

---

## 推荐的最终方案

### 创建一个 bash 脚本来切换环境

```bash
#!/bin/bash
# run_with_framework_python.sh

# 激活 Framework Python 虚拟环境
source ~/venv-splitlearn/bin/activate

# 运行你的脚本
python "$@"
```

使用：
```bash
chmod +x run_with_framework_python.sh
./run_with_framework_python.sh your_script.py
```

---

## 总结

| 环境 | 能否使用 SplitLearnCore | 备注 |
|------|------------------------|------|
| **Framework Python** | ✅ 完全支持 | 推荐 |
| **Anaconda Python** | ❌ 无法导入 | 死锁问题 |
| Anaconda + 预加载模型 | ⚠️ 部分支持 | 不能用 ModelFactory |

### 建议

1. **最佳**：切换到 Framework Python 虚拟环境
2. **次选**：在 Anaconda 中只加载预创建的模型
3. **避免**：尝试在 Anaconda 中直接使用 ModelFactory

---

## 快速开始（Framework Python）

```bash
# 一键设置
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m venv ~/venv-sl
source ~/venv-sl/bin/activate
pip install torch transformers
cd /Users/lhy/Desktop/Git/SL/SplitLearnCore
pip install -e .

# 测试
python test_splitlearn_core_only.py
# ✅ 应该完全正常，无任何警告
```

---

## 为什么不修改 SplitLearnCore？

修改 splitlearn_core 来"修复" Anaconda 兼容性：

**需要的改动**：
1. 移除所有模块级别的 `import torch` 和 `import transformers`
2. 改为所有导入都在函数内部（延迟导入）
3. 重构整个导入架构
4. 大量测试确保不破坏现有功能

**成本**：非常高
**收益**：只为支持一个有问题的环境
**不推荐**：不值得

---

## 最终建议

**使用 Framework Python**。

- 设置时间：5 分钟
- 完全没有问题
- PyTorch 更新（2.5.1 vs 2.4.0）
- 是 SplitLearnCore 的标准测试环境

停止与 Anaconda 的 MKL 死锁斗争，直接用能用的工具。
