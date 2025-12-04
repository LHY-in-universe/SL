# 导入问题分析

## 🔍 问题现象

当运行 `test_simple_load.py` 时，程序在导入 `splitlearn_core` 时就卡住，并出现：
```
[mutex.cc : 452] RAW: Lock blocking 0x145bdfb18
```

## 📋 原因分析

### 导入链分析

当执行 `import splitlearn_core` 时，`SplitLearnCore/src/splitlearn_core/__init__.py` 会执行以下导入：

```python
# 1. 版本信息（简单）
from .__version__ import __version__

# 2. 核心类（可能触发 torch 导入）
from .core import (
    BaseSplitModel,
    BaseBottomModel,
    BaseTrunkModel,
    BaseTopModel,
)

# 3. 工厂类（可能触发 torch/transformers 导入）
from .factory import ModelFactory

# 4. 工具类（可能触发 torch 导入）
from .utils import ParamMapper, StorageManager

# 5. 导入 models 模块（会触发模型注册）
from . import models

# 6. 导入 GPT-2 模型（会触发 transformers 导入）
from .models.gpt2 import GPT2BottomModel, GPT2TrunkModel, GPT2TopModel
```

### 关键问题

1. **`from .factory import ModelFactory`**：
   - `factory.py` 会导入 `torch` 和 `transformers`
   - 这会在导入时就初始化 PyTorch 的 C++ 后端
   - 导致 mutex 警告

2. **`from .models.gpt2 import ...`**：
   - `gpt2/trunk.py` 等文件会导入 `transformers.models.gpt2.modeling_gpt2`
   - 这会触发 transformers 库的初始化
   - 可能也会触发 mutex 警告

## 💡 解决方案

### 方案 1：延迟导入（推荐）

不在模块级别导入，而是在需要时才导入：

```python
# 不这样做：
from splitlearn_core.models.gpt2 import GPT2TrunkModel

# 而是这样做：
def load_model():
    from splitlearn_core.models.gpt2 import GPT2TrunkModel
    # 使用 GPT2TrunkModel
```

### 方案 2：直接使用 torch.load()

如果只是测试模型加载，可以直接使用 `torch.load()`，不需要导入 `splitlearn_core`：

```python
import torch

# 直接加载模型文件
model = torch.load("gpt2_trunk_full.pt", map_location='cpu', weights_only=False)
```

### 方案 3：接受 mutex 警告

`[mutex.cc : 452]` 警告通常不影响功能，可以：
- 忽略这个警告（它只是警告，不是错误）
- 设置环境变量抑制警告：
  ```python
  os.environ['GLOG_minloglevel'] = '2'
  ```

## 🎯 对于测试脚本的建议

由于 `test_simple_load.py` 的目标是测试模型加载，建议：

1. **如果只需要测试 `torch.load()`**：
   - 不需要导入 `splitlearn_core`
   - 直接使用 `torch.load()` 即可

2. **如果需要使用 core 库的模型类**：
   - 在函数内部延迟导入
   - 或者接受 mutex 警告（它不影响功能）

## 📝 总结

- **问题**：导入 `splitlearn_core` 会触发 `torch` 和 `transformers` 的初始化，导致 mutex 警告
- **影响**：警告不影响功能，但会阻塞输出
- **解决**：延迟导入或直接使用 `torch.load()`

