# 为什么代码支持 LoRA 微调 - 完整技术细节

## 目录

1. [核心原因分析](#核心原因分析)
2. [代码架构层次](#代码架构层次)
3. [PEFT 库工作原理](#peft-库工作原理)
4. [兼容性机制](#兼容性机制)
5. [详细技术流程](#详细技术流程)
6. [关键代码分析](#关键代码分析)

---

## 核心原因分析

### 为什么支持 LoRA？核心原因：

1. **你的模型使用标准的 PyTorch `nn.Module`**
2. **你的模型使用标准的 transformers 组件（GPT2Block）**
3. **PEFT 库通过名称模式匹配和包装机制工作**
4. **拆分模型仍然是完整的 PyTorch 模块树**

---

## 代码架构层次

### 完整的继承链

```
torch.nn.Module (PyTorch 基础类)
    ↓
BaseSplitModel (你的抽象基类)
    ↓
BaseBottomModel / BaseTopModel (你的抽象基类)
    ↓
GPT2BottomModel / GPT2TopModel (你的具体实现)
    ↓
包含 GPT2Block (来自 transformers 库)
    ↓
GPT2Block 内部包含标准的 nn.Linear 层
    ↓
PEFT 可以包装这些 nn.Linear 层
```

### 关键点分析

#### 1. 标准 PyTorch Module 继承

```python
# 你的代码
class GPT2BottomModel(BaseBottomModel):
    # BaseBottomModel 继承自 BaseSplitModel
    # BaseSplitModel 继承自 nn.Module
    pass

# 这意味着：
isinstance(GPT2BottomModel(...), nn.Module)  # True
```

**为什么重要？**
- PEFT 库可以遍历 `model.named_modules()`
- 可以访问所有子模块
- 可以替换或包装任何 `nn.Module`

#### 2. 使用标准的 transformers 组件

```python
# 你的代码（bottom.py 第 45-46 行）
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

self.h = nn.ModuleList(
    [GPT2Block(config, layer_idx=i) for i in range(end_layer)]
)
```

**为什么重要？**
- `GPT2Block` 是标准的 transformers 组件
- PEFT 库完全支持 GPT-2 架构
- 模块名称模式是已知的（`c_attn`, `c_fc`, `c_proj`）

#### 3. 模块名称和结构

```python
# GPT2Block 内部结构（来自 transformers 库）
class GPT2Block(nn.Module):
    def __init__(self, config, layer_idx):
        # Attention
        self.attn = GPT2Attention(config, layer_idx)
        # 其中包含：self.attn.c_attn (线性层)
        
        # MLP
        self.mlp = GPT2MLP(config)
        # 其中包含：self.mlp.c_fc, self.mlp.c_proj (线性层)
```

**你的模型结构：**
```
GPT2BottomModel (nn.Module)
├── wte (nn.Embedding)
├── wpe (nn.Embedding)
├── drop (nn.Dropout)
└── h (nn.ModuleList)
    ├── h[0] (GPT2Block)
    │   ├── attn
    │   │   └── c_attn (nn.Linear / Conv1D)  ← PEFT 目标
    │   └── mlp
    │       ├── c_fc (nn.Linear)  ← PEFT 目标
    │       └── c_proj (nn.Linear)  ← PEFT 目标
    └── h[1] (GPT2Block)
        └── ... (相同结构)
```

**为什么重要？**
- PEFT 可以通过名称模式找到这些模块
- 模块结构是标准的，PEFT 知道如何包装

---

## PEFT 库工作原理

### 核心机制：模块替换

PEFT 库的工作流程：

```python
# PEFT 内部工作流程（简化版）

def get_peft_model(model, peft_config):
    # 1. 遍历模型的所有模块
    for name, module in model.named_modules():
        # 2. 检查是否是目标模块（通过名称匹配）
        if is_target_module(name, peft_config.target_modules):
            # 例如：name = "h.0.attn.c_attn"
            # target_modules = ["c_attn"]
            # 匹配成功！
            
            # 3. 获取父模块
            parent = get_parent_module(model, name)
            
            # 4. 创建 LoRA 包装器
            lora_module = LoRALinear(original_module, config)
            
            # 5. 替换原模块
            setattr(parent, child_name, lora_module)
    
    return model  # 返回包装后的模型
```

### 名称匹配机制

```python
# 你的模型中的模块名称
"h.0.attn.c_attn"  # Bottom 模型的第一个 block 的 attention
"h.1.attn.c_attn"  # Bottom 模型的第二个 block 的 attention

# PEFT 配置
target_modules = ["c_attn", "c_fc", "c_proj"]

# 匹配逻辑
for target in target_modules:
    if target in module_name:  # "c_attn" in "h.0.attn.c_attn" → True
        # 匹配成功，应用 LoRA
```

### LoRA 包装机制

```python
# PEFT 创建的 LoRALinear 包装器

class LoRALinear(nn.Module):
    def __init__(self, original_layer, config):
        self.original_layer = original_layer
        
        # 冻结原始权重
        for param in original_layer.parameters():
            param.requires_grad = False
        
        # 添加 LoRA 矩阵
        self.lora_A = nn.Parameter(...)  # 可训练
        self.lora_B = nn.Parameter(...)  # 可训练
    
    def forward(self, x):
        # 原始输出（冻结权重）
        output = self.original_layer(x)
        
        # LoRA 输出（可训练）
        lora_output = self.lora_B @ (self.lora_A @ x) * scaling
        
        return output + lora_output
```

---

## 兼容性机制

### 1. PyTorch Module 兼容性

```python
# 你的模型
bottom = GPT2BottomModel(config, end_layer=2)

# 检查兼容性
print(type(bottom))  # <class 'GPT2BottomModel'>
print(isinstance(bottom, nn.Module))  # True ✅

# PEFT 可以工作，因为：
# - 它是 nn.Module 的子类
# - 可以使用 model.named_modules() 遍历
# - 可以访问和替换子模块
```

### 2. 模块结构兼容性

```python
# PEFT 需要访问的模块路径
# 完整路径：h.0.attn.c_attn

# 你的模型结构：
bottom.h[0].attn.c_attn  # ✅ 存在
bottom.h[0].mlp.c_fc     # ✅ 存在
bottom.h[0].mlp.c_proj   # ✅ 存在

# PEFT 可以通过名称路径访问：
module = bottom
for part in "h.0.attn.c_attn".split("."):
    module = getattr(module, part)
# 成功获取到线性层！
```

### 3. 标准组件兼容性

```python
# 你使用的 GPT2Block 来自 transformers 库
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

# 这意味着：
# 1. 模块结构是标准的（PEFT 知道这个结构）
# 2. 模块名称是标准的（c_attn, c_fc, c_proj）
# 3. 参数格式是标准的（可以冻结和替换）
```

---

## 详细技术流程

### 步骤 1: PEFT 如何找到目标模块

```python
# PEFT 内部代码（简化）

def find_target_modules(model, target_modules):
    target_modules_found = []
    
    # 遍历所有模块
    for name, module in model.named_modules():
        # 检查每个目标模块名称
        for target in target_modules:
            if target in name:  # 字符串包含检查
                # 例如：name = "h.0.attn.c_attn"
                #      target = "c_attn"
                #      "c_attn" in "h.0.attn.c_attn" → True ✅
                target_modules_found.append((name, module))
                break
    
    return target_modules_found

# 在你的模型中：
# "h.0.attn.c_attn" → 匹配 "c_attn" ✅
# "h.0.mlp.c_fc"    → 匹配 "c_fc" ✅
# "h.0.mlp.c_proj"  → 匹配 "c_proj" ✅
```

### 步骤 2: PEFT 如何替换模块

```python
# PEFT 内部代码（简化）

def replace_with_lora(model, module_path, original_module, lora_config):
    # 1. 获取父模块和子模块名
    # module_path = "h.0.attn.c_attn"
    parts = module_path.split(".")
    parent_path = ".".join(parts[:-1])  # "h.0.attn"
    child_name = parts[-1]               # "c_attn"
    
    # 2. 获取父模块对象
    parent = model
    for part in parent_path.split("."):
        parent = getattr(parent, part)
    # parent = bottom.h[0].attn
    
    # 3. 创建 LoRA 包装器
    lora_module = LoRALinear(original_module, lora_config)
    
    # 4. 替换
    setattr(parent, child_name, lora_module)
    # 现在：bottom.h[0].attn.c_attn = LoRALinear(...)
```

### 步骤 3: LoRA 如何在前向传播中工作

```python
# LoRALinear 的前向传播

class LoRALinear(nn.Module):
    def forward(self, x):
        # 1. 原始线性层输出（冻结权重）
        original_output = self.original_layer(x)
        #    W_original @ x  (W_original 冻结，不更新)
        
        # 2. LoRA 输出（可训练权重）
        lora_output = (self.lora_B @ (self.lora_A @ x)) * scaling
        #    B @ (A @ x) * (alpha / rank)
        #    其中 A 和 B 是可训练的
        
        # 3. 合并
        return original_output + lora_output
        #    = W_original @ x + B @ (A @ x) * scaling
        #    = (W_original + B @ A * scaling) @ x
```

---

## 关键代码分析

### 你的模型结构（Bottom）

```python
# SplitLearnCore/src/splitlearn_core/models/gpt2/bottom.py

class GPT2BottomModel(BaseBottomModel):  # 继承链：nn.Module
    def __init__(self, config, end_layer=2):
        # 使用标准的 GPT2Block
        self.h = nn.ModuleList([
            GPT2Block(config, layer_idx=i)  # 来自 transformers 库
            for i in range(end_layer)
        ])
```

**关键点：**
- `self.h` 是 `nn.ModuleList`
- 包含标准的 `GPT2Block` 对象
- `GPT2Block` 内部有标准的线性层（`c_attn`, `c_fc`, `c_proj`）

### PEFT 应用过程

```python
# test/client/train_lora_simple.py

from peft import LoraConfig, get_peft_model, TaskType

# 1. 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,
    target_modules=["c_attn", "c_fc", "c_proj"]
)

# 2. 应用 PEFT
bottom_peft = get_peft_model(bottom, lora_config)

# 内部发生了什么：
# - PEFT 遍历 bottom.named_modules()
# - 找到所有包含 "c_attn", "c_fc", "c_proj" 的模块
# - 将这些模块替换为 LoRALinear 包装器
# - 冻结原始权重，添加可训练的 LoRA 参数
```

### 模块名称匹配示例

```python
# 你的模型中的实际模块名称：

# Bottom 模型：
bottom.named_modules()
# 输出：
# ('wte', Embedding(...))
# ('wpe', Embedding(...))
# ('drop', Dropout(...))
# ('h', ModuleList(...))
# ('h.0', GPT2Block(...))
# ('h.0.attn', GPT2Attention(...))
# ('h.0.attn.c_attn', Linear(...))  ← PEFT 匹配 "c_attn" ✅
# ('h.0.mlp', GPT2MLP(...))
# ('h.0.mlp.c_fc', Linear(...))     ← PEFT 匹配 "c_fc" ✅
# ('h.0.mlp.c_proj', Linear(...))   ← PEFT 匹配 "c_proj" ✅
# ('h.1', GPT2Block(...))
# ... 重复结构

# PEFT 配置：
target_modules = ["c_attn", "c_fc", "c_proj"]

# 匹配过程：
for name, module in bottom.named_modules():
    for target in ["c_attn", "c_fc", "c_proj"]:
        if target in name:  # 字符串包含检查
            # 找到匹配，替换为 LoRALinear
```

---

## 为什么拆分模型仍然兼容？

### 关键理解

虽然模型被拆分成了三个部分，但每个部分仍然是：
1. **完整的 PyTorch `nn.Module`**
2. **包含标准的 transformers 组件**
3. **具有标准的模块结构**

### 拆分不影响兼容性

```
完整模型:
GPT2LMHeadModel
  └─ transformer
      └─ h (ModuleList)
          └─ h[0] (GPT2Block)
              └─ attn.c_attn (Linear)  ← PEFT 可以访问

拆分模型:
GPT2BottomModel
  └─ h (ModuleList)
      └─ h[0] (GPT2Block)
          └─ attn.c_attn (Linear)  ← PEFT 同样可以访问！

结构相同，只是层级不同！
```

### 为什么可以？

```python
# PEFT 通过名称路径访问模块
# 完整模型：transformer.h.0.attn.c_attn
# 拆分模型：h.0.attn.c_attn

# PEFT 使用字符串匹配，不关心完整路径
# 只要名称中包含 "c_attn" 就匹配

# 所以：
"transformer.h.0.attn.c_attn"  → 匹配 "c_attn" ✅
"h.0.attn.c_attn"              → 匹配 "c_attn" ✅

# 两者都可以！
```

---

## 技术实现细节

### 1. 模块遍历机制

```python
# PyTorch 的 named_modules() 方法
# 递归遍历所有子模块

model = GPT2BottomModel(...)
for name, module in model.named_modules():
    print(name, type(module))

# 输出：
# wte <class 'torch.nn.modules.sparse.Embedding'>
# h <class 'torch.nn.modules.container.ModuleList'>
# h.0 <class 'transformers.models.gpt2.modeling_gpt2.GPT2Block'>
# h.0.attn <class 'transformers.models.gpt2.modeling_gpt2.GPT2Attention'>
# h.0.attn.c_attn <class 'torch.nn.modules.linear.Linear'>

# PEFT 使用这个机制找到所有模块
```

### 2. 模块替换机制

```python
# PyTorch 允许动态替换模块

# 原始模块
original = bottom.h[0].attn.c_attn
print(type(original))  # <class 'torch.nn.modules.linear.Linear'>

# 创建 LoRA 包装器
lora_wrapper = LoRALinear(original, config)

# 替换
bottom.h[0].attn.c_attn = lora_wrapper

# 现在
print(type(bottom.h[0].attn.c_attn))  # <class 'LoRALinear'>

# 前向传播时
x = torch.randn(1, 10, 768)
output = bottom(x)  # 会自动使用 LoRALinear！
```

### 3. 权重冻结机制

```python
# LoRALinear 内部

class LoRALinear(nn.Module):
    def __init__(self, original_layer, config):
        self.original_layer = original_layer
        
        # 冻结原始权重
        for param in original_layer.parameters():
            param.requires_grad = False
            # 这样在反向传播时，这些参数不会更新
        
        # 添加可训练的 LoRA 参数
        self.lora_A = nn.Parameter(...)
        self.lora_B = nn.Parameter(...)
        # 这些参数的 requires_grad = True（默认）
```

### 4. 梯度流机制

```python
# 训练时的梯度流

# 前向传播
x → LoRALinear → output
   ├─ original_layer(x)  (冻结，无梯度)
   └─ lora_B @ (lora_A @ x)  (可训练，有梯度)

# 反向传播
loss.backward()
   ├─ 计算 lora_B 的梯度 ✅
   ├─ 计算 lora_A 的梯度 ✅
   └─ 不计算 original_layer 的梯度 ❌ (冻结)

# 参数更新
optimizer.step()
   ├─ 更新 lora_B ✅
   ├─ 更新 lora_A ✅
   └─ 不更新 original_layer ❌ (冻结)
```

---

## 详细的模块结构分析

### Bottom 模型完整结构

```python
GPT2BottomModel (nn.Module)
│
├── wte: Embedding(50257, 768)  # Token embedding
│
├── wpe: Embedding(1024, 768)   # Position embedding
│
├── drop: Dropout(p=0.1)        # Dropout
│
└── h: ModuleList                # Transformer blocks
    │
    ├── h[0]: GPT2Block         # 第一个 transformer block
    │   │
    │   ├── attn: GPT2Attention
    │   │   │
    │   │   ├── c_attn: Linear(768, 2304)  ← PEFT 目标 1
    │   │   │   # 实际上是 Conv1D，但 PEFT 支持
    │   │   │
    │   │   ├── c_proj: Linear(768, 768)   # 不在 target_modules
    │   │   │
    │   │   └── ...
    │   │
    │   └── mlp: GPT2MLP
    │       │
    │       ├── c_fc: Linear(768, 3072)    ← PEFT 目标 2
    │       │
    │       └── c_proj: Linear(3072, 768)  ← PEFT 目标 3
    │
    └── h[1]: GPT2Block         # 第二个 transformer block
        └── ... (相同结构)
```

### PEFT 应用后的结构

```python
GPT2BottomModel (nn.Module)
│
├── wte: Embedding(...)  # 不变
│
├── wpe: Embedding(...)  # 不变
│
├── drop: Dropout(...)   # 不变
│
└── h: ModuleList
    │
    └── h[0]: GPT2Block
        │
        ├── attn: GPT2Attention
        │   │
        │   └── c_attn: LoRALinear  ← 被替换了！
        │       ├── original_layer: Linear(冻结)
        │       ├── lora_A: Parameter(8, 768)  (可训练)
        │       └── lora_B: Parameter(2304, 8) (可训练)
        │
        └── mlp: GPT2MLP
            ├── c_fc: LoRALinear    ← 被替换了！
            │   ├── original_layer: Linear(冻结)
            │   ├── lora_A: Parameter(8, 768)  (可训练)
            │   └── lora_B: Parameter(3072, 8) (可训练)
            │
            └── c_proj: LoRALinear  ← 被替换了！
                ├── original_layer: Linear(冻结)
                ├── lora_A: Parameter(8, 3072) (可训练)
                └── lora_B: Parameter(768, 8)  (可训练)
```

---

## 关键代码路径追踪

### 1. 模型创建

```python
# 文件: SplitLearnCore/src/splitlearn_core/models/gpt2/bottom.py

class GPT2BottomModel(BaseBottomModel):
    def __init__(self, config, end_layer=2):
        # 继承自 BaseBottomModel
        # BaseBottomModel 继承自 BaseSplitModel
        # BaseSplitModel 继承自 nn.Module
        
        # 使用标准组件
        self.h = nn.ModuleList([
            GPT2Block(config, layer_idx=i)  # 来自 transformers 库
            for i in range(end_layer)
        ])
```

**关键点：**
- 最终继承自 `nn.Module`
- 使用标准的 `GPT2Block`
- 结构符合 PyTorch 规范

### 2. PEFT 应用

```python
# 文件: test/client/train_lora_simple.py

from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(
    target_modules=["c_attn", "c_fc", "c_proj"]
)

bottom_peft = get_peft_model(bottom, lora_config)
```

**内部流程：**

```python
# PEFT 库内部（简化）

def get_peft_model(model, peft_config):
    # 1. 创建 PeftModel 包装器
    peft_model = PeftModelForFeatureExtraction(model, peft_config)
    
    # 2. 查找目标模块
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, Conv1D)):  # 检查类型
            for target in peft_config.target_modules:
                if target in name:  # 名称匹配
                    target_modules.append((name, module))
    
    # 3. 替换为 LoRA 版本
    for name, module in target_modules:
        parent, child_name = get_parent_module(model, name)
        lora_module = create_lora_module(module, peft_config)
        setattr(parent, child_name, lora_module)
    
    return peft_model
```

### 3. 前向传播

```python
# 应用 PEFT 后的前向传播

# 原始调用
hidden = bottom(input_ids)

# 内部流程：
# 1. bottom.forward(input_ids)
# 2. bottom.wte(input_ids) → embeddings
# 3. bottom.h[0](embeddings)
#    3.1 bottom.h[0].attn.c_attn(embeddings)  ← 这里是 LoRALinear
#        → LoRALinear.forward(embeddings)
#           → original_layer(embeddings) + lora_B @ (lora_A @ embeddings)
#    3.2 ... 其他层
# 4. 返回 hidden_states
```

---

## 为什么这样设计就能兼容？

### 1. 抽象层次正确

```
标准接口层: nn.Module (PyTorch 标准)
    ↓
你的抽象层: BaseSplitModel (你的抽象)
    ↓
实现层: GPT2BottomModel (你的实现)
    ↓
组件层: GPT2Block (transformers 标准)
    ↓
基础层: nn.Linear (PyTorch 标准)
```

每一层都遵循标准接口，所以兼容。

### 2. 模块化设计

```python
# 你的模型是模块化的

bottom.h[0]  # 可以独立访问
bottom.h[0].attn  # 可以独立访问
bottom.h[0].attn.c_attn  # 可以独立访问和替换

# PEFT 利用这个特性进行模块替换
```

### 3. 名称约定一致

```python
# transformers 库的标准命名
GPT2Block.attn.c_attn  # 标准的命名约定

# 你的模型遵循这个约定
bottom.h[0].attn.c_attn  # 同样的命名

# PEFT 知道这个命名约定，可以匹配
```

---

## 完整的技术流程示例

### 示例：应用 LoRA 到 Bottom 模型

```python
# 步骤 1: 创建模型
from splitlearn_core.models.gpt2 import GPT2BottomModel
from transformers import GPT2Config

config = GPT2Config()
bottom = GPT2BottomModel(config, end_layer=2)

# 步骤 2: 检查模块结构
for name, module in bottom.named_modules():
    print(f"{name}: {type(module).__name__}")

# 输出（关键部分）：
# h.0.attn.c_attn: Linear
# h.0.mlp.c_fc: Linear
# h.0.mlp.c_proj: Linear

# 步骤 3: 应用 PEFT
from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(
    target_modules=["c_attn", "c_fc", "c_proj"]
)

bottom_peft = get_peft_model(bottom, lora_config)

# 步骤 4: 检查应用结果
for name, module in bottom_peft.named_modules():
    if 'lora' in name.lower():
        print(f"{name}: {type(module).__name__}")

# 输出：
# base_model.h.0.attn.c_attn: LoRALinear
# base_model.h.0.mlp.c_fc: LoRALinear
# base_model.h.0.mlp.c_proj: LoRALinear

# 步骤 5: 验证参数
trainable_params = [p for p in bottom_peft.parameters() if p.requires_grad]
print(f"可训练参数: {sum(p.numel() for p in trainable_params)}")
# 输出: 可训练参数: 196608 (只有 LoRA 参数)
```

---

## 关键技术点总结

### 1. 为什么可以直接使用标准库？

**原因：**
- 你的模型是标准的 `nn.Module` 子类
- 使用标准的 transformers 组件
- 模块结构符合标准约定
- PEFT 通过通用机制工作（不依赖特定模型类）

### 2. 兼容性的技术基础

**基础 1: PyTorch Module 系统**
```python
# 所有 PyTorch 模块都支持：
model.named_modules()  # 遍历所有子模块
getattr(model, 'attr') # 访问属性
setattr(model, 'attr', value)  # 设置属性
```

**基础 2: 动态模块替换**
```python
# PyTorch 允许运行时替换模块
model.layer = NewLayer()  # 可以动态替换
```

**基础 3: 名称路径访问**
```python
# 可以通过名称路径访问嵌套模块
module = model
for name in "h.0.attn.c_attn".split("."):
    module = getattr(module, name)
```

### 3. LoRA 工作的技术原理

**原理 1: 权重冻结**
```python
# 原始权重被冻结
for param in original_layer.parameters():
    param.requires_grad = False
```

**原理 2: 可训练的低秩矩阵**
```python
# 只训练小的 LoRA 矩阵
lora_A: [rank, in_features]  # 小矩阵
lora_B: [out_features, rank]  # 小矩阵
```

**原理 3: 权重叠加**
```python
# 有效权重 = 原始权重 + LoRA 权重
W_effective = W_original + (B @ A) * scaling
```

---

## 实际验证

### 验证 1: 模块类型检查

```python
# 检查你的模型是否兼容
from splitlearn_core.models.gpt2 import GPT2BottomModel
from transformers import GPT2Config
import torch.nn as nn

bottom = GPT2BottomModel(GPT2Config(), end_layer=2)

# 验证 1: 是否是 nn.Module
print(isinstance(bottom, nn.Module))  # True ✅

# 验证 2: 是否有 named_modules 方法
print(hasattr(bottom, 'named_modules'))  # True ✅

# 验证 3: 模块结构是否正确
for name, module in bottom.named_modules():
    if 'c_attn' in name:
        print(f"找到目标模块: {name}, 类型: {type(module)}")
        # 输出: 找到目标模块: h.0.attn.c_attn, 类型: <class '...Linear'>
```

### 验证 2: PEFT 应用检查

```python
from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(target_modules=["c_attn"])

bottom_peft = get_peft_model(bottom, lora_config)

# 检查是否成功应用
for name, module in bottom_peft.named_modules():
    if 'lora' in name.lower() or 'LoRA' in str(type(module)):
        print(f"LoRA 模块: {name}")
        # 输出: LoRA 模块: base_model.h.0.attn.c_attn
```

---

## 总结

### 为什么你的代码支持 LoRA？技术原因：

1. **标准继承链**
   - `GPT2BottomModel` → `BaseBottomModel` → `BaseSplitModel` → `nn.Module`
   - 完全符合 PyTorch 标准

2. **标准组件使用**
   - 使用 `GPT2Block`（来自 transformers）
   - 包含标准的 `nn.Linear` 层
   - 模块名称符合标准约定

3. **模块化设计**
   - 模块可以独立访问
   - 支持动态替换
   - 结构清晰

4. **PEFT 通用机制**
   - 通过 `named_modules()` 遍历
   - 通过名称模式匹配
   - 通过模块替换实现

### 关键结论

**你的代码支持 LoRA 的根本原因：**
- 遵循了 PyTorch 的标准设计模式
- 使用了标准的 transformers 组件
- 保持了模块的独立性和可访问性

**这证明了你的架构设计是正确的！**
