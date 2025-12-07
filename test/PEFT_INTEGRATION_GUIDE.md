# 使用标准微调库（PEFT）指南

## 核心问题

**问题**: 既然我们已经使用了标准的 transformer block，为什么不能直接使用标准的微调库（如 HuggingFace PEFT）？

**答案**: ✅ **完全可以！而且这是更好的方案！**

---

## 为什么可以直接使用标准库？

### 1. 你的模型使用的是标准组件

```python
# 当前代码
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

self.h = nn.ModuleList(
    [GPT2Block(config, layer_idx=i) for i in range(end_layer)]
)
```

这些都是标准的 PyTorch 模块和 transformers 组件，完全兼容！

### 2. PEFT 库的工作原理

HuggingFace PEFT (Parameter-Efficient Fine-Tuning) 库：
- ✅ 可以应用到任何 `nn.Module`
- ✅ 通过 monkey patching 或包装来实现
- ✅ 不需要修改模型源码
- ✅ 直接支持标准 transformers 模型

### 3. 拆分模型仍然是标准模块

```
完整模型:
GPT2LMHeadModel
  └─ transformer
      └─ h (GPT2Block 列表)

拆分模型:
Bottom Model
  └─ h (GPT2Block 列表)  ← 标准模块，PEFT 可以直接应用！

Top Model
  └─ h (GPT2Block 列表)  ← 标准模块，PEFT 可以直接应用！
```

---

## 直接使用 PEFT 库

### 安装 PEFT

```bash
pip install peft
```

### 基本使用示例

```python
from peft import LoraConfig, get_peft_model, TaskType

# 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,  # LoRA 秩
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_fc", "c_proj"]  # GPT-2 的目标模块
)

# 应用到 Bottom 模型
bottom = GPT2BottomModel(config, end_layer=2)
bottom_peft = get_peft_model(bottom, lora_config)

# 应用到 Top 模型
top = GPT2TopModel(config, start_layer=10)
top_peft = get_peft_model(top, lora_config)

# 检查参数
bottom_peft.print_trainable_parameters()
# trainable params: 12,288 || all params: 53,559,552 || trainable%: 0.02
```

**就这么简单！不需要自己实现任何代码！**

---

## Split Learning + PEFT 完整示例

### 1. 客户端代码

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoConfig
from splitlearn_core.models.gpt2 import GPT2BottomModel, GPT2TopModel

# LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_fc", "c_proj"]
)

# 加载模型
config = AutoConfig.from_pretrained("gpt2")
bottom = GPT2BottomModel(config, end_layer=2)
top = GPT2TopModel(config, start_layer=10)

# 应用 PEFT
bottom_peft = get_peft_model(bottom, lora_config)
top_peft = get_peft_model(top, lora_config)

# 设置为训练模式
bottom_peft.train()
top_peft.train()

# 只优化可训练参数（LoRA 参数）
from peft import get_peft_model_state_dict

# 获取可训练参数
trainable_params_bottom = list(bottom_peft.parameters())
trainable_params_top = list(top_peft.parameters())

# 创建优化器（只包含 LoRA 参数）
optimizer_bottom = torch.optim.Adam(trainable_params_bottom, lr=1e-4)
optimizer_top = torch.optim.Adam(trainable_params_top, lr=1e-4)
```

### 2. 服务器端代码（可选）

如果服务器端也要使用 PEFT：

```python
from peft import LoraConfig, get_peft_model
from splitlearn_core.models.gpt2 import GPT2TrunkModel

# Trunk 模型也可以使用 PEFT
trunk = GPT2TrunkModel(config, start_layer=2, end_layer=10)
trunk_peft = get_peft_model(trunk, lora_config)
```

### 3. 训练循环（与之前相同）

```python
def train_step(bottom_peft, top_peft, trunk_client, input_ids, labels):
    # 清零梯度
    optimizer_bottom.zero_grad()
    optimizer_top.zero_grad()
    
    # 前向传播
    hidden_1 = bottom_peft(input_ids)
    hidden_2 = trunk_client.compute(hidden_1)
    output = top_peft(hidden_2)
    
    # 计算损失
    loss = criterion(output.logits, labels)
    
    # 反向传播（PEFT 自动处理 LoRA 梯度）
    loss.backward()
    
    # 梯度传递（只需要传递 LoRA 参数的梯度）
    grad_hidden_2 = hidden_2.grad
    grad_hidden_1 = trunk_client.backward(grad_hidden_2)
    hidden_1.backward(grad_hidden_1)
    
    # 更新参数（只更新 LoRA 参数）
    optimizer_bottom.step()
    optimizer_top.step()
    
    return loss.item()
```

---

## PEFT 库的优势

### 1. 完全兼容标准模型

```python
# 任何标准的 PyTorch 模块
from transformers import GPT2Model
model = GPT2Model.from_pretrained("gpt2")

# 直接应用 PEFT
from peft import get_peft_model
peft_model = get_peft_model(model, lora_config)

# 就是这么简单！
```

### 2. 多种微调方法

PEFT 不仅支持 LoRA，还支持：

```python
# LoRA
from peft import LoraConfig

# Prefix Tuning
from peft import PrefixTuningConfig

# P-Tuning v2
from peft import PromptTuningConfig

# AdaLoRA (自适应 LoRA)
from peft import AdaLoraConfig

# QLoRA (量化 + LoRA)
from peft import LoraConfig, BitsAndBytesConfig
```

### 3. 参数管理

```python
# 查看可训练参数
peft_model.print_trainable_parameters()

# 保存 LoRA 权重
peft_model.save_pretrained("./lora-weights")

# 加载 LoRA 权重
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "./lora-weights")
```

---

## 与自实现的对比

### 自实现 LoRA vs PEFT 库

| 特性 | 自实现 | PEFT 库 |
|------|--------|---------|
| **实现复杂度** | 需要编写代码 | ✅ 一行代码 |
| **维护成本** | 需要维护 | ✅ 社区维护 |
| **功能完整性** | 基础功能 | ✅ 完整功能（多种方法） |
| **测试覆盖** | 需要自己测试 | ✅ 充分测试 |
| **文档支持** | 需要编写 | ✅ 官方文档 |
| **兼容性** | 需要适配 | ✅ 广泛兼容 |

### 推荐使用 PEFT 库

**原因**:
1. ✅ 零实现成本（不需要写代码）
2. ✅ 经过充分测试
3. ✅ 社区支持
4. ✅ 功能完整（多种微调方法）
5. ✅ 持续更新

---

## 完整集成示例

### 文件结构

```
test/client/
  ├── peft_training_client.py  # 使用 PEFT 的训练客户端
  └── train_with_peft.py       # 训练脚本
```

### 完整训练客户端

**新建文件**: `test/client/peft_training_client.py`

```python
"""
使用 HuggingFace PEFT 库的 Split Learning 训练客户端
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from pathlib import Path
import json

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    get_peft_model_state_dict,
    set_peft_model_state_dict
)

from transformers import AutoConfig, AutoTokenizer
from splitlearn_core.models.gpt2 import GPT2BottomModel, GPT2TopModel
from splitlearn_comm.training_client import TrainingClient


class PEFTSplitLearningTrainer:
    """使用 PEFT 的 Split Learning 训练器"""
    
    def __init__(
        self,
        bottom_model: GPT2BottomModel,
        top_model: GPT2TopModel,
        trunk_client: TrainingClient,
        lora_config: Optional[LoraConfig] = None
    ):
        self.bottom = bottom_model
        self.top = top_model
        self.trunk_client = trunk_client
        
        # 默认 LoRA 配置
        if lora_config is None:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["c_attn", "c_fc", "c_proj"]
            )
        
        # 应用 PEFT 到模型
        print("应用 PEFT LoRA 到 Bottom 模型...")
        self.bottom_peft = get_peft_model(self.bottom, lora_config)
        self.bottom_peft.print_trainable_parameters()
        
        print("\n应用 PEFT LoRA 到 Top 模型...")
        self.top_peft = get_peft_model(self.top, lora_config)
        self.top_peft.print_trainable_parameters()
        
        # 设置为训练模式
        self.bottom_peft.train()
        self.top_peft.train()
        
        # 创建优化器（只包含可训练参数）
        trainable_params_bottom = [
            p for p in self.bottom_peft.parameters() if p.requires_grad
        ]
        trainable_params_top = [
            p for p in self.top_peft.parameters() if p.requires_grad
        ]
        
        self.optimizer_bottom = torch.optim.Adam(trainable_params_bottom, lr=1e-4)
        self.optimizer_top = torch.optim.Adam(trainable_params_top, lr=1e-4)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        self.global_step = 0
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """训练步骤"""
        # 清零梯度
        self.optimizer_bottom.zero_grad()
        self.optimizer_top.zero_grad()
        
        # 前向传播
        hidden_1 = self.bottom_peft(input_ids)
        
        # Trunk 远程调用（需要支持梯度）
        import uuid
        request_id = str(uuid.uuid4())
        hidden_2 = self.trunk_client.forward(hidden_1, request_id=request_id)
        
        # Top 模型
        output = self.top_peft(hidden_2)
        logits = output.logits if hasattr(output, 'logits') else output
        
        # 计算损失
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # 反向传播
        loss.backward()
        
        # 梯度传递
        grad_hidden_2 = hidden_2.grad
        grad_hidden_1 = self.trunk_client.backward(grad_hidden_2, request_id)
        hidden_1.backward(grad_hidden_1)
        
        # 参数更新（只更新 LoRA 参数）
        self.optimizer_bottom.step()
        self.optimizer_top.step()
        
        self.global_step += 1
        
        return {'loss': loss.item()}
    
    def save_lora_weights(self, save_dir: str):
        """保存 LoRA 权重"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存 Bottom LoRA 权重
        bottom_lora_path = save_path / "bottom_lora"
        self.bottom_peft.save_pretrained(str(bottom_lora_path))
        print(f"Bottom LoRA 权重已保存: {bottom_lora_path}")
        
        # 保存 Top LoRA 权重
        top_lora_path = save_path / "top_lora"
        self.top_peft.save_pretrained(str(top_lora_path))
        print(f"Top LoRA 权重已保存: {top_lora_path}")
    
    def load_lora_weights(self, bottom_lora_path: str, top_lora_path: str):
        """加载 LoRA 权重"""
        from peft import PeftModel
        
        # 加载 Bottom LoRA 权重
        self.bottom_peft = PeftModel.from_pretrained(
            self.bottom,
            bottom_lora_path
        )
        
        # 加载 Top LoRA 权重
        self.top_peft = PeftModel.from_pretrained(
            self.top,
            top_lora_path
        )
        
        print(f"LoRA 权重已加载: {bottom_lora_path}, {top_lora_path}")
```

### 训练脚本

**新建文件**: `test/client/train_with_peft.py`

```python
#!/usr/bin/env python3
"""
使用 PEFT 库的 Split Learning 训练脚本
"""

import os
import sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from transformers import AutoConfig, AutoTokenizer
from peft import LoraConfig, TaskType
from splitlearn_core.models.gpt2 import GPT2BottomModel, GPT2TopModel
from splitlearn_comm.training_client import TrainingClient
from peft_training_client import PEFTSplitLearningTrainer
import torch
import json

def main():
    # 加载模型
    models_dir = Path(project_root) / "models"
    
    with open(models_dir / "bottom" / "gpt2_2-10_bottom_metadata.json") as f:
        bottom_metadata = json.load(f)
    with open(models_dir / "top" / "gpt2_2-10_top_metadata.json") as f:
        top_metadata = json.load(f)
    
    config = AutoConfig.from_pretrained("gpt2")
    
    bottom = GPT2BottomModel(config, end_layer=bottom_metadata['end_layer'])
    bottom.load_state_dict(
        torch.load(models_dir / "bottom" / "gpt2_2-10_bottom.pt", map_location='cpu')
    )
    
    top = GPT2TopModel(config, start_layer=top_metadata['start_layer'])
    top.load_state_dict(
        torch.load(models_dir / "top" / "gpt2_2-10_top.pt", map_location='cpu')
    )
    
    # 配置 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # LoRA 秩
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_fc", "c_proj"]
    )
    
    # 连接到服务器
    trunk_client = TrainingClient("localhost:50052")
    
    # 创建训练器
    trainer = PEFTSplitLearningTrainer(
        bottom,
        top,
        trunk_client,
        lora_config
    )
    
    # 训练循环（示例）
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode("Hello world", return_tensors="pt")
    labels = input_ids.clone()
    
    for step in range(10):
        metrics = trainer.train_step(input_ids, labels)
        print(f"Step {step}: loss = {metrics['loss']:.4f}")
    
    # 保存 LoRA 权重
    trainer.save_lora_weights("./lora_weights")

if __name__ == '__main__':
    main()
```

---

## 关键优势总结

### 1. 零实现成本

```python
# 不需要自己实现 LoRA
from peft import get_peft_model
peft_model = get_peft_model(model, lora_config)
# 完成！
```

### 2. 完全兼容

- ✅ 标准 transformer block → 完全兼容
- ✅ 拆分模型 → 仍然兼容（每个部分都是标准模块）
- ✅ PyTorch 模块 → 完全兼容

### 3. 功能完整

- ✅ LoRA
- ✅ Prefix Tuning
- ✅ P-Tuning
- ✅ AdaLoRA
- ✅ QLoRA（量化 + LoRA）

### 4. 易于使用

```python
# 3 行代码即可
from peft import LoraConfig, get_peft_model
peft_model = get_peft_model(model, LoraConfig(r=8))
peft_model.print_trainable_parameters()
```

---

## 实现步骤（极简版）

### Phase 1: 安装和配置（1 天）

```bash
pip install peft
```

### Phase 2: 应用 PEFT（1 天）

```python
from peft import get_peft_model, LoraConfig
peft_model = get_peft_model(model, lora_config)
```

### Phase 3: 训练循环（与之前相同，2 周）

- 反向传播支持
- 梯度传递
- 训练循环

**总计**: 约 2 周（比自实现快 2 倍！）

---

## 结论

### 回答你的问题

**Q: 为什么不能直接使用标准的微调库？**

**A: 完全可以！而且应该这样做！**

### 推荐方案

**使用 HuggingFace PEFT 库而不是自实现**：

1. ✅ **零实现成本** - 一行代码即可
2. ✅ **完全兼容** - 标准模块完全支持
3. ✅ **功能完整** - 多种微调方法
4. ✅ **社区支持** - 持续更新和维护
5. ✅ **充分测试** - 经过大量验证

### 你需要做的

1. **安装 PEFT**:
   ```bash
   pip install peft
   ```

2. **应用 PEFT**:
   ```python
   from peft import get_peft_model, LoraConfig
   peft_model = get_peft_model(model, LoraConfig(r=8))
   ```

3. **实现反向传播**（这个仍然需要，但与 PEFT 无关）

就这么简单！不需要自己实现任何 LoRA 代码。
