# LoRA 微调实现指南 - Split Learning 场景

## 核心问题

**问题**: 如果只增加反向传播功能，能否实现 LoRA 等微调功能？我们使用的是标准的 transformer block。

**答案**: ✅ **完全可以！** 而且这是一个更简单、更高效的方案。

---

## 为什么 LoRA 更适合 Split Learning？

### 1. LoRA 的优势

LoRA (Low-Rank Adaptation) 是一种参数高效的微调方法：

```
原始权重更新:
W_new = W_old - lr * grad_W

LoRA 权重更新:
W_new = W_old  (冻结，不更新)
ΔW = BA  (只更新小矩阵 A 和 B)
W_effective = W_old + ΔW = W_old + BA
```

**关键特点**:
- ✅ 只训练少量参数（通常 < 1%）
- ✅ 梯度计算量小
- ✅ 传输开销低（只需要传输 LoRA 参数的梯度）
- ✅ 兼容标准 transformer block

### 2. Split Learning + LoRA 的完美结合

```
标准训练:
- 需要传输所有参数的梯度（几 GB）
- 内存占用大
- 通信开销高

LoRA 训练:
- 只需要传输 LoRA 参数的梯度（几 MB）
- 内存占用小
- 通信开销低
```

---

## LoRA 原理回顾

### 标准权重更新 vs LoRA

```
┌─────────────────────────────────────────────────────────┐
│                  标准全量微调                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  原始模型:  W [d × d]  (例如: 768 × 768 = 589K 参数)    │
│  更新方式:  W_new = W_old - lr * grad_W                 │
│  参数数量:  全部参数都要更新                            │
│                                                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  LoRA 微调                               │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  原始模型:  W [d × d]  (冻结，不更新)                    │
│  LoRA 矩阵: A [r × d], B [d × r]  (r << d, 例如 r=8)   │
│  有效权重:  W_effective = W + BA                        │
│  更新方式:  只更新 A 和 B                               │
│  参数数量:  r × d × 2 = 8 × 768 × 2 ≈ 12K 参数         │
│            (只有 2% 的参数需要更新！)                    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### LoRA 的数学表示

```
前向传播:
h = (W + BA) x
  = Wx + BAx
  = Wx + B(Ax)

其中:
- W: 原始权重 [d × d]，冻结
- A: 低秩矩阵 [r × d]，可训练
- B: 低秩矩阵 [d × r]，可训练
- r: 秩（rank），通常 4-16

参数效率:
原始: d² 参数
LoRA: 2rd 参数
比例: 2rd / d² = 2r / d

例如 d=768, r=8:
比例 = 16 / 768 ≈ 2%
```

---

## 实现方案

### 方案对比

| 方案 | 需要的工作 | 优势 | 劣势 |
|------|-----------|------|------|
| **完整训练** | 反向传播 + 梯度传递 + 参数更新 | 可以更新所有参数 | 通信开销大、内存占用大 |
| **LoRA 微调** | 反向传播 + LoRA 适配器 | 通信开销小、实现简单 | 只能更新部分参数 |

### 实现 LoRA 的两种方式

#### 方式 1: 包装标准 Block（推荐）

不需要修改标准 GPT2Block，只需要包装它：

```python
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

class LoRALinear(nn.Module):
    """LoRA 线性层包装器"""
    
    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # 冻结原始权重
        for param in original_layer.parameters():
            param.requires_grad = False
        
        # 创建 LoRA 矩阵
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.scaling = alpha / rank
    
    def forward(self, x):
        # 原始输出
        output = self.original_layer(x)
        
        # LoRA 输出
        lora_output = self.lora_B(self.lora_A(x) * self.scaling)
        
        return output + lora_output

class LoRAGPT2Block(nn.Module):
    """包装标准 GPT2Block 添加 LoRA"""
    
    def __init__(self, block: GPT2Block, rank: int = 8):
        super().__init__()
        self.block = block
        
        # 在 Attention 和 MLP 的线性层添加 LoRA
        self._apply_lora_to_block(block, rank)
    
    def _apply_lora_to_block(self, block, rank):
        """为 Block 中的线性层添加 LoRA"""
        # Attention 的 Q, K, V, O 投影
        if hasattr(block.attn, 'c_attn'):
            # GPT-2 使用组合的 QKV 投影
            block.attn.c_attn = LoRALinear(block.attn.c_attn, rank)
        
        # MLP 的两个线性层
        if hasattr(block.mlp, 'c_fc'):
            block.mlp.c_fc = LoRALinear(block.mlp.c_fc, rank)
        if hasattr(block.mlp, 'c_proj'):
            block.mlp.c_proj = LoRALinear(block.mlp.c_proj, rank)
    
    def forward(self, *args, **kwargs):
        return self.block(*args, **kwargs)
```

#### 方式 2: 直接修改模型（更灵活）

在模型初始化时直接替换线性层：

```python
def apply_lora_to_model(model, rank=8, target_modules=None):
    """
    为模型应用 LoRA
    
    Args:
        model: 要应用 LoRA 的模型
        rank: LoRA 的秩
        target_modules: 目标模块名称（如果为 None，应用所有线性层）
    """
    if target_modules is None:
        target_modules = ['c_attn', 'c_fc', 'c_proj', 'lm_head']
    
    for name, module in model.named_modules():
        # 检查是否是目标模块
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # 替换为 LoRA 版本
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent_module = model
                for p in parent_name.split('.'):
                    if p:
                        parent_module = getattr(parent_module, p)
                
                lora_module = LoRALinear(module, rank=rank)
                setattr(parent_module, child_name, lora_module)
    
    return model
```

---

## Split Learning + LoRA 实现

### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    客户端 (Client)                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Bottom Model                                               │
│  ┌─────────────────────────────────────┐                    │
│  │ 标准 GPT2Block (冻结)               │                    │
│  │  + LoRA 适配器 (可训练)              │                    │
│  │                                     │                    │
│  │  原始权重: 冻结                      │                    │
│  │  LoRA 参数: 可训练 (只有几 MB)       │                    │
│  └─────────────────────────────────────┘                    │
│                                                              │
│  Top Model                                                  │
│  ┌─────────────────────────────────────┐                    │
│  │ 标准 GPT2Block (冻结)               │                    │
│  │  + LoRA 适配器 (可训练)              │                    │
│  │  + LM Head (可以冻结或微调)          │                    │
│  └─────────────────────────────────────┘                    │
│                                                              │
│  梯度传输: 只传输 LoRA 参数的梯度 (几 MB)                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ 网络传输 (小)
                          │
┌─────────────────────────┼───────────────────────────────────┐
│                         │                                    │
│  Trunk Model (服务器)    │                                    │
│  ┌───────────────────────▼───────────────────┐               │
│  │ 标准 GPT2Block (冻结)                     │               │
│  │  + LoRA 适配器 (可训练)                    │               │
│  │                                           │               │
│  │  原始权重: 冻结                            │               │
│  │  LoRA 参数: 可训练 (只有几 MB)             │               │
│  └───────────────────────────────────────────┘               │
│                                                              │
│  梯度传输: 只传输 LoRA 参数的梯度 (几 MB)                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 完整实现代码

#### 1. LoRA 适配器实现

**新建文件**: `SplitLearnCore/src/splitlearn_core/adapters/lora.py`

```python
"""
LoRA (Low-Rank Adaptation) 适配器实现
"""

import math
import torch
import torch.nn as nn
from typing import Optional

class LoRALinear(nn.Module):
    """
    LoRA 线性层
    
    将标准的 nn.Linear 包装为 LoRA 版本，只训练低秩矩阵。
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 冻结原始权重
        for param in original_layer.parameters():
            param.requires_grad = False
        
        # LoRA 矩阵
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化 LoRA 参数"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出（冻结权重）
        output = self.original_layer(x)
        
        # LoRA 输出（可训练）
        x_dropout = self.lora_dropout(x)
        lora_output = (self.lora_B @ (self.lora_A @ x_dropout.transpose(0, 1))).transpose(0, 1)
        lora_output = lora_output * self.scaling
        
        return output + lora_output
    
    def get_lora_parameters(self):
        """获取 LoRA 参数（用于优化器）"""
        return [self.lora_A, self.lora_B]
    
    def merge_weights(self):
        """
        合并 LoRA 权重到原始权重（用于推理加速）
        合并后可以直接使用原始权重，不需要 LoRA 计算
        """
        if not self.training:
            with torch.no_grad():
                delta_w = self.lora_B @ self.lora_A * self.scaling
                self.original_layer.weight.data += delta_w
    
    def unmerge_weights(self):
        """分离权重（用于恢复训练）"""
        # 注意：这需要保存原始权重
        # 实际使用中建议不合并，保持分离状态
        pass

class LoRAAdapter:
    """LoRA 适配器工具类"""
    
    @staticmethod
    def apply_lora_to_linear_layer(
        layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ) -> LoRALinear:
        """将线性层替换为 LoRA 版本"""
        return LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout)
    
    @staticmethod
    def apply_lora_to_model(
        model: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        target_modules: Optional[list] = None,
        dropout: float = 0.0
    ) -> nn.Module:
        """
        为模型应用 LoRA
        
        Args:
            model: 要应用 LoRA 的模型
            rank: LoRA 的秩
            alpha: LoRA 的缩放因子
            target_modules: 目标模块名称列表
            dropout: LoRA dropout 概率
        
        Returns:
            应用了 LoRA 的模型
        """
        if target_modules is None:
            # GPT-2 默认目标模块
            target_modules = ['c_attn', 'c_fc', 'c_proj']
        
        replaced_modules = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # 检查是否是目标模块
                if any(target in name for target in target_modules):
                    # 获取父模块和子模块名
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    
                    # 获取父模块
                    parent_module = model
                    for p in parent_name.split('.'):
                        if p:
                            parent_module = getattr(parent_module, p)
                    
                    # 替换为 LoRA 版本
                    lora_module = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                    setattr(parent_module, child_name, lora_module)
                    replaced_modules[name] = lora_module
        
        print(f"已为 {len(replaced_modules)} 个模块应用 LoRA")
        return model
    
    @staticmethod
    def get_lora_parameters(model: nn.Module):
        """获取模型中的所有 LoRA 参数"""
        lora_params = []
        for module in model.modules():
            if isinstance(module, LoRALinear):
                lora_params.extend(module.get_lora_parameters())
        return lora_params
    
    @staticmethod
    def count_parameters(model: nn.Module):
        """统计模型参数"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lora_params = sum(p.numel() for p in LoRAAdapter.get_lora_parameters(model))
        
        return {
            'total': total,
            'trainable': trainable,
            'lora_only': lora_params,
            'frozen': total - trainable
        }
```

#### 2. 在 Split Model 中应用 LoRA

**修改文件**: `SplitLearnCore/src/splitlearn_core/models/gpt2/bottom.py`

```python
from ...adapters.lora import LoRAAdapter

class GPT2BottomModel(BaseBottomModel):
    """支持 LoRA 的 Bottom 模型"""
    
    def __init__(self, config: GPT2Config, end_layer: int = 2, use_lora: bool = False, lora_rank: int = 8):
        super().__init__(config, end_layer=end_layer)
        
        self.use_lora = use_lora
        
        if use_lora:
            # 应用 LoRA
            LoRAAdapter.apply_lora_to_model(
                self,
                rank=lora_rank,
                target_modules=['c_attn', 'c_fc', 'c_proj']
            )
            print(f"Bottom 模型已应用 LoRA (rank={lora_rank})")
    
    def get_lora_parameters(self):
        """获取 LoRA 参数"""
        if self.use_lora:
            return LoRAAdapter.get_lora_parameters(self)
        return []
```

#### 3. 训练客户端（LoRA 版本）

**新建文件**: `test/client/lora_training_client.py`

```python
"""
LoRA 微调客户端
只需要反向传播功能 + LoRA 适配器
"""

import torch
import torch.nn as nn
from training_config import TrainingConfig
from splitlearn_core.adapters.lora import LoRAAdapter

class LoRATrainer:
    """LoRA 微调训练器"""
    
    def __init__(self, bottom, top, trunk_client, config: TrainingConfig):
        self.bottom = bottom
        self.top = top
        self.trunk_client = trunk_client
        self.config = config
        
        # 设置为训练模式
        self.bottom.train()
        self.top.train()
        
        # 只优化 LoRA 参数
        lora_params_bottom = LoRAAdapter.get_lora_parameters(self.bottom)
        lora_params_top = LoRAAdapter.get_lora_parameters(self.top)
        
        # 创建优化器（只包含 LoRA 参数）
        self.optimizer_bottom = torch.optim.Adam(
            lora_params_bottom,
            lr=config.learning_rate
        )
        self.optimizer_top = torch.optim.Adam(
            lora_params_top,
            lr=config.learning_rate
        )
        
        # 损失函数
        self.criterion = config.get_loss_fn()
        
        # 统计信息
        bottom_stats = LoRAAdapter.count_parameters(self.bottom)
        top_stats = LoRAAdapter.count_parameters(self.top)
        
        print(f"\n参数统计:")
        print(f"Bottom - 总参数: {bottom_stats['total']/1e6:.2f}M, "
              f"可训练: {bottom_stats['trainable']/1e6:.2f}M "
              f"({bottom_stats['trainable']/bottom_stats['total']*100:.1f}%)")
        print(f"Top - 总参数: {top_stats['total']/1e6:.2f}M, "
              f"可训练: {top_stats['trainable']/1e6:.2f}M "
              f"({top_stats['trainable']/top_stats['total']*100:.1f}%)")
    
    def train_step(self, input_ids, labels):
        """训练步骤（只需要反向传播）"""
        # 清零梯度
        self.optimizer_bottom.zero_grad()
        self.optimizer_top.zero_grad()
        
        # 前向传播
        hidden_1 = self.bottom(input_ids)
        hidden_2 = self.trunk_client.compute(hidden_1)  # 需要支持梯度
        output = self.top(hidden_2)
        
        # 计算损失
        loss = self.criterion(
            output.logits.view(-1, output.logits.size(-1)),
            labels.view(-1)
        )
        
        # 反向传播
        loss.backward()
        
        # 梯度传递（只需要传递 LoRA 参数的梯度，数据量很小）
        grad_hidden_2 = hidden_2.grad
        grad_hidden_1 = self.trunk_client.backward(grad_hidden_2)
        hidden_1.backward(grad_hidden_1)
        
        # 更新参数（只更新 LoRA 参数）
        self.optimizer_bottom.step()
        self.optimizer_top.step()
        
        return {'loss': loss.item()}
```

---

## 与完整训练对比

### 实现复杂度对比

| 功能 | 完整训练 | LoRA 微调 |
|------|---------|----------|
| **反向传播** | ✅ 需要 | ✅ 需要 |
| **梯度传递** | ✅ 需要（全量梯度，几 GB） | ✅ 需要（LoRA 梯度，几 MB） |
| **参数更新** | ✅ 需要（所有参数） | ✅ 需要（仅 LoRA 参数） |
| **架构修改** | ❌ 不需要 | ✅ 需要添加 LoRA 适配器 |
| **模型冻结** | ❌ 不需要 | ✅ 需要冻结原始参数 |

### 通信开销对比

```
完整训练:
- Bottom 模型梯度: ~200 MB
- Trunk 模型梯度: ~800 MB  
- Top 模型梯度: ~200 MB
总计: ~1.2 GB/步

LoRA 微调 (rank=8):
- Bottom LoRA 梯度: ~2 MB
- Trunk LoRA 梯度: ~8 MB
- Top LoRA 梯度: ~2 MB
总计: ~12 MB/步

减少: 100 倍！
```

### 内存占用对比

```
完整训练:
- 梯度存储: 2x 模型大小 (~2.4 GB)
- 优化器状态: 2x 模型大小 (~2.4 GB)
总计: ~4.8 GB

LoRA 微调:
- LoRA 参数: ~12 MB
- LoRA 梯度: ~12 MB
- 优化器状态: ~24 MB
总计: ~48 MB

减少: 100 倍！
```

---

## 实现步骤（简化版）

### Phase 1: 添加 LoRA 适配器（1 周）

1. ✅ 实现 `LoRALinear` 类
2. ✅ 实现 `LoRAAdapter` 工具类
3. ✅ 在模型中应用 LoRA
4. ✅ 测试 LoRA 参数统计

### Phase 2: 反向传播支持（2 周）

1. ✅ 扩展通信协议（同完整训练）
2. ✅ 实现梯度传递（数据量小，实现更简单）
3. ✅ 测试梯度传递

### Phase 3: 训练循环（1 周）

1. ✅ 创建 LoRA 训练器
2. ✅ 实现训练步骤
3. ✅ 测试训练流程

**总计**: 约 4 周（比完整训练快 2-3 倍）

---

## 总结

### 回答你的问题

**Q: 只增加反向传播功能可以实现 LoRA 等微调功能吗？**

**A: ✅ 完全可以！而且更简单！**

### 关键点

1. **标准 Transformer Block 完全兼容 LoRA**
   - 不需要修改 Block 本身
   - 只需要包装线性层
   - 使用标准的前向/反向传播

2. **实现更简单**
   - 只需要反向传播 + LoRA 适配器
   - 不需要修改模型架构（只需包装）
   - 梯度传输量小（100倍减少）

3. **性能更好**
   - 通信开销小
   - 内存占用小
   - 训练速度快

### 推荐方案

**使用 LoRA 微调而不是完整训练**：
- ✅ 实现简单（4周 vs 12周）
- ✅ 通信高效（12MB vs 1.2GB）
- ✅ 参数高效（只训练 2% 参数）
- ✅ 效果接近（LoRA 通常能达到 90%+ 的效果）

### 下一步

1. 实现 LoRA 适配器（核心）
2. 添加反向传播支持（简化版）
3. 创建训练循环（LoRA 版本）

需要我帮你开始实现 LoRA 适配器吗？
