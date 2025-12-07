# Split Learning 训练功能完整实现指南

## 目录

1. [概述](#概述)
2. [Split Learning 训练原理](#split-learning-训练原理)
3. [实现架构设计](#实现架构设计)
4. [详细实现步骤](#详细实现步骤)
5. [代码实现示例](#代码实现示例)
6. [测试和验证](#测试和验证)
7. [可能的问题和解决方案](#可能的问题和解决方案)

---

## 概述

本文档详细说明如何在当前的 Split Learning 推理系统基础上添加完整的训练功能。

### 当前系统状态
- ✅ 支持分布式推理（Bottom → Trunk → Top）
- ✅ gRPC 通信协议（前向传播）
- ✅ 模型加载和管理
- ❌ 不支持梯度计算
- ❌ 不支持参数更新
- ❌ 不支持梯度传递

### 目标
- ✅ 支持完整的分布式训练
- ✅ 实现梯度传递机制
- ✅ 支持客户端和服务器的参数同步更新

---

## Split Learning 训练原理

### 传统训练流程

```
输入数据 → 完整模型 → 输出 → 损失函数
                ↑               ↓
                反向传播 ← 梯度计算
                ↓
            参数更新
```

### Split Learning 训练流程

```
[客户端] 输入 → Bottom模型 → 隐藏状态(h1)
                              ↓ (网络传输)
[服务器]                   隐藏状态(h1) → Trunk模型 → 隐藏状态(h2)
                                                        ↓ (网络传输)
[客户端]                                          隐藏状态(h2) → Top模型 → 输出
                                                                           ↓
                                                                       损失函数
                                                                           ↓
                                                                       反向传播
                                                                      ↙   ↓   ↘
                                                              梯度h2  梯度h1  梯度input
                                                                 ↓      ↓       ↓
                                                              Top    Trunk   Bottom
                                                            更新    更新    更新
```

### 关键挑战

1. **梯度传递**：Top 模型的梯度需要通过网络传递回 Trunk 和 Bottom
2. **状态同步**：客户端和服务器需要同步训练状态
3. **通信开销**：梯度传输会增加网络通信量
4. **错误恢复**：需要处理网络中断、服务器故障等情况

---

## 实现架构设计

### 1. 通信协议扩展

需要扩展 gRPC 服务以支持梯度传递：

```protobuf
// 现有的 compute_service.proto 扩展

service ComputeService {
    // 现有的前向传播
    rpc Compute(ComputeRequest) returns (ComputeResponse);
    
    // 新增：反向传播 - 传递梯度
    rpc Backward(BackwardRequest) returns (BackwardResponse);
    
    // 新增：获取服务器端梯度
    rpc GetGradients(GradientRequest) returns (GradientResponse);
    
    // 新增：同步训练状态
    rpc SyncTrainingState(TrainingStateRequest) returns (TrainingStateResponse);
}
```

### 2. 客户端架构

```
TrainingClient
├── Bottom Model (可训练)
├── Top Model (可训练)
├── Optimizer (Bottom)
├── Optimizer (Top)
├── Loss Function
├── Training Loop
└── Gradient Handler (处理梯度传递)
```

### 3. 服务器架构

```
TrainingServer
├── Trunk Model (可训练)
├── Optimizer (Trunk)
├── Gradient Store (存储前向传播的中间状态)
├── Backward Handler (处理反向传播)
└── State Sync (同步训练状态)
```

---

## 详细实现步骤

### 阶段 1: 准备工作和基础修改

#### 1.1 修改模型模式管理

**文件**: `test/client/interactive_client.py`, `test/client/test_client.py`

**当前代码**:
```python
bottom.eval()
top.eval()

with torch.no_grad():
    hidden_1 = bottom(input_ids)
```

**需要修改为**:
```python
class TrainingConfig:
    def __init__(self, mode='inference'):
        self.mode = mode  # 'inference' or 'training'
        
    def set_model_mode(self, models):
        if self.mode == 'training':
            for model in models:
                model.train()
        else:
            for model in models:
                model.eval()

# 使用示例
config = TrainingConfig(mode='training')
config.set_model_mode([bottom, top])

# 根据模式决定是否使用 no_grad
if config.mode == 'training':
    hidden_1 = bottom(input_ids)  # 保留梯度
else:
    with torch.no_grad():
        hidden_1 = bottom(input_ids)  # 不保留梯度
```

#### 1.2 创建训练配置类

**新建文件**: `test/client/training_config.py`

```python
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础配置
    mode: str = 'training'  # 'training' or 'inference'
    device: str = 'cpu'
    
    # 优化器配置
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    optimizer_type: str = 'adam'  # 'adam', 'sgd', etc.
    
    # 训练配置
    batch_size: int = 4
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0  # 梯度裁剪
    
    # 损失函数配置
    loss_type: str = 'cross_entropy'
    label_smoothing: float = 0.0
    
    # 学习率调度
    use_scheduler: bool = False
    scheduler_type: Optional[str] = None
    warmup_steps: int = 0
    
    # 日志和保存
    log_interval: int = 10
    save_interval: int = 100
    save_dir: str = './checkpoints'
    
    def get_optimizer(self, model, model_name=''):
        """创建优化器"""
        if self.optimizer_type == 'adam':
            return torch.optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'sgd':
            return torch.optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
    
    def get_loss_fn(self):
        """创建损失函数"""
        if self.loss_type == 'cross_entropy':
            return torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
```

### 阶段 2: 扩展 gRPC 通信协议

#### 2.1 扩展 Protocol Buffer 定义

**文件**: `SplitLearnComm/src/splitlearn_comm/protocol/compute_service.proto`

**需要添加**:

```protobuf
// 梯度传递请求
message BackwardRequest {
    bytes gradient_data = 1;  // 序列化的梯度张量
    TensorShape gradient_shape = 2;
    string request_id = 3;  // 与前向传播的请求ID关联
    bool requires_grad = 4;
}

// 梯度传递响应
message BackwardResponse {
    bytes gradient_data = 1;  // 传递给前一层模型的梯度
    TensorShape gradient_shape = 2;
    bool success = 3;
    string error_message = 4;
}

// 获取梯度请求
message GradientRequest {
    string request_id = 1;  // 获取特定请求的梯度
}

// 梯度响应
message GradientResponse {
    bytes gradient_data = 1;
    TensorShape gradient_shape = 2;
    bool has_gradient = 3;
}

// 训练状态请求
message TrainingStateRequest {
    int32 epoch = 1;
    int32 step = 2;
    float learning_rate = 3;
}

// 训练状态响应
message TrainingStateResponse {
    bool success = 1;
    string error_message = 2;
}
```

#### 2.2 实现梯度序列化工具

**新建文件**: `SplitLearnComm/src/splitlearn_comm/utils/gradient_utils.py`

```python
import torch
import numpy as np
from typing import Optional, Tuple
import pickle
import zlib

def serialize_gradient(gradient: torch.Tensor, compress: bool = True) -> bytes:
    """
    序列化梯度张量
    
    Args:
        gradient: 梯度张量
        compress: 是否压缩
    
    Returns:
        序列化的字节数据
    """
    if gradient is None:
        return b''
    
    # 转换为 numpy 并序列化
    grad_np = gradient.detach().cpu().numpy()
    data = pickle.dumps(grad_np)
    
    # 可选压缩
    if compress:
        data = zlib.compress(data)
    
    return data

def deserialize_gradient(
    data: bytes, 
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: str = 'cpu',
    compressed: bool = True
) -> Optional[torch.Tensor]:
    """
    反序列化梯度张量
    
    Args:
        data: 序列化的字节数据
        shape: 张量形状
        dtype: 数据类型
        device: 设备
        compressed: 是否压缩
    
    Returns:
        梯度张量，如果失败返回 None
    """
    if not data or len(data) == 0:
        return None
    
    try:
        # 解压缩
        if compressed:
            data = zlib.decompress(data)
        
        # 反序列化
        grad_np = pickle.loads(data)
        
        # 转换为 torch 张量
        gradient = torch.from_numpy(grad_np).to(dtype=dtype, device=device)
        
        # 确保形状正确
        if gradient.shape != shape:
            gradient = gradient.view(shape)
        
        return gradient
    except Exception as e:
        print(f"反序列化梯度失败: {e}")
        return None

def get_tensor_info(tensor: torch.Tensor) -> dict:
    """获取张量信息用于传输"""
    return {
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'numel': tensor.numel(),
        'size_bytes': tensor.numel() * tensor.element_size()
    }
```

#### 2.3 扩展客户端通信接口

**文件**: `SplitLearnComm/src/splitlearn_comm/quickstart.py` 或新建 `training_client.py`

```python
from typing import Optional
import torch
from .utils.gradient_utils import serialize_gradient, deserialize_gradient, get_tensor_info

class TrainingClient:
    """支持训练的客户端"""
    
    def __init__(self, server_address: str):
        self.client = GRPCComputeClient(server_address)
        self._forward_cache = {}  # 缓存前向传播的中间状态
        
    def forward(self, hidden_states: torch.Tensor, request_id: str = None) -> torch.Tensor:
        """
        前向传播（支持梯度）
        
        Args:
            hidden_states: 隐藏状态
            request_id: 请求ID，用于关联前向和反向传播
        
        Returns:
            输出隐藏状态
        """
        if request_id is None:
            import uuid
            request_id = str(uuid.uuid4())
        
        # 如果需要梯度，保存输入状态
        if hidden_states.requires_grad:
            self._forward_cache[request_id] = {
                'input': hidden_states.detach().clone(),
                'input_shape': hidden_states.shape
            }
        
        # 调用服务器
        output = self.client.compute(hidden_states)
        
        # 如果需要梯度，创建新的可导张量
        if hidden_states.requires_grad:
            output = output.requires_grad_(True)
            self._forward_cache[request_id]['output'] = output
        
        return output
    
    def backward(
        self, 
        gradient: torch.Tensor, 
        request_id: str
    ) -> Optional[torch.Tensor]:
        """
        反向传播
        
        Args:
            gradient: 来自后一层的梯度
            request_id: 对应的前向传播请求ID
        
        Returns:
            传递给前一层的梯度
        """
        if request_id not in self._forward_cache:
            raise ValueError(f"找不到请求ID {request_id} 的前向传播缓存")
        
        # 序列化梯度
        grad_data = serialize_gradient(gradient)
        grad_shape = list(gradient.shape)
        
        # 调用服务器的反向传播
        response = self.client.backward(
            gradient_data=grad_data,
            gradient_shape=grad_shape,
            request_id=request_id
        )
        
        if not response.success:
            raise RuntimeError(f"反向传播失败: {response.error_message}")
        
        # 反序列化返回的梯度
        input_info = self._forward_cache[request_id]
        input_gradient = deserialize_gradient(
            response.gradient_data,
            tuple(response.gradient_shape),
            dtype=input_info['input'].dtype,
            device=input_info['input'].device
        )
        
        return input_gradient
```

### 阶段 3: 服务器端训练支持

#### 3.1 扩展服务器接口

**文件**: `SplitLearnManager/src/splitlearn_manager/server/managed_server.py`

需要添加：

```python
class ManagedTrainingServer(ManagedServer):
    """支持训练的服务器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._forward_cache = {}  # 缓存前向传播状态
        self._optimizers = {}  # 每个模型的优化器
        
    def setup_training(self, model_name: str, optimizer_config: dict):
        """为模型设置训练"""
        if model_name not in self._models:
            raise ValueError(f"模型 {model_name} 不存在")
        
        model = self._models[model_name]
        model.train()  # 设置为训练模式
        
        # 创建优化器
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config.get('learning_rate', 1e-4)
        )
        self._optimizers[model_name] = optimizer
    
    def compute_with_grad(self, hidden_states: torch.Tensor, request_id: str):
        """
        支持梯度的前向传播
        
        Args:
            hidden_states: 输入隐藏状态
            request_id: 请求ID
        
        Returns:
            输出隐藏状态
        """
        # 保存输入用于反向传播
        if hidden_states.requires_grad:
            self._forward_cache[request_id] = {
                'input': hidden_states.detach().clone(),
                'model_name': self._current_model_name
            }
        
        # 前向传播
        model = self._models[self._current_model_name]
        output = model(hidden_states)
        
        return output
    
    def backward(self, gradient: torch.Tensor, request_id: str):
        """
        处理反向传播
        
        Args:
            gradient: 来自后一层的梯度
            request_id: 请求ID
        
        Returns:
            传递给前一层的梯度
        """
        if request_id not in self._forward_cache:
            raise ValueError(f"找不到请求 {request_id} 的缓存")
        
        cache = self._forward_cache[request_id]
        model = self._models[cache['model_name']]
        input_tensor = cache['input'].requires_grad_(True)
        
        # 重新进行前向传播以计算梯度
        output = model(input_tensor)
        output.backward(gradient=gradient)
        
        # 获取输入梯度
        input_gradient = input_tensor.grad
        
        # 更新模型参数
        if cache['model_name'] in self._optimizers:
            optimizer = self._optimizers[cache['model_name']]
            optimizer.step()
            optimizer.zero_grad()
        
        # 清理缓存
        del self._forward_cache[request_id]
        
        return input_gradient
```

### 阶段 4: 完整训练循环实现

#### 4.1 创建训练客户端

**新建文件**: `test/client/training_client.py`

```python
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from pathlib import Path
import json
import time

from splitlearn_core.models.gpt2 import GPT2BottomModel, GPT2TopModel
from splitlearn_comm.training_client import TrainingClient
from training_config import TrainingConfig

class SplitLearningTrainer:
    """Split Learning 训练器"""
    
    def __init__(
        self,
        bottom_model: GPT2BottomModel,
        top_model: GPT2TopModel,
        trunk_client: TrainingClient,
        config: TrainingConfig
    ):
        self.bottom = bottom_model
        self.top = top_model
        self.trunk_client = trunk_client
        self.config = config
        
        # 设置训练模式
        if config.mode == 'training':
            self.bottom.train()
            self.top.train()
        
        # 创建优化器
        self.optimizer_bottom = config.get_optimizer(self.bottom, 'bottom')
        self.optimizer_top = config.get_optimizer(self.top, 'top')
        
        # 创建损失函数
        self.criterion = config.get_loss_fn()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        
    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        执行一个训练步骤
        
        Args:
            input_ids: 输入token IDs [batch, seq_len]
            labels: 标签 [batch, seq_len]
        
        Returns:
            包含损失和指标的字典
        """
        # 清零梯度
        self.optimizer_bottom.zero_grad()
        self.optimizer_top.zero_grad()
        
        # ========== 前向传播 ==========
        
        # Bottom 模型
        hidden_1 = self.bottom(input_ids)  # [batch, seq_len, hidden_size]
        
        # Trunk 模型（远程）
        import uuid
        request_id = str(uuid.uuid4())
        hidden_2 = self.trunk_client.forward(hidden_1, request_id=request_id)
        
        # Top 模型
        output = self.top(hidden_2)
        logits = output.logits  # [batch, seq_len, vocab_size]
        
        # ========== 计算损失 ==========
        
        # 准备标签：将 logits 和 labels 展平
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # ========== 反向传播 ==========
        
        loss.backward()
        
        # 梯度传递
        # 1. Top 模型已经计算好梯度（自动）
        # 2. 获取 Top 输出的梯度
        grad_hidden_2 = hidden_2.grad
        
        # 3. 传递梯度到 Trunk
        grad_hidden_1 = self.trunk_client.backward(grad_hidden_2, request_id)
        
        # 4. 传递梯度到 Bottom
        hidden_1.backward(grad_hidden_1)
        
        # ========== 梯度裁剪 ==========
        
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.bottom.parameters(),
                self.config.max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.top.parameters(),
                self.config.max_grad_norm
            )
        
        # ========== 参数更新 ==========
        
        self.optimizer_bottom.step()
        self.optimizer_top.step()
        
        # ========== 返回指标 ==========
        
        self.global_step += 1
        
        return {
            'loss': loss.item(),
            'learning_rate': self.optimizer_bottom.param_groups[0]['lr']
        }
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.bottom.train()
        self.top.train()
        
        total_loss = 0.0
        num_steps = 0
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            # 执行训练步骤
            metrics = self.train_step(input_ids, labels)
            total_loss += metrics['loss']
            num_steps += 1
            
            # 日志输出
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = total_loss / num_steps
                print(
                    f"Epoch {self.current_epoch}, Step {batch_idx + 1}/{len(dataloader)}, "
                    f"Loss: {avg_loss:.4f}, LR: {metrics['learning_rate']:.2e}"
                )
            
            # 保存检查点
            if (self.global_step + 1) % self.config.save_interval == 0:
                self.save_checkpoint(self.global_step)
        
        return total_loss / num_steps if num_steps > 0 else 0.0
    
    def save_checkpoint(self, step: int):
        """保存检查点"""
        checkpoint_dir = Path(self.config.save_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'step': step,
            'epoch': self.current_epoch,
            'bottom_state_dict': self.bottom.state_dict(),
            'top_state_dict': self.top.state_dict(),
            'optimizer_bottom_state': self.optimizer_bottom.state_dict(),
            'optimizer_top_state': self.optimizer_top.state_dict(),
            'config': self.config.__dict__
        }
        
        checkpoint_path = checkpoint_dir / f'checkpoint_step_{step}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")
```

### 阶段 5: 数据加载和完整训练脚本

#### 5.1 创建数据加载器

**新建文件**: `test/data/dataset.py`

```python
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch

class TextDataset(Dataset):
    """文本数据集"""
    
    def __init__(self, texts: list, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize 所有文本
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.encodings['input_ids'][idx].clone()  # 用于语言建模
        }

def create_dataloader(texts: list, tokenizer, batch_size: int = 4, **kwargs):
    """创建数据加载器"""
    dataset = TextDataset(texts, tokenizer, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

#### 5.2 完整训练脚本

**新建文件**: `test/client/train.py`

```python
#!/usr/bin/env python3
"""
Split Learning 完整训练脚本
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from transformers import AutoTokenizer, AutoConfig
from splitlearn_core.models.gpt2 import GPT2BottomModel, GPT2TopModel
from splitlearn_comm.training_client import TrainingClient
from training_config import TrainingConfig
from training_client import SplitLearningTrainer
from data.dataset import create_dataloader
import torch
import json

def load_models(config_path: Path, device='cpu'):
    """加载模型"""
    models_dir = config_path.parent.parent / "models"
    
    # 加载元数据
    with open(models_dir / "bottom" / "gpt2_2-10_bottom_metadata.json") as f:
        bottom_metadata = json.load(f)
    with open(models_dir / "top" / "gpt2_2-10_top_metadata.json") as f:
        top_metadata = json.load(f)
    
    # 加载配置
    model_config = AutoConfig.from_pretrained("gpt2")
    
    # 加载模型
    bottom = GPT2BottomModel(model_config, end_layer=bottom_metadata['end_layer'])
    bottom.load_state_dict(
        torch.load(models_dir / "bottom" / "gpt2_2-10_bottom.pt", map_location=device)
    )
    
    top = GPT2TopModel(model_config, start_layer=top_metadata['start_layer'])
    top.load_state_dict(
        torch.load(models_dir / "top" / "gpt2_2-10_top.pt", map_location=device)
    )
    
    return bottom, top

def main():
    parser = argparse.ArgumentParser(description='Split Learning 训练')
    parser.add_argument('--trunk-server', type=str, default='localhost:50052',
                       help='Trunk 服务器地址')
    parser.add_argument('--data-file', type=str, required=True,
                       help='训练数据文件（每行一个文本）')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    parser.add_argument('--log-interval', type=int, default=10)
    
    args = parser.parse_args()
    
    # 创建训练配置
    config = TrainingConfig(
        mode='training',
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir,
        log_interval=args.log_interval
    )
    
    # 加载模型
    print("加载模型...")
    bottom, top = load_models(Path(__file__))
    
    # 连接到服务器
    print(f"连接到 Trunk 服务器: {args.trunk_server}")
    trunk_client = TrainingClient(args.trunk_server)
    
    # 创建训练器
    trainer = SplitLearningTrainer(bottom, top, trunk_client, config)
    
    # 加载数据
    print(f"加载训练数据: {args.data_file}")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    with open(args.data_file, 'r') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    dataloader = create_dataloader(
        texts,
        tokenizer,
        batch_size=args.batch_size,
        max_length=512
    )
    
    # 开始训练
    print(f"\n开始训练: {args.epochs} epochs, {len(dataloader)} steps/epoch")
    print("=" * 70)
    
    for epoch in range(args.epochs):
        trainer.current_epoch = epoch
        avg_loss = trainer.train_epoch(dataloader)
        
        print(f"\nEpoch {epoch + 1}/{args.epochs} 完成, 平均损失: {avg_loss:.4f}")
        print("-" * 70)
        
        # 每个 epoch 结束后保存
        trainer.save_checkpoint(trainer.global_step)
    
    print("\n训练完成！")

if __name__ == '__main__':
    main()
```

---

## 代码实现示例

### 最小可运行示例

创建一个简化的示例，展示核心训练流程：

**新建文件**: `test/examples/minimal_training_example.py`

```python
"""
最小化的训练示例 - 展示核心概念
"""

import torch
import torch.nn as nn

# 模拟模型
class SimpleBottom(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20)
    
    def forward(self, x):
        return torch.relu(self.linear(x))

class SimpleTop(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(20, 5)
    
    def forward(self, x):
        return self.linear(x)

def minimal_training_example():
    """最小化训练示例"""
    
    # 创建模型
    bottom = SimpleBottom()
    top = SimpleTop()
    
    # 设置为训练模式
    bottom.train()
    top.train()
    
    # 创建优化器
    optimizer_bottom = torch.optim.Adam(bottom.parameters(), lr=0.001)
    optimizer_top = torch.optim.Adam(top.parameters(), lr=0.001)
    
    # 损失函数
    criterion = nn.MSELoss()
    
    # 模拟数据
    input_data = torch.randn(4, 10)  # batch_size=4
    target = torch.randn(4, 5)
    
    # ========== 前向传播 ==========
    optimizer_bottom.zero_grad()
    optimizer_top.zero_grad()
    
    hidden = bottom(input_data)  # 保留梯度
    # 在这里，hidden 会被传输到服务器（Trunk），然后返回
    # 为了简化，我们假设 hidden 直接传到 Top
    output = top(hidden)
    
    # ========== 计算损失 ==========
    loss = criterion(output, target)
    print(f"损失: {loss.item():.4f}")
    
    # ========== 反向传播 ==========
    loss.backward()
    
    # 梯度已经自动计算
    # 如果 Trunk 在远程，需要：
    # 1. 获取 hidden 的梯度: hidden.grad
    # 2. 传递给服务器: server.backward(hidden.grad)
    # 3. 获取 input_data 的梯度: server 返回的梯度
    
    # ========== 更新参数 ==========
    optimizer_bottom.step()
    optimizer_top.step()
    
    print("训练步骤完成！")
    print(f"Bottom 参数变化: {sum(p.abs().sum() for p in bottom.parameters())}")
    print(f"Top 参数变化: {sum(p.abs().sum() for p in top.parameters())}")

if __name__ == '__main__':
    minimal_training_example()
```

---

## 测试和验证

### 1. 单元测试

**新建文件**: `test/tests/test_training.py`

```python
import unittest
import torch
from test.client.training_client import SplitLearningTrainer

class TestTraining(unittest.TestCase):
    
    def test_gradient_flow(self):
        """测试梯度流"""
        # 创建简单的模型
        bottom = SimpleBottom()
        top = SimpleTop()
        
        input_data = torch.randn(2, 10, requires_grad=True)
        target = torch.randn(2, 5)
        
        hidden = bottom(input_data)
        output = top(hidden)
        
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # 验证梯度存在
        self.assertIsNotNone(input_data.grad)
        self.assertTrue(torch.any(input_data.grad != 0))
    
    def test_parameter_update(self):
        """测试参数更新"""
        model = SimpleBottom()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        old_params = [p.clone() for p in model.parameters()]
        
        # 执行一步训练
        optimizer.zero_grad()
        loss = model(torch.randn(2, 10)).sum()
        loss.backward()
        optimizer.step()
        
        # 验证参数已更新
        new_params = list(model.parameters())
        for old, new in zip(old_params, new_params):
            self.assertTrue(torch.any(old != new))

if __name__ == '__main__':
    unittest.main()
```

### 2. 集成测试

测试完整训练流程：

```python
def test_end_to_end_training():
    """端到端训练测试"""
    # 1. 加载模型
    # 2. 连接到服务器
    # 3. 运行几个训练步骤
    # 4. 验证损失下降
    # 5. 验证参数更新
    pass
```

---

## 可能的问题和解决方案

### 问题 1: 内存不足

**原因**: 训练模式需要保存中间状态，内存消耗更大

**解决方案**:
- 使用梯度累积减少批次大小
- 使用混合精度训练（FP16）
- 定期清理缓存

```python
# 梯度累积示例
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 问题 2: 网络延迟影响训练速度

**原因**: 梯度传递需要网络通信

**解决方案**:
- 使用异步通信
- 批量梯度传输
- 梯度压缩

### 问题 3: 梯度同步问题

**原因**: 客户端和服务器梯度更新不同步

**解决方案**:
- 实现梯度同步协议
- 使用梯度聚合
- 定期同步模型参数

### 问题 4: 训练不稳定

**原因**: 分布式训练中的数值不稳定

**解决方案**:
- 梯度裁剪
- 学习率调度
- 梯度归一化

---

## 实施路线图

### 阶段 1: 基础准备（1-2周）
- [ ] 创建训练配置类
- [ ] 修改模型模式管理
- [ ] 本地训练测试（不涉及网络）

### 阶段 2: 通信协议扩展（2-3周）
- [ ] 扩展 Protocol Buffer 定义
- [ ] 实现梯度序列化/反序列化
- [ ] 实现客户端梯度传递接口
- [ ] 实现服务器梯度处理接口

### 阶段 3: 训练循环实现（2-3周）
- [ ] 实现基础训练循环
- [ ] 实现数据加载
- [ ] 实现检查点保存/加载
- [ ] 实现训练监控

### 阶段 4: 测试和优化（2-3周）
- [ ] 单元测试
- [ ] 集成测试
- [ ] 性能优化
- [ ] 错误处理

### 阶段 5: 文档和部署（1周）
- [ ] 完善文档
- [ ] 使用示例
- [ ] 部署指南

**总计**: 约 8-12 周

---

## 总结

实现完整的 Split Learning 训练功能需要：

1. **理论基础**: 理解分布式训练的梯度传递机制
2. **架构设计**: 扩展通信协议和系统架构
3. **代码实现**: 实现梯度传递、训练循环等核心功能
4. **测试验证**: 确保训练的正确性和稳定性
5. **优化调试**: 解决性能和稳定性问题

这是一个**大型项目**，建议分阶段实施，每个阶段都进行充分测试。
