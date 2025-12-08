# 完整的三部分联合训练实现方案

## 目标

实现完整的 Split Learning 联合训练，让 Bottom、Trunk、Top 三个部分一起进行训练。

---

## 当前状态 vs 目标状态

### 当前状态（简化版本）

```
训练流程：
  ✅ Bottom 模型：完整训练（前向 + 反向 + 更新）
  ❌ Trunk 模型：只做前向传播（不训练）
  ✅ Top 模型：完整训练（前向 + 反向 + 更新）
```

### 目标状态（完整版本）

```
训练流程：
  ✅ Bottom 模型：完整训练（前向 + 反向 + 更新）
  ✅ Trunk 模型：完整训练（前向 + 反向 + 更新）
  ✅ Top 模型：完整训练（前向 + 反向 + 更新）
```

---

## 实现方案概览

需要实现以下内容：

1. **扩展 gRPC 协议** - 添加反向传播服务
2. **实现梯度传递机制** - 序列化和传输梯度
3. **服务器端支持** - 反向传播和梯度缓存
4. **客户端支持** - 完整的梯度传递
5. **完整的训练脚本** - 三部分联合训练

---

## 步骤 1: 扩展 gRPC 协议

### 1.1 修改 Protocol Buffer 定义

**文件**: `SplitLearnComm/src/splitlearn_comm/protocol/protos/compute_service.proto`

需要添加反向传播相关的消息和服务：

```protobuf
syntax = "proto3";

package splitlearn.comm;

// ComputeService - 通用计算服务
service ComputeService {
    // 执行计算（前向传播）
    rpc Compute(ComputeRequest) returns (ComputeResponse);
    
    // 反向传播（新增）
    rpc Backward(BackwardRequest) returns (BackwardResponse);
    
    // 健康检查
    rpc HealthCheck(HealthRequest) returns (HealthResponse);
    
    // 获取服务信息
    rpc GetServiceInfo(ServiceInfoRequest) returns (ServiceInfoResponse);
}

// ... 现有的消息定义 ...

// 反向传播请求（新增）
message BackwardRequest {
    // 梯度张量数据（二进制格式）
    bytes gradient_data = 1;
    
    // 梯度张量形状
    repeated int32 gradient_shape = 2;
    
    // 请求 ID（与前向传播的请求 ID 关联）
    string request_id = 3;
    
    // 是否保留梯度
    bool requires_grad = 4;
}

// 反向传播响应（新增）
message BackwardResponse {
    // 传递给前一层的梯度数据
    bytes gradient_data = 1;
    
    // 梯度张量形状
    repeated int32 gradient_shape = 2;
    
    // 是否成功
    bool success = 3;
    
    // 错误消息（如果失败）
    string error_message = 4;
}

// ... 其他现有的消息定义 ...
```

### 1.2 重新生成 Python 代码

```bash
cd SplitLearnComm/src/splitlearn_comm/protocol/protos
protoc --python_out=.. --grpc_python_out=.. compute_service.proto
```

---

## 步骤 2: 实现梯度工具

### 2.1 创建梯度序列化工具

**新建文件**: `SplitLearnComm/src/splitlearn_comm/utils/gradient_utils.py`

```python
"""
梯度序列化和反序列化工具
"""
import torch
import numpy as np
import pickle
import zlib
from typing import Optional, Tuple


def serialize_tensor(tensor: torch.Tensor, compress: bool = True) -> bytes:
    """
    序列化张量
    
    Args:
        tensor: 要序列化的张量
        compress: 是否压缩
    
    Returns:
        序列化的字节数据
    """
    if tensor is None:
        return b''
    
    # 转换为 numpy 并序列化
    tensor_np = tensor.detach().cpu().numpy()
    data = pickle.dumps(tensor_np)
    
    # 可选压缩
    if compress:
        data = zlib.compress(data)
    
    return data


def deserialize_tensor(
    data: bytes,
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: str = 'cpu',
    compressed: bool = True,
    requires_grad: bool = False
) -> Optional[torch.Tensor]:
    """
    反序列化张量
    
    Args:
        data: 序列化的字节数据
        shape: 张量形状
        dtype: 数据类型
        device: 设备
        compressed: 是否压缩
        requires_grad: 是否需要梯度
    
    Returns:
        张量，如果失败返回 None
    """
    if not data or len(data) == 0:
        return None
    
    try:
        # 解压缩
        if compressed:
            data = zlib.decompress(data)
        
        # 反序列化
        tensor_np = pickle.loads(data)
        
        # 转换为 PyTorch 张量
        tensor = torch.from_numpy(tensor_np).to(dtype=dtype, device=device)
        
        # 确保形状匹配
        if tuple(tensor.shape) != tuple(shape):
            tensor = tensor.reshape(shape)
        
        # 设置梯度需求
        if requires_grad:
            tensor.requires_grad_(True)
        
        return tensor
    except Exception as e:
        print(f"反序列化失败: {e}")
        return None
```

---

## 步骤 3: 扩展服务器端代码

### 3.1 修改服务器 Servicer

**文件**: `SplitLearnComm/src/splitlearn_comm/server/servicer.py`

需要添加反向传播方法和梯度缓存：

```python
from typing import Dict
import torch
from .utils.gradient_utils import serialize_tensor, deserialize_tensor

class ComputeServicer(compute_service_pb2_grpc.ComputeServiceServicer):
    """计算服务实现"""
    
    def __init__(self, compute_func):
        self.compute_func = compute_func
        self._forward_cache: Dict[str, Dict] = {}  # 缓存前向传播状态
        
    def Compute(self, request, context):
        """前向传播（现有方法，需要修改以支持缓存）"""
        # ... 现有的前向传播代码 ...
        
        # 如果需要梯度，缓存输入
        request_id = request.request_id if request.HasField('request_id') else None
        if request_id is None:
            import uuid
            request_id = str(uuid.uuid4())
        
        # 缓存输入（用于反向传播）
        # ... 缓存逻辑 ...
        
        return response
    
    def Backward(self, request, context):
        """反向传播（新增方法）"""
        try:
            # 获取请求 ID
            request_id = request.request_id
            
            # 检查缓存
            if request_id not in self._forward_cache:
                return compute_service_pb2.BackwardResponse(
                    success=False,
                    error_message=f"找不到请求 ID {request_id} 的前向传播缓存"
                )
            
            cache = self._forward_cache[request_id]
            
            # 反序列化梯度
            gradient = deserialize_tensor(
                request.gradient_data,
                tuple(request.gradient_shape),
                device=cache['device'],
                requires_grad=True
            )
            
            if gradient is None:
                return compute_service_pb2.BackwardResponse(
                    success=False,
                    error_message="梯度反序列化失败"
                )
            
            # 执行反向传播
            cache['output'].backward(gradient=gradient)
            
            # 获取输入梯度
            input_gradient = cache['input'].grad
            
            if input_gradient is None:
                return compute_service_pb2.BackwardResponse(
                    success=False,
                    error_message="输入梯度计算失败"
                )
            
            # 序列化输入梯度
            grad_data = serialize_tensor(input_gradient)
            grad_shape = list(input_gradient.shape)
            
            # 清理缓存
            del self._forward_cache[request_id]
            
            return compute_service_pb2.BackwardResponse(
                gradient_data=grad_data,
                gradient_shape=grad_shape,
                success=True
            )
            
        except Exception as e:
            return compute_service_pb2.BackwardResponse(
                success=False,
                error_message=str(e)
            )
```

---

## 步骤 4: 扩展客户端代码

### 4.1 扩展 TrunkClient

**文件**: `test/client/trunk_client_with_training.py`（新建）

```python
"""
支持完整训练的 Trunk 客户端
"""
import torch
from typing import Optional, Dict
from splitlearn_comm.client.grpc_client import GrpcClient
from splitlearn_comm.core.tensor_codec import TensorCodec
from splitlearn_comm.utils.gradient_utils import serialize_tensor, deserialize_tensor


class TrunkClientWithTraining:
    """支持完整训练的 Trunk 客户端"""
    
    def __init__(self, server_address: str = "localhost:50052"):
        self.client = GrpcClient(server_address)
        self.codec = TensorCodec()
        self._forward_cache: Dict[str, Dict] = {}
        
    def compute(self, hidden_states: torch.Tensor, request_id: Optional[str] = None) -> torch.Tensor:
        """
        前向传播（支持梯度）
        
        Args:
            hidden_states: 输入隐藏状态
            request_id: 请求 ID（用于关联前向和反向）
        
        Returns:
            输出隐藏状态
        """
        # 生成请求 ID
        if request_id is None:
            import uuid
            request_id = str(uuid.uuid4())
        
        # 如果需要梯度，缓存输入
        if hidden_states.requires_grad:
            self._forward_cache[request_id] = {
                'input': hidden_states.detach().clone(),
                'input_shape': hidden_states.shape,
                'device': hidden_states.device,
                'dtype': hidden_states.dtype
            }
        
        # 调用服务器
        output_data, output_shape = self.codec.encode(hidden_states.detach())
        
        response = self.client.compute(
            data=output_data,
            shape=list(output_shape),
            request_id=int(hash(request_id) % (2**31))  # 转换为 int32
        )
        
        # 解码输出
        output = self.codec.decode(response.data, tuple(response.shape))
        output = output.to(device=hidden_states.device)
        
        # 如果需要梯度，创建新的可导张量
        if hidden_states.requires_grad:
            output = output.requires_grad_(True)
            self._forward_cache[request_id]['output'] = output
        
        return output
    
    def backward(self, gradient: torch.Tensor, request_id: str) -> torch.Tensor:
        """
        反向传播
        
        Args:
            gradient: 来自后一层的梯度
            request_id: 对应的前向传播请求 ID
        
        Returns:
            传递给前一层的梯度
        """
        if request_id not in self._forward_cache:
            raise ValueError(f"找不到请求 ID {request_id} 的前向传播缓存")
        
        cache = self._forward_cache[request_id]
        
        # 序列化梯度
        grad_data = serialize_tensor(gradient)
        grad_shape = list(gradient.shape)
        
        # 调用服务器的反向传播
        backward_request = compute_service_pb2.BackwardRequest(
            gradient_data=grad_data,
            gradient_shape=grad_shape,
            request_id=request_id,
            requires_grad=True
        )
        
        response = self.client.Backward(backward_request)
        
        if not response.success:
            raise RuntimeError(f"反向传播失败: {response.error_message}")
        
        # 反序列化返回的梯度
        input_gradient = deserialize_tensor(
            response.gradient_data,
            tuple(response.gradient_shape),
            dtype=cache['dtype'],
            device=cache['device'],
            requires_grad=True
        )
        
        # 清理缓存
        del self._forward_cache[request_id]
        
        return input_gradient
```

---

## 步骤 5: 完整的训练脚本

### 5.1 创建完整的联合训练脚本

**新建文件**: `test/client/train_full_joint.py`

```python
"""
完整的三部分联合训练脚本
Bottom、Trunk、Top 三个部分一起训练
"""
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from splitlearn_core.models.gpt2.bottom import GPT2BottomModel
from splitlearn_core.models.gpt2.top import GPT2TopModel
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model, TaskType

# 导入支持训练的客户端
from trunk_client_with_training import TrunkClientWithTraining


def train_full_joint(
    bottom_peft,
    top_peft,
    trunk_client,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    optimizer_bottom,
    optimizer_trunk,
    optimizer_top,
    criterion
) -> dict:
    """执行完整的联合训练步骤"""
    
    # 生成请求 ID
    import uuid
    request_id = str(uuid.uuid4())
    
    # ========== 前向传播 ==========
    
    # Bottom 模型（本地，保留梯度）
    hidden_1 = bottom_peft.base_model(input_ids)
    
    # Trunk 模型（远程服务器，保留梯度）
    hidden_2 = trunk_client.compute(hidden_1, request_id=request_id)
    
    # Top 模型（本地，保留梯度）
    output = top_peft.base_model(hidden_2)
    logits = output.logits if hasattr(output, 'logits') else output
    
    # 计算损失
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = criterion(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    # ========== 反向传播 ==========
    
    # 反向传播（Top 模型）
    loss.backward()
    
    # 获取 hidden_2 的梯度
    grad_hidden_2 = hidden_2.grad
    
    # Trunk 模型反向传播（远程服务器）
    grad_hidden_1 = trunk_client.backward(grad_hidden_2, request_id=request_id)
    
    # Bottom 模型反向传播（本地）
    hidden_1.backward(gradient=grad_hidden_1)
    
    # ========== 参数更新 ==========
    
    # 更新所有三个模型的参数
    optimizer_bottom.step()
    optimizer_trunk.step()  # 需要在服务器端实现
    optimizer_top.step()
    
    return {
        'loss': loss.item(),
        'hidden_1_norm': hidden_1.norm().item(),
        'hidden_2_norm': hidden_2.norm().item()
    }
```

---

## 实现检查清单

### 需要实现的功能

- [ ] **步骤 1**: 扩展 gRPC 协议
  - [ ] 添加 `BackwardRequest` 消息
  - [ ] 添加 `BackwardResponse` 消息
  - [ ] 添加 `Backward` RPC 服务
  - [ ] 重新生成 Python 代码

- [ ] **步骤 2**: 实现梯度工具
  - [ ] 创建 `gradient_utils.py`
  - [ ] 实现张量序列化/反序列化
  - [ ] 支持压缩选项

- [ ] **步骤 3**: 扩展服务器端
  - [ ] 添加梯度缓存机制
  - [ ] 实现 `Backward` 方法
  - [ ] 支持 Trunk 模型的优化器

- [ ] **步骤 4**: 扩展客户端
  - [ ] 创建 `TrunkClientWithTraining`
  - [ ] 实现前向传播缓存
  - [ ] 实现反向传播调用

- [ ] **步骤 5**: 完整训练脚本
  - [ ] 创建完整的联合训练脚本
  - [ ] 实现三部分参数更新
  - [ ] 测试完整训练流程

---

## 关键挑战和解决方案

### 挑战 1: 梯度缓存

**问题**: 服务器需要缓存前向传播的状态，以便反向传播时使用。

**解决方案**: 使用请求 ID 关联前向和反向传播，缓存中间状态。

### 挑战 2: 梯度传递

**问题**: 梯度需要通过网络传输，可能有性能开销。

**解决方案**: 使用压缩和高效的序列化方法。

### 挑战 3: 服务器端优化器

**问题**: Trunk 模型的优化器需要在服务器端管理。

**解决方案**: 在服务器端为每个模型创建优化器，通过 RPC 调用更新参数。

---

## 下一步行动

1. 开始实现步骤 1（扩展 gRPC 协议）
2. 逐步实现其他步骤
3. 测试每个步骤的功能
4. 整合完整的训练流程

你想要我开始实现哪个步骤？



