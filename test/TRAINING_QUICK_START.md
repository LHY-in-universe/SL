# Split Learning 训练功能实现 - 快速参考

## 🎯 核心工作清单

### 1. 基础修改（必须完成）

#### 1.1 移除推理模式限制
```python
# ❌ 当前代码
bottom.eval()
with torch.no_grad():
    output = model(input)

# ✅ 修改为
bottom.train()  # 或根据配置动态设置
output = model(input)  # 移除 no_grad
```

**需要修改的文件**:
- `test/client/test_client.py` (第87, 94, 141, 180行)
- `test/client/interactive_client.py` (第77, 84, 138行)

#### 1.2 添加训练配置类
创建 `test/client/training_config.py`:
- 优化器配置
- 学习率配置
- 损失函数配置
- 训练参数配置

---

### 2. 通信协议扩展（核心工作）

#### 2.1 扩展 Protocol Buffer

**文件**: `SplitLearnComm/src/splitlearn_comm/protocol/compute_service.proto`

**需要添加的消息类型**:
```protobuf
message BackwardRequest {
    bytes gradient_data = 1;
    TensorShape gradient_shape = 2;
    string request_id = 3;
}

message BackwardResponse {
    bytes gradient_data = 1;
    TensorShape gradient_shape = 2;
    bool success = 3;
}
```

#### 2.2 实现梯度序列化工具

**新建文件**: `SplitLearnComm/src/splitlearn_comm/utils/gradient_utils.py`
- `serialize_gradient()`: 序列化梯度张量
- `deserialize_gradient()`: 反序列化梯度张量
- 支持压缩以减少网络传输

#### 2.3 扩展 gRPC 服务接口

**修改**: `SplitLearnComm/src/splitlearn_comm/protocol/compute_service.proto`

**添加服务方法**:
```protobuf
service ComputeService {
    rpc Compute(ComputeRequest) returns (ComputeResponse);
    
    // 新增
    rpc Backward(BackwardRequest) returns (BackwardResponse);
    rpc GetGradients(GradientRequest) returns (GradientResponse);
}
```

---

### 3. 客户端训练支持

#### 3.1 创建训练客户端

**新建文件**: `SplitLearnComm/src/splitlearn_comm/training_client.py`

**核心功能**:
- 前向传播（保留梯度）
- 反向传播（传递梯度）
- 请求ID管理（关联前向和反向）
- 中间状态缓存

#### 3.2 创建训练器类

**新建文件**: `test/client/training_client.py`

**核心功能**:
- 训练步骤（forward + backward + update）
- 训练循环（epoch loop）
- 损失计算
- 优化器管理
- 检查点保存

---

### 4. 服务器端训练支持

#### 4.1 扩展服务器类

**修改**: `SplitLearnManager/src/splitlearn_manager/server/managed_server.py`

**需要添加**:
- 训练模式支持
- 前向传播状态缓存
- 反向传播处理
- 优化器管理

#### 4.2 实现梯度处理

**核心逻辑**:
```python
def backward(self, gradient, request_id):
    # 1. 从缓存获取前向传播状态
    cache = self._forward_cache[request_id]
    
    # 2. 重新前向传播（保留梯度）
    output = model(cache['input'])
    
    # 3. 反向传播
    output.backward(gradient=gradient)
    
    # 4. 获取输入梯度
    input_gradient = cache['input'].grad
    
    # 5. 更新参数
    optimizer.step()
    
    return input_gradient
```

---

### 5. 完整训练流程

#### 5.1 训练步骤流程

```
1. 清零梯度
   optimizer.zero_grad()

2. 前向传播
   hidden_1 = bottom(input)
   hidden_2 = trunk_client.forward(hidden_1, request_id)
   output = top(hidden_2)

3. 计算损失
   loss = criterion(output, labels)

4. 反向传播
   loss.backward()
   grad_hidden_2 = hidden_2.grad
   grad_hidden_1 = trunk_client.backward(grad_hidden_2, request_id)
   hidden_1.backward(grad_hidden_1)

5. 梯度裁剪（可选）
   clip_grad_norm()

6. 参数更新
   optimizer_bottom.step()
   optimizer_top.step()
```

#### 5.2 数据加载

**新建文件**: `test/data/dataset.py`
- 文本数据集类
- DataLoader 封装

---

## 📋 实施清单

### Phase 1: 基础准备 ✅
- [ ] 创建 `TrainingConfig` 类
- [ ] 修改模型模式管理（支持 train/eval 切换）
- [ ] 移除 `torch.no_grad()` 限制
- [ ] 本地训练测试（单机，不涉及网络）

### Phase 2: 通信协议 ⚠️
- [ ] 扩展 Protocol Buffer 定义
- [ ] 实现梯度序列化工具
- [ ] 重新生成 gRPC 代码
- [ ] 测试梯度序列化/反序列化

### Phase 3: 客户端训练支持 ⚠️
- [ ] 创建 `TrainingClient` 类
- [ ] 实现前向传播（保留梯度）
- [ ] 实现反向传播（梯度传递）
- [ ] 创建 `SplitLearningTrainer` 类
- [ ] 实现训练循环

### Phase 4: 服务器端训练支持 ⚠️
- [ ] 扩展 `ManagedServer` 类
- [ ] 实现前向传播状态缓存
- [ ] 实现反向传播处理
- [ ] 添加优化器支持
- [ ] 测试服务器端训练

### Phase 5: 集成测试 ⚠️
- [ ] 端到端训练测试
- [ ] 梯度传递正确性验证
- [ ] 参数更新验证
- [ ] 性能测试

---

## 🔑 关键技术点

### 1. 梯度传递流程

```
Top模型 → 计算梯度 → grad_hidden_2
                            ↓
                    网络传输（序列化）
                            ↓
Trunk服务器 → 反向传播 → grad_hidden_1
                            ↓
                    网络传输（序列化）
                            ↓
Bottom模型 → 接收梯度 → 更新参数
```

### 2. 状态缓存机制

**为什么需要缓存？**
- 反向传播需要前向传播的中间状态
- 需要保存输入张量用于重新计算

**缓存内容**:
```python
{
    'request_id': {
        'input': tensor,      # 输入张量
        'output': tensor,     # 输出张量
        'model_name': str,    # 模型名称
        'timestamp': float    # 时间戳（用于清理）
    }
}
```

### 3. 请求ID关联

**用途**: 将前向传播和反向传播关联起来

```python
# 前向传播
request_id = uuid.uuid4().hex
output = client.forward(input, request_id=request_id)

# 反向传播（使用相同的 request_id）
gradient = client.backward(output_grad, request_id=request_id)
```

---

## 🚨 常见问题和解决方案

### Q1: 内存不足
**A**: 使用梯度累积、混合精度训练、减少批次大小

### Q2: 梯度消失/爆炸
**A**: 使用梯度裁剪、学习率调度、梯度归一化

### Q3: 训练不稳定
**A**: 调整学习率、使用学习率调度器、增加梯度裁剪

### Q4: 网络延迟影响性能
**A**: 使用异步通信、批量梯度传输、梯度压缩

---

## 📚 参考文件

详细实现指南: `test/TRAINING_IMPLEMENTATION_GUIDE.md`

包含:
- 完整的代码示例
- 详细的实现步骤
- 架构设计说明
- 测试方法
- 问题解决方案

---

## ⏱️ 预计工作量

| 阶段 | 工作量 | 难度 |
|------|--------|------|
| Phase 1: 基础准备 | 1-2周 | ⭐⭐ |
| Phase 2: 通信协议 | 2-3周 | ⭐⭐⭐⭐ |
| Phase 3: 客户端支持 | 2-3周 | ⭐⭐⭐ |
| Phase 4: 服务器支持 | 2-3周 | ⭐⭐⭐⭐ |
| Phase 5: 测试优化 | 2-3周 | ⭐⭐⭐ |

**总计**: 约 9-14 周（2-3.5个月）

---

## 🎓 学习资源

1. **PyTorch 分布式训练**:
   - PyTorch 官方文档: Distributed Training
   
2. **Split Learning 论文**:
   - "Split Learning for Health: Distributed Deep Learning without Sharing Raw Patient Data"

3. **gRPC 高级特性**:
   - gRPC 流式传输
   - gRPC 异步调用

---

## 💡 快速开始

1. **阅读详细指南**: `TRAINING_IMPLEMENTATION_GUIDE.md`
2. **从 Phase 1 开始**: 实现基础训练功能（不涉及网络）
3. **逐步扩展**: 按照清单逐项完成
4. **充分测试**: 每个阶段都要进行测试

---

## ✅ 成功标准

训练功能实现成功的标志：

1. ✅ 能够执行完整的训练步骤
2. ✅ 梯度能够正确传递（客户端 ↔ 服务器）
3. ✅ 模型参数能够正确更新
4. ✅ 训练损失能够正常下降
5. ✅ 支持检查点保存和恢复
6. ✅ 支持训练和推理模式切换
