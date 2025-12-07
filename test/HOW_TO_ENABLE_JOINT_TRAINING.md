# 如何启用完整的三部分联合训练

## 你的需求

**"但是我要一起进行训练"** - 你想要实现 Bottom + Trunk + Top 三个部分一起训练。

---

## 当前状态 vs 目标状态

### 当前状态（简化版本）

```python
# 当前代码 (train_lora_simple.py)
hidden_2 = trunk_client.compute(hidden_1.detach())  # 断开梯度
hidden_2 = hidden_2.requires_grad_(True)            # 重新启用

loss.backward()  # 只计算 Bottom 和 Top 的梯度

optimizer_bottom.step()  # 只更新 Bottom
optimizer_top.step()     # 只更新 Top
# ❌ Trunk 模型不更新
```

### 目标状态（完整版本）

```python
# 目标代码
hidden_2 = trunk_client.compute(hidden_1, request_id=request_id)  # 保留梯度

loss.backward()  # 计算所有三个模型的梯度

# 反向传播到 Trunk
grad_hidden_1 = trunk_client.backward(grad_hidden_2, request_id)
hidden_1.backward(grad_hidden_1)

optimizer_bottom.step()  # 更新 Bottom
optimizer_trunk.step()   # 更新 Trunk ✅
optimizer_top.step()     # 更新 Top
```

---

## 需要实现的内容

### 1. 扩展 gRPC 协议 ⭐（必需）

**当前协议只有前向传播：**
```protobuf
service ComputeService {
    rpc Compute(ComputeRequest) returns (ComputeResponse);
}
```

**需要添加反向传播：**
```protobuf
service ComputeService {
    rpc Compute(ComputeRequest) returns (ComputeResponse);
    rpc Backward(BackwardRequest) returns (BackwardResponse);  // 新增
}
```

### 2. 服务器端支持反向传播 ⭐（必需）

服务器需要：
- 缓存前向传播的状态（输入、输出、模型）
- 实现 `Backward` 方法处理反向传播
- 支持 Trunk 模型的优化器

### 3. 客户端支持反向传播 ⭐（必需）

客户端需要：
- 扩展 `TrunkClient` 支持 `backward()` 方法
- 缓存前向传播的请求 ID
- 传递梯度到服务器并接收返回的梯度

### 4. 完整的训练脚本 ⭐（必需）

创建新的训练脚本，实现三部分联合训练。

---

## 实现步骤

### 步骤 1: 扩展协议（最重要）

**文件**: `SplitLearnComm/src/splitlearn_comm/protocol/protos/compute_service.proto`

需要添加：
- `BackwardRequest` 消息
- `BackwardResponse` 消息
- `Backward` RPC 服务

### 步骤 2: 重新生成代码

运行 protoc 重新生成 Python 代码。

### 步骤 3: 实现服务器端反向传播

**文件**: `SplitLearnComm/src/splitlearn_comm/server/servicer.py`

需要：
- 添加梯度缓存机制
- 实现 `Backward` 方法
- 支持模型的反向传播

### 步骤 4: 实现客户端反向传播

创建支持训练的客户端：
- 缓存前向传播状态
- 实现 `backward()` 方法
- 处理梯度传递

### 步骤 5: 创建完整训练脚本

修改或创建新的训练脚本，实现三部分联合训练。

---

## 关键挑战

### 挑战 1: 梯度缓存

**问题**: 服务器需要记住前向传播的输入，以便反向传播。

**解决方案**: 使用请求 ID 关联前向和反向传播。

### 挑战 2: 梯度传递

**问题**: 梯度需要通过网络传输。

**解决方案**: 使用高效的序列化和压缩。

### 挑战 3: 服务器端优化器

**问题**: Trunk 模型的优化器在服务器端。

**解决方案**: 在服务器端为每个模型创建优化器。

---

## 开始实现

我已经创建了详细的实现指南：

1. **`test/FULL_TRAINING_IMPLEMENTATION.md`** - 完整的实现方案
2. **`test/JOINT_TRAINING_ROADMAP.md`** - 实现路线图

### 建议的实现顺序

1. ✅ **扩展 gRPC 协议** - 这是基础，必须先做
2. ✅ **实现梯度工具** - 用于序列化/反序列化梯度
3. ✅ **扩展服务器端** - 支持反向传播
4. ✅ **扩展客户端** - 支持完整的梯度传递
5. ✅ **创建训练脚本** - 完整的联合训练

---

## 我可以开始实现

你想要我从哪个步骤开始？

我建议从**步骤 1（扩展 gRPC 协议）**开始，因为这是所有其他步骤的基础。

开始实现吗？
