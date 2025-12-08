# 完整联合训练实现路线图

## 目标

实现完整的 Bottom + Trunk + Top 三部分联合训练。

---

## 实现路线图

### 阶段 1: 协议扩展（必需）

1. **扩展 Protocol Buffer 定义**
   - 添加 `BackwardRequest` 消息
   - 添加 `BackwardResponse` 消息
   - 在 `ComputeService` 中添加 `Backward` RPC

2. **重新生成 Python 代码**
   - 运行 protoc 生成新的 Python 代码

### 阶段 2: 工具实现（必需）

1. **梯度序列化工具**
   - 创建 `gradient_utils.py`
   - 实现张量序列化/反序列化

### 阶段 3: 服务器端扩展（必需）

1. **扩展服务器 Servicer**
   - 添加梯度缓存机制
   - 实现 `Backward` 方法
   - 支持 Trunk 模型的反向传播

### 阶段 4: 客户端扩展（必需）

1. **创建支持训练的客户端**
   - 扩展 `TrunkClient` 支持反向传播
   - 实现前向传播缓存
   - 实现反向传播调用

### 阶段 5: 训练脚本（必需）

1. **完整的联合训练脚本**
   - 创建新的训练脚本
   - 实现三部分联合训练
   - 测试完整流程

---

## 当前进度

- [ ] 阶段 1: 协议扩展
- [ ] 阶段 2: 工具实现
- [ ] 阶段 3: 服务器端扩展
- [ ] 阶段 4: 客户端扩展
- [ ] 阶段 5: 训练脚本

---

## 关键技术点

### 1. 梯度缓存机制

服务器需要缓存前向传播的中间状态，以便反向传播时使用：

```python
# 前向传播时缓存
forward_cache[request_id] = {
    'input': hidden_states,
    'output': output,
    'model': model
}

# 反向传播时使用
gradient = backward(gradient, request_id)
```

### 2. 梯度传递流程

```
客户端 Top:
  loss.backward() → grad_hidden_2

客户端 → 服务器:
  Backward(grad_hidden_2, request_id)

服务器 Trunk:
  output.backward(grad_hidden_2) → grad_hidden_1

服务器 → 客户端:
  BackwardResponse(grad_hidden_1)

客户端 Bottom:
  hidden_1.backward(grad_hidden_1)
```

### 3. 参数更新

- Bottom: 客户端本地更新
- Trunk: 服务器端更新（需要优化器）
- Top: 客户端本地更新

---

## 下一步

你想要我开始实现哪个阶段？

我建议按照以下顺序：
1. 阶段 1: 协议扩展
2. 阶段 2: 工具实现
3. 阶段 3-5: 逐步实现和测试

开始实现吗？

