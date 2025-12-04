# Comm 库 vs Manager 库：架构分工

## 你的理解是正确的！

### 核心分工

#### **SplitLearnComm（通信库）**
**职责：建立两个 IP 之间的通信连接**

- ✅ 提供 gRPC 服务器和客户端
- ✅ 处理网络连接（TCP/IP）
- ✅ 张量数据的序列化/反序列化（编解码）
- ✅ 消息传输（请求/响应）
- ✅ 连接管理（连接、断开、重试）
- ❌ **不关心具体的模型逻辑**
- ❌ **不知道模型如何加载、如何推理**

#### **SplitLearnManager（管理层）**
**职责：具体如何使用模型**

- ✅ 模型加载和卸载（从 HuggingFace、本地文件等）
- ✅ 模型生命周期管理
- ✅ 模型路由（多模型管理）
- ✅ 资源管理（内存、GPU）
- ✅ 使用 Comm 库进行通信
- ✅ 提供高级 API（如 `ManagedServer`）

---

## 架构层次

```
┌─────────────────────────────────────────────────┐
│         SplitLearnManager（管理层）              │
│  - 模型加载/卸载                                 │
│  - 模型路由                                      │
│  - 资源管理                                      │
│  - 高级 API（ManagedServer）                    │
└─────────────────────────────────────────────────┘
                    │ 使用
                    ▼
┌─────────────────────────────────────────────────┐
│         SplitLearnComm（通信层）                 │
│  - gRPC 服务器/客户端                            │
│  - 网络连接                                      │
│  - 张量编解码                                    │
│  - 消息传输                                      │
└─────────────────────────────────────────────────┘
                    │ 使用
                    ▼
┌─────────────────────────────────────────────────┐
│         gRPC / TCP/IP（网络层）                  │
└─────────────────────────────────────────────────┘
```

---

## 代码示例：它们如何协作

### 1. Manager 使用 Comm 进行通信

**位置**: `SplitLearnManager/src/splitlearn_manager/server/async_managed_server.py`

```python
from splitlearn_comm import AsyncGRPCComputeServer  # 使用 Comm 库
from splitlearn_comm.core import AsyncComputeFunction

class AsyncManagedServer:
    def __init__(self, config):
        # Manager 负责：模型管理
        self.model_manager = AsyncModelManager(...)  # 加载、管理模型
        
        # Manager 负责：创建计算函数（决定如何使用模型）
        self.compute_fn = AsyncManagedComputeFunction(
            model_manager=self.model_manager,  # 使用模型管理器
            router=self.router,
            ...
        )
        
        # Manager 使用 Comm：创建 gRPC 服务器（建立通信）
        self.grpc_server = AsyncGRPCComputeServer(  # 来自 Comm 库
            compute_fn=self.compute_fn,  # 传入计算函数
            host=self.config.host,
            port=self.config.port,
            ...
        )
```

### 2. Manager 的计算函数决定如何使用模型

**位置**: `SplitLearnManager/src/splitlearn_manager/server/async_managed_server.py`

```python
class AsyncManagedComputeFunction(AsyncComputeFunction):
    """Manager 层：决定如何使用模型"""
    
    async def compute(self, input_tensor):
        # Manager 负责：路由到哪个模型
        model_id = self.router.route_to_model()
        
        # Manager 负责：获取模型（从模型管理器）
        managed_model = await self.model_manager.get_model(model_id)
        
        # Manager 负责：执行推理（如何使用模型）
        output = managed_model.model(input_tensor)
        
        return output
```

### 3. Comm 只负责传输数据

**位置**: `SplitLearnComm/src/splitlearn_comm/server/async_servicer.py`

```python
class AsyncComputeServicer:
    """Comm 层：只负责通信，不关心模型逻辑"""
    
    async def Compute(self, request, context):
        # Comm 负责：解码输入张量（从网络接收）
        input_tensor = self.codec.decode(request.data, request.shape)
        
        # Comm 负责：调用计算函数（由 Manager 提供）
        output_tensor = await self.compute_fn.compute(input_tensor)
        
        # Comm 负责：编码输出张量（发送到网络）
        output_data, output_shape = self.codec.encode(output_tensor)
        
        return ComputeResponse(data=output_data, shape=output_shape)
```

---

## 详细对比

| 功能 | SplitLearnComm | SplitLearnManager |
|------|----------------|-------------------|
| **网络连接** | ✅ 负责 | ❌ 不负责 |
| **gRPC 服务器/客户端** | ✅ 提供 | ❌ 不提供 |
| **张量编解码** | ✅ 负责 | ❌ 不负责 |
| **消息传输** | ✅ 负责 | ❌ 不负责 |
| **模型加载** | ❌ 不负责 | ✅ 负责 |
| **模型推理** | ❌ 不负责 | ✅ 负责 |
| **模型管理** | ❌ 不负责 | ✅ 负责 |
| **资源管理** | ❌ 不负责 | ✅ 负责 |
| **模型路由** | ❌ 不负责 | ✅ 负责 |

---

## 数据流示例

### 客户端请求 → 服务器响应

```
1. 客户端（Manager）
   └─> 准备输入张量
   └─> 调用 Comm 客户端发送请求
       │
       ▼
2. Comm 客户端（Comm）
   └─> 编码张量为 bytes
   └─> 通过 gRPC 发送到服务器
       │
       ▼
3. 网络传输（TCP/IP）
   └─> 数据包传输
       │
       ▼
4. Comm 服务器（Comm）
   └─> 接收 gRPC 请求
   └─> 解码 bytes 为张量
   └─> 调用计算函数（由 Manager 提供）
       │
       ▼
5. Manager 计算函数（Manager）
   └─> 路由到模型
   └─> 从模型管理器获取模型
   └─> 执行模型推理
   └─> 返回结果
       │
       ▼
6. Comm 服务器（Comm）
   └─> 编码输出张量为 bytes
   └─> 通过 gRPC 发送响应
       │
       ▼
7. 网络传输（TCP/IP）
   └─> 数据包传输
       │
       ▼
8. Comm 客户端（Comm）
   └─> 接收 gRPC 响应
   └─> 解码 bytes 为张量
   └─> 返回给 Manager
       │
       ▼
9. 客户端（Manager）
   └─> 获得输出张量
```

---

## 为什么这样设计？

### 1. **关注点分离（Separation of Concerns）**

- **Comm**：专注于通信，可以用于任何需要传输张量的场景
- **Manager**：专注于模型管理，可以替换不同的通信方式

### 2. **可复用性**

- **Comm** 可以独立使用，不依赖 Manager
- **Manager** 可以使用 Comm，也可以使用其他通信方式

### 3. **灵活性**

- 可以单独使用 Comm 进行简单的模型服务
- 可以使用 Manager 进行复杂的模型管理

---

## 实际使用场景

### 场景 1：只使用 Comm（简单场景）

```python
# 直接使用 Comm，自己管理模型
from splitlearn_comm import GRPCComputeServer
from splitlearn_comm.core import ModelComputeFunction

# 自己加载模型
model = load_model_from_huggingface("gpt2")

# 包装为计算函数
compute_fn = ModelComputeFunction(model, device="cpu")

# 使用 Comm 创建服务器
server = GRPCComputeServer(compute_fn, port=50051)
server.start()
```

### 场景 2：使用 Manager（复杂场景）

```python
# 使用 Manager，自动管理模型
from splitlearn_manager.quickstart import ManagedServer

# Manager 负责：
# - 自动加载模型
# - 管理模型生命周期
# - 使用 Comm 进行通信
server = ManagedServer("gpt2", port=50051)
server.start()
```

---

## 总结

### ✅ 你的理解完全正确！

1. **Comm 库**：负责建立两个 IP 之间的通信连接
   - 提供 gRPC 服务器/客户端
   - 处理网络连接和数据传输
   - 不关心模型逻辑

2. **Manager 库**：负责具体如何使用模型
   - 模型加载、管理、路由
   - 使用 Comm 库进行通信
   - 提供高级 API

### 关系

- **Manager 依赖 Comm**：Manager 使用 Comm 进行通信
- **Comm 独立**：Comm 可以单独使用，不依赖 Manager
- **分工明确**：通信 vs 模型管理

