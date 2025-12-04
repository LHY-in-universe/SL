# 异步版本 vs 同步版本详解

## 什么是异步版本？

### 同步版本（当前使用的）

```python
from splitlearn_comm import GRPCComputeServer

# 同步版本使用线程池
server = GRPCComputeServer(
    compute_fn=compute_fn,
    max_workers=10  # 使用 10 个线程处理请求
)
```

**工作原理：**
- 使用 `ThreadPoolExecutor`（线程池）
- 每个请求在**独立的线程**中处理
- 10 个请求 = 10 个线程
- 线程之间会竞争资源

### 异步版本（推荐）

```python
from splitlearn_comm import AsyncGRPCComputeServer

# 异步版本使用协程
server = AsyncGRPCComputeServer(
    compute_fn=compute_fn,
    host="0.0.0.0",
    port=50051
)
```

**工作原理：**
- 使用 `asyncio`（协程）
- 每个请求在**同一个线程**中处理（使用协程切换）
- 10 个请求 = 1 个线程（但可以并发处理）
- 没有线程竞争问题

## 核心区别

### 1. 线程模型

#### 同步版本（GRPCComputeServer）

```
请求 1 ──┐
请求 2 ──┤
请求 3 ──┼──> ThreadPoolExecutor (10 个线程)
请求 4 ──┤      ├─> 线程 1 处理请求 1
请求 5 ──┤      ├─> 线程 2 处理请求 2
...      ┘      ├─> 线程 3 处理请求 3
                └─> ... (最多 10 个线程)
```

**问题：**
- 10 个线程同时运行
- 每个线程可能调用 PyTorch
- PyTorch 内部也有多线程
- 结果：线程数量爆炸（10 × 4 = 40 个线程）

#### 异步版本（AsyncGRPCComputeServer）

```
请求 1 ──┐
请求 2 ──┤
请求 3 ──┼──> asyncio 事件循环 (1 个线程)
请求 4 ──┤      ├─> 协程 1 处理请求 1
请求 5 ──┤      ├─> 协程 2 处理请求 2
...      ┘      ├─> 协程 3 处理请求 3
                └─> ... (所有协程在 1 个线程中切换)
```

**优势：**
- 只有 1 个线程
- 使用协程切换，不是真正的多线程
- 没有线程竞争问题
- 可以处理大量并发请求

### 2. 代码实现

#### 同步版本

```python
# SplitLearnComm/src/splitlearn_comm/server/grpc_server.py

# 使用 ThreadPoolExecutor
self.server = grpc.server(
    futures.ThreadPoolExecutor(max_workers=max_workers),  # 线程池
    options=[...]
)

# 每个请求在独立线程中执行
def Compute(self, request, context):
    # 这个函数在独立线程中运行
    output = self.compute_fn.compute(input_tensor)
    return response
```

#### 异步版本

```python
# SplitLearnComm/src/splitlearn_comm/server/async_grpc_server.py

# 使用 grpc.aio.server (异步版本)
self.server = grpc.aio.server(options=[...])  # 不需要线程池

# 每个请求使用协程
async def Compute(self, request, context):
    # 这个函数使用协程，在同一个线程中运行
    output = await self.compute_fn.compute(input_tensor)
    return response
```

## 为什么异步版本能解决冲突？

### 问题根源

**同步版本的问题：**
```
gRPC 线程池 (10 个线程)
    ↓
每个线程调用 PyTorch
    ↓
PyTorch 内部多线程 (4 个线程)
    ↓
结果：10 × 4 = 40 个线程竞争
```

### 异步版本的解决方案

**异步版本的优势：**
```
asyncio 事件循环 (1 个线程)
    ↓
使用协程处理请求（不是真正的多线程）
    ↓
调用 PyTorch（仍然只有 1 个线程）
    ↓
结果：只有 1 个线程，没有竞争
```

## 协程 vs 线程

### 线程（Thread）

- **真正的并发**：多个线程同时运行
- **需要操作系统调度**：线程切换有开销
- **有竞争问题**：多个线程访问共享资源需要锁
- **资源消耗大**：每个线程需要独立的内存栈

### 协程（Coroutine）

- **伪并发**：看起来并发，但实际是顺序执行
- **用户态调度**：协程切换开销很小
- **无竞争问题**：在同一个线程中，不需要锁
- **资源消耗小**：协程共享同一个线程的栈

## 实际对比

### 场景：处理 100 个并发请求

#### 同步版本

```
100 个请求
    ↓
ThreadPoolExecutor (max_workers=10)
    ↓
10 个线程同时处理
    ↓
每个线程调用 PyTorch
    ↓
PyTorch 使用 4 个线程
    ↓
结果：最多 10 × 4 = 40 个线程
```

**问题：**
- 线程数量多
- 线程竞争严重
- mutex 警告频繁
- 性能下降

#### 异步版本

```
100 个请求
    ↓
asyncio 事件循环
    ↓
1 个线程，使用协程切换处理
    ↓
调用 PyTorch（单线程模式）
    ↓
结果：只有 1 个线程
```

**优势：**
- 线程数量少
- 没有线程竞争
- 无 mutex 警告
- 性能更好

## 使用示例

### 同步版本（当前）

```python
from splitlearn_comm import GRPCComputeServer
from splitlearn_comm.core import ModelComputeFunction

# 创建计算函数
compute_fn = ModelComputeFunction(model, device="cpu")

# 创建服务器（使用线程池）
server = GRPCComputeServer(
    compute_fn=compute_fn,
    host="0.0.0.0",
    port=50051,
    max_workers=1  # 单线程模式（避免冲突）
)

server.start()
server.wait_for_termination()
```

### 异步版本（推荐）

```python
import asyncio
from splitlearn_comm import AsyncGRPCComputeServer
from splitlearn_comm.core import AsyncModelComputeFunction

# 创建异步计算函数
compute_fn = AsyncModelComputeFunction(model, device="cpu")

# 创建异步服务器（使用协程）
server = AsyncGRPCComputeServer(
    compute_fn=compute_fn,
    host="0.0.0.0",
    port=50051
)

# 异步启动
async def main():
    await server.start()
    await server.wait_for_termination()

asyncio.run(main())
```

## 在 SplitLearnManager 中的使用

### 当前实现（已使用异步版本）

```python
# SplitLearnManager/src/splitlearn_manager/server/async_managed_server.py

from splitlearn_comm import AsyncGRPCComputeServer  # 使用异步版本

class AsyncManagedServer:
    def __init__(self):
        # 使用异步服务器
        self.grpc_server = AsyncGRPCComputeServer(
            compute_fn=self.compute_fn,
            host=self.config.host,
            port=self.config.port
        )
```

**这就是为什么 `ManagedServer` 没有 mutex 问题的原因！**

## 总结

### 异步版本的含义

**异步版本 = 使用协程而不是线程**

- **同步版本**：使用线程池，每个请求一个线程
- **异步版本**：使用协程，所有请求在一个线程中处理

### 为什么推荐异步版本？

1. **避免线程冲突**：只有 1 个线程，没有竞争
2. **性能更好**：协程切换开销小
3. **资源消耗少**：不需要多个线程
4. **适合高并发**：可以处理大量并发请求

### 什么时候使用哪个？

- **同步版本**：简单场景，测试用
- **异步版本**：生产环境，高并发场景（推荐）

### 关键理解

**异步 ≠ 多线程**

- 异步使用协程（1 个线程）
- 同步使用线程池（多个线程）
- 异步版本能解决线程冲突问题，因为根本没有多线程！

