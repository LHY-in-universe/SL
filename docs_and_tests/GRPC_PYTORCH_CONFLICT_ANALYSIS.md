# gRPC 和 PyTorch 冲突问题分析

## 问题现象

**单独使用 gRPC 或 PyTorch 都没问题，但同时使用就会出现问题。**

## 常见问题

### 1. mutex 警告

**症状：**
```
[mutex.cc : 452] RAW: Lock blocking 0x...
```

**原因：**
- gRPC 使用 `ThreadPoolExecutor` 处理请求（多线程）
- PyTorch 内部也使用多线程（计算线程）
- 多个线程同时访问 PyTorch 的内部资源导致 mutex 竞争

**解决方案：**
```python
# 1. 设置 gRPC 单线程模式
server = GRPCComputeServer(
    compute_fn=compute_fn,
    max_workers=1  # 单线程
)

# 2. 设置 PyTorch 单线程
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# 3. 设置环境变量（在导入 torch 之前）
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
```

### 2. 性能下降

**症状：**
- 同时使用时性能明显下降
- 响应时间变长

**原因：**
- 线程竞争导致上下文切换开销
- 锁竞争导致等待时间增加

**解决方案：**
- 使用单线程模式（牺牲并发，保证稳定性）
- 使用异步版本（`AsyncGRPCComputeServer`，使用协程而不是线程）

### 3. 死锁或卡住

**症状：**
- 程序卡住不响应
- 请求超时

**原因：**
- 线程死锁
- 锁的嵌套使用

**解决方案：**
- 避免在计算函数中使用锁
- 使用超时机制
- 检查锁的使用顺序

### 4. 内存问题

**症状：**
- 内存持续增长
- 内存泄漏

**原因：**
- 线程池缓存
- 模型未释放
- 张量未释放

**解决方案：**
```python
# 使用 torch.no_grad() 避免梯度计算
with torch.no_grad():
    output = model(input_tensor)

# 及时释放不需要的张量
del large_tensor
torch.cuda.empty_cache()  # 如果使用 GPU
```

## 根本原因

### 1. 线程模型冲突

**gRPC 的线程模型：**
- 使用 `ThreadPoolExecutor` 处理请求
- 每个请求在独立线程中执行
- 默认使用多个线程（如 10 个）

**PyTorch 的线程模型：**
- 内部使用多线程进行并行计算
- 使用 OpenMP/MKL 进行并行
- 默认使用多个线程

**冲突：**
- 多个 gRPC 线程同时调用 PyTorch
- PyTorch 内部也在使用多线程
- 导致线程数量爆炸（10 个 gRPC 线程 × 4 个 PyTorch 线程 = 40 个线程）
- 线程竞争和上下文切换开销巨大

### 2. 初始化顺序问题

**问题：**
- 如果先导入 `torch`，PyTorch 会初始化线程池
- 再导入 `grpc`，gRPC 也会初始化线程池
- 两者可能冲突

**解决：**
```python
# ✅ 正确的顺序
# 1. 先设置环境变量
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# 2. 再导入 torch
import torch
torch.set_num_threads(1)

# 3. 最后导入 grpc
from splitlearn_comm import GRPCComputeServer
```

### 3. 资源竞争

**问题：**
- gRPC 线程池和 PyTorch 线程池竞争 CPU 资源
- 多个线程同时访问 PyTorch 的内部数据结构
- 导致 mutex 竞争

## 解决方案

### 方案 1：单线程模式（推荐用于测试）

```python
# gRPC 单线程
server = GRPCComputeServer(
    compute_fn=compute_fn,
    max_workers=1  # 单线程
)

# PyTorch 单线程
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
```

**优点：**
- 简单直接
- 避免线程冲突
- 适合测试和小规模使用

**缺点：**
- 无法充分利用多核 CPU
- 性能可能下降

### 方案 2：异步版本（推荐用于生产）

```python
# 使用异步版本
from splitlearn_comm import AsyncGRPCComputeServer

server = AsyncGRPCComputeServer(
    compute_fn=compute_fn,
    host="0.0.0.0",
    port=50051
)
```

**优点：**
- 使用协程而不是线程
- 避免线程冲突
- 性能更好

**缺点：**
- 需要异步代码
- 稍微复杂一些

### 方案 3：线程隔离

```python
# 使用线程本地存储
import threading

class ThreadLocalModel:
    _local = threading.local()
    
    def get_model(self):
        if not hasattr(self._local, 'model'):
            self._local.model = load_model()
        return self._local.model
```

**优点：**
- 每个线程有独立的模型实例
- 避免共享状态竞争

**缺点：**
- 内存占用增加
- 实现复杂

## 最佳实践

### 1. 环境变量设置

```python
# 在导入 torch 之前设置
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
```

### 2. PyTorch 配置

```python
import torch

# 设置单线程
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
```

### 3. gRPC 配置

```python
# 单线程模式
server = GRPCComputeServer(
    compute_fn=compute_fn,
    max_workers=1
)

# 或使用异步版本
server = AsyncGRPCComputeServer(
    compute_fn=compute_fn
)
```

### 4. 计算函数

```python
class MyComputeFunction(ComputeFunction):
    def compute(self, input_tensor):
        # 使用 torch.no_grad() 避免梯度计算
        with torch.no_grad():
            output = self.model(input_tensor)
        return output
```

## 诊断工具

运行诊断脚本：
```bash
python testcode/test_grpc_pytorch_conflict.py
```

这个脚本会：
1. 测试 PyTorch 单独使用
2. 测试 gRPC 单独使用
3. 测试两者同时使用
4. 检查线程配置
5. 诊断常见问题

## 总结

### 为什么单独使用没问题？

- **单独使用 PyTorch**：只有 PyTorch 的线程，没有冲突
- **单独使用 gRPC**：只有 gRPC 的线程，没有 PyTorch 的线程竞争

### 为什么同时使用有问题？

- **线程冲突**：gRPC 的多线程 + PyTorch 的多线程 = 线程爆炸
- **资源竞争**：多个线程同时访问 PyTorch 的内部资源
- **初始化顺序**：如果顺序不对，可能导致冲突

### 如何解决？

1. **使用单线程模式**（最简单）
2. **使用异步版本**（推荐）
3. **正确设置环境变量和线程数**

