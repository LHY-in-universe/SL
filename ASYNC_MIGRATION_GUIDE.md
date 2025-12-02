# 异步 API 迁移指南

## 概述

本指南帮助您从同步 API 迁移到新的异步 API。异步版本解决了锁阻塞问题，提供更好的并发性能。

## 为什么要迁移？

### 同步版本的问题

在同步版本中，`ModelManager.load_model()` 在持有全局锁期间加载模型：

```python
# 同步版本（存在问题）
with self.lock:  # 获取锁
    model = self.loader.load_from_config(config)  # 阻塞几秒到几分钟！
    self.models[model_id] = model
# 在此期间，所有操作都被阻塞：list_models()、unload_model()、推理请求等
```

**问题影响：**
- ❌ `list_models()` 在模型加载期间被阻塞（>1000ms）
- ❌ 无法并发加载多个模型
- ❌ 健康检查和监控被阻塞
- ❌ 系统看起来"卡住"了

### 异步版本的优势

```python
# 异步版本（已解决）
async with self.lock:  # 短暂持锁 <1ms
    self.models[model_id] = LoadingPlaceholder()  # 占位符
# 释放锁

# 锁外加载模型（不阻塞其他操作）
model = await loop.run_in_executor(executor, load_model, config)

async with self.lock:  # 短暂持锁 <1ms
    self.models[model_id] = ManagedModel(model)
# 释放锁
```

**性能提升：**
- ✅ `list_models()` 延迟：>1000ms → <10ms（降低 99%）
- ✅ 支持并发加载多个模型
- ✅ 并发 QPS 提升 2-3倍
- ✅ P99 延迟降低 >30%

---

## 迁移步骤

### 1. 更新依赖

确保您使用的是最新版本：

```bash
cd SplitLearnManager
pip install -e . --upgrade

cd ../SplitLearnComm
pip install -e . --upgrade
```

### 2. 代码迁移

#### 2.1 ModelManager 迁移

**同步版本（旧）：**
```python
from splitlearn_manager import ModelManager, ModelConfig

# 创建管理器
manager = ModelManager(max_models=5)

# 加载模型（阻塞）
config = ModelConfig(model_id="model1", ...)
manager.load_model(config)

# 列出模型
models = manager.list_models()
```

**异步版本（新）：**
```python
from splitlearn_manager import AsyncModelManager, ModelConfig
import asyncio

async def main():
    # 创建异步管理器
    manager = AsyncModelManager(max_models=5)

    # 异步加载模型（不阻塞！）
    config = ModelConfig(model_id="model1", ...)
    await manager.load_model(config)

    # 异步列出模型
    models = await manager.list_models()

    # 清理
    await manager.shutdown()

# 运行
asyncio.run(main())
```

#### 2.2 ManagedServer 迁移

**同步版本（旧）：**
```python
from splitlearn_manager import ManagedServer, ServerConfig

# 创建服务器
config = ServerConfig(host="0.0.0.0", port=50051)
server = ManagedServer(config)

# 启动（阻塞）
server.start()
server.wait_for_termination()
```

**异步版本（新）：**
```python
from splitlearn_manager.server import AsyncManagedServer
from splitlearn_manager.config import ServerConfig
import asyncio

async def main():
    # 创建异步服务器
    config = ServerConfig(host="0.0.0.0", port=50051)
    server = AsyncManagedServer(config)

    # 异步启动（不阻塞！）
    await server.start()

    # 异步加载模型
    await server.load_model(model_config)

    # 等待终止
    await server.wait_for_termination()

asyncio.run(main())
```

#### 2.3 gRPC Server 迁移

**同步版本（旧）：**
```python
from splitlearn_comm import GRPCComputeServer, ModelComputeFunction

compute_fn = ModelComputeFunction(model, device="cuda")
server = GRPCComputeServer(compute_fn, port=50051)

server.start()
server.wait_for_termination()
```

**异步版本（新）：**
```python
from splitlearn_comm import AsyncGRPCComputeServer, AsyncModelComputeFunction
import asyncio

async def main():
    compute_fn = AsyncModelComputeFunction(model, device="cuda")
    server = AsyncGRPCComputeServer(compute_fn, port=50051)

    await server.start()
    await server.wait_for_termination()

asyncio.run(main())
```

---

## 常见迁移场景

### 场景 1：并发加载多个模型

**同步版本（串行加载）：**
```python
# 串行加载，耗时相加
for config in model_configs:
    manager.load_model(config)  # 阻塞
```

**异步版本（并行加载）：**
```python
# 并行加载，耗时为最长的那个
await asyncio.gather(
    *[manager.load_model(config) for config in model_configs]
)
```

### 场景 2：在加载期间执行其他操作

**同步版本（无法实现）：**
```python
# 无法在加载期间做其他事情
manager.load_model(config)  # 阻塞，必须等待
```

**异步版本（可以实现）：**
```python
# 启动加载任务
load_task = asyncio.create_task(manager.load_model(config))

# 在加载期间执行其他操作
while not load_task.done():
    models = await manager.list_models()  # 不会被阻塞！
    print(f"已加载: {len(models)}个模型")
    await asyncio.sleep(1)

# 等待加载完成
await load_task
```

### 场景 3：上下文管理器

**异步版本支持自动管理生命周期：**
```python
async with AsyncManagedServer(config) as server:
    await server.load_model(model_config)
    # 做一些工作
    await asyncio.sleep(10)
# 自动停止服务器
```

---

## 向后兼容性

### 保留的同步 API

所有同步 API 仍然可用，不会破坏现有代码：

```python
# 这些仍然有效（但不推荐用于新代码）
from splitlearn_manager import ModelManager  # 同步版本
from splitlearn_comm import GRPCComputeServer  # 同步版本

manager = ModelManager()
manager.load_model(config)  # 仍然工作，但有锁阻塞问题
```

### 弃用警告

在未来版本中，同步 API 可能会显示弃用警告：

```python
DeprecationWarning:
ModelManager is deprecated, use AsyncModelManager instead for better performance.
See: https://github.com/yourusername/SplitLearnManager/docs/MIGRATION_GUIDE.md
```

---

## 常见问题（FAQ）

### Q1: 我必须立即迁移吗？

**A:** 不，同步 API 仍然可用。但如果您遇到以下问题，强烈建议迁移：
- 模型加载期间系统响应缓慢
- 需要并发加载多个模型
- 需要更高的并发 QPS

### Q2: 迁移工作量大吗？

**A:** 取决于您的代码规模，但通常很小：
- 主要是添加 `async`/`await` 关键字
- 将同步函数包装在 `asyncio.run()` 中
- 大部分 API 签名相同

### Q3: 异步版本的性能真的更好吗？

**A:** 是的，我们的基准测试显示：
- ✅ `list_models()` 延迟降低 99%
- ✅ 并发 QPS 提升 2-3倍
- ✅ P99 延迟降低 >30%

### Q4: 可以混用同步和异步 API 吗？

**A:** 技术上可以，但不推荐。建议：
- 新代码：使用异步 API
- 现有代码：逐步迁移

### Q5: asyncio 学习曲线陡峭吗？

**A:** 对于基本使用，只需要掌握：
- `async def` 定义异步函数
- `await` 等待异步操作
- `asyncio.run()` 运行异步主函数
- `asyncio.gather()` 并发执行多个任务

---

## 完整示例

### 从同步到异步：完整对比

**同步版本完整示例：**
```python
from splitlearn_manager import ManagedServer, ModelConfig, ServerConfig

def main():
    # 创建配置
    server_config = ServerConfig(port=50051)
    model_config = ModelConfig(model_id="m1", ...)

    # 创建服务器
    server = ManagedServer(server_config)
    server.start()  # 阻塞

    # 加载模型（阻塞）
    server.load_model(model_config)  # 其他操作被阻塞

    # 运行
    server.wait_for_termination()

if __name__ == "__main__":
    main()
```

**异步版本完整示例：**
```python
from splitlearn_manager.server import AsyncManagedServer
from splitlearn_manager.config import ModelConfig, ServerConfig
import asyncio

async def main():
    # 创建配置
    server_config = ServerConfig(port=50051)
    model_config = ModelConfig(model_id="m1", ...)

    # 创建异步服务器
    server = AsyncManagedServer(server_config)
    await server.start()  # 不阻塞

    # 异步加载模型（不阻塞其他操作）
    await server.load_model(model_config)

    # 运行
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 更多资源

- **示例代码**: `SplitLearnManager/examples/async_server_example.py`
- **API 文档**: `SplitLearnManager/docs/async_api.md`
- **性能基准**: `testcode/benchmark_async.py`

---

## 获取帮助

如果您在迁移过程中遇到问题：

1. 查看示例代码：`SplitLearnManager/examples/async_server_example.py`
2. 检查日志：启用 DEBUG 级别日志查看详细信息
3. 提交 Issue：https://github.com/yourusername/SplitLearnManager/issues

---

**祝迁移顺利！异步版本会为您带来更好的性能和用户体验。** 🚀
