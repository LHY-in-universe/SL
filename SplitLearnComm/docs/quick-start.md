# 快速入门指南

在 5 分钟内开始使用 splitlearn-comm。

## 安装

```bash
pip install splitlearn-comm
```

或者从源码安装：

```bash
git clone https://github.com/yourusername/splitlearn-comm.git
cd splitlearn-comm
pip install -e .
```

## 基本用法

### 1. 创建服务器 (server.py)

```python
#!/usr/bin/env python3
import torch
from splitlearn_comm import GRPCComputeServer
from splitlearn_comm.core import ModelComputeFunction

# 创建你的 PyTorch 模型
model = torch.nn.Sequential(
    torch.nn.Linear(768, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 768)
)

# 将其包装在 ComputeFunction 中
compute_fn = ModelComputeFunction(
    model=model,
    device="cpu",  # 如果你有 GPU 则使用 "cuda"
    model_name="MyModel"
)

# 启动服务器
server = GRPCComputeServer(
    compute_fn=compute_fn,
    host="0.0.0.0",
    port=50051
)

print("Starting server on port 50051...")
server.start()
server.wait_for_termination()
```

运行服务器：
```bash
python server.py
```

### 2. 创建客户端 (client.py)

```python
#!/usr/bin/env python3
import torch
from splitlearn_comm import GRPCComputeClient

# 连接到服务器
client = GRPCComputeClient("localhost:50051")

if not client.connect():
    print("Failed to connect to server")
    exit(1)

print("Connected to server!")

# 执行计算
input_tensor = torch.randn(1, 10, 768)
output_tensor = client.compute(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")

# 获取统计信息
stats = client.get_statistics()
print(f"Total requests: {stats['total_requests']}")
print(f"Avg time: {stats['avg_total_time_ms']:.2f}ms")

client.close()
```

运行客户端：
```bash
python client.py
```

## 下一步

### 使用上下文管理器

为了更清晰的资源管理：

```python
# 服务器
with GRPCComputeServer(compute_fn, port=50051) as server:
    print("Server running...")
    # 退出上下文时服务器自动停止

# 客户端
with GRPCComputeClient("localhost:50051") as client:
    output = client.compute(input_tensor)
# 连接自动关闭
```

### 自定义计算逻辑

```python
from splitlearn_comm.core import ComputeFunction

class MyCustomFunction(ComputeFunction):
    def compute(self, input_tensor):
        # 你的自定义逻辑
        return input_tensor * 2 + 1

    def get_info(self):
        return {"name": "MyCustomFunction"}

# 使用它
server = GRPCComputeServer(
    compute_fn=MyCustomFunction(),
    port=50051
)
```

### 添加重试逻辑

```python
from splitlearn_comm import GRPCComputeClient, ExponentialBackoff

retry = ExponentialBackoff(
    max_retries=5,
    initial_delay=1.0,
    max_delay=30.0
)

client = GRPCComputeClient(
    "localhost:50051",
    retry_strategy=retry
)
```

### 使用压缩

对于 WAN 上的远程服务器：

```python
from splitlearn_comm.core import CompressedTensorCodec

codec = CompressedTensorCodec(compression_level=6)

server = GRPCComputeServer(
    compute_fn=compute_fn,
    codec=codec,
    port=50051
)

client = GRPCComputeClient(
    "remote-server:50051",
    codec=codec
)
```

### 健康检查

```python
if client.health_check():
    print("Server is healthy")
else:
    print("Server is not responding")
```

### 服务信息

```python
info = client.get_service_info()
print(f"Service: {info['service_name']}")
print(f"Version: {info['version']}")
print(f"Device: {info['device']}")
print(f"Uptime: {info['uptime_seconds']:.1f}s")
print(f"Total requests: {info['total_requests']}")
```

## 常见模式

### 模式 1: 单机测试

```python
# 终端 1: 启动服务器
python examples/simple_server.py

# 终端 2: 运行客户端
python examples/simple_client.py
```

### 模式 2: LAN 部署

```python
# 服务器 (192.168.1.10)
server = GRPCComputeServer(
    compute_fn=compute_fn,
    host="0.0.0.0",  # 监听所有接口
    port=50051
)

# 客户端 (LAN 上的任何机器)
client = GRPCComputeClient("192.168.1.10:50051")
```

### 模式 3: GPU 服务器, CPU 客户端

```python
# 带有 GPU 的服务器
compute_fn = ModelComputeFunction(
    model=model,
    device="cuda:0"  # 使用 GPU
)

# 客户端 (可以在仅 CPU 的机器上)
client = GRPCComputeClient("gpu-server:50051")
output = client.compute(input_tensor)  # 在 GPU 上计算
```

### 模式 4: 多客户端

```python
# 多个客户端可以连接到同一个服务器
client1 = GRPCComputeClient("localhost:50051")
client2 = GRPCComputeClient("localhost:50051")
client3 = GRPCComputeClient("localhost:50051")

# 都可以并行计算
output1 = client1.compute(input1)
output2 = client2.compute(input2)
output3 = client3.compute(input3)
```

## 故障排除

### 连接被拒绝 (Connection Refused)

**问题:** 客户端无法连接到服务器

**解决方案:**
1. 检查服务器是否正在运行
2. 检查防火墙设置
3. 验证 host:port 是否正确
4. 在服务器上使用 `0.0.0.0` 以进行 LAN 访问

### 超时错误 (Timeout Errors)

**问题:** 请求超时

**解决方案:**
1. 增加超时时间:
   ```python
   client = GRPCComputeClient("localhost:50051", timeout=60.0)
   ```
2. 检查服务器是否有足够的资源
3. 减少批量大小

### CUDA 内存不足 (CUDA Out of Memory)

**问题:** 服务器上的 GPU 内存错误

**解决方案:**
1. 减少批量大小
2. 使用 CPU 代替 GPU
3. 使用参数较少的模型
4. 清除缓存: `torch.cuda.empty_cache()`

### 导入错误 (Import Errors)

**问题:** `ModuleNotFoundError: No module named 'splitlearn_comm'`

**解决方案:**
1. 检查安装: `pip show splitlearn-comm`
2. 重新安装: `pip install --upgrade splitlearn-comm`
3. 如果使用源码: `pip install -e .`

## 示例

查看 examples 目录获取更多信息：

- `examples/simple_server.py` - 基本服务器
- `examples/simple_client.py` - 基本客户端
- `examples/custom_service.py` - 自定义计算函数

## 了解更多

- [API 参考](api.md) - 完整 API 文档
- [协议文档](protocol.md) - 协议详情
- [扩展指南](extending.md) - 自定义指南

## 获取帮助

- GitHub Issues: https://github.com/yourusername/splitlearn-comm/issues
- 文档: https://github.com/yourusername/splitlearn-comm/docs
