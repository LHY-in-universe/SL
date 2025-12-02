# API 参考

splitlearn-comm 的完整 API 文档。

## 目录

- [核心抽象 (Core Abstractions)](#core-abstractions)
  - [ComputeFunction](#computefunction)
  - [ModelComputeFunction](#modelcomputefunction)
  - [TensorCodec](#tensorcodec)
  - [CompressedTensorCodec](#compressedtensorcodec)
- [服务端 (Server)](#server)
  - [GRPCComputeServer](#grpccomputeserver)
  - [ComputeServicer](#computeservicer)
- [客户端 (Client)](#client)
  - [GRPCComputeClient](#grpccomputeclient)
  - [RetryStrategy](#retrystrategy)
  - [ExponentialBackoff](#exponentialbackoff)
  - [FixedDelay](#fixeddelay)

---

## 核心抽象 (Core Abstractions)

### ComputeFunction

实现自定义计算逻辑的抽象基类。

```python
from splitlearn_comm.core import ComputeFunction
```

#### 方法

##### `compute(input_tensor: torch.Tensor) -> torch.Tensor`

**抽象方法** - 必须由子类实现。

对输入张量执行实际计算。

**参数:**
- `input_tensor` (torch.Tensor): 要处理的输入张量

**返回:**
- torch.Tensor: 计算后的输出张量

**示例:**
```python
class MyFunction(ComputeFunction):
    def compute(self, input_tensor):
        # 自定义计算逻辑
        return input_tensor * 2
```

##### `get_info() -> Dict[str, Any]`

返回有关计算函数的信息。

**返回:**
- dict: 包含至少 `{"name": class_name}` 的信息字典

**示例:**
```python
def get_info(self):
    return {
        "name": "MyCustomFunction",
        "version": "1.0",
        "device": "cuda"
    }
```

##### `setup()`

在服务开始前调用的可选设置方法。

**示例:**
```python
def setup(self):
    print("Initializing resources...")
    self.cache = {}
```

##### `teardown()`

在服务关闭时调用的可选清理方法。

**示例:**
```python
def teardown(self):
    print("Cleaning up...")
    self.cache.clear()
```

---

### ModelComputeFunction

用于 PyTorch 模型的 ComputeFunction 具体实现。

```python
from splitlearn_comm.core import ModelComputeFunction
```

#### 构造函数

```python
ModelComputeFunction(
    model: torch.nn.Module,
    device: str = "cpu",
    model_name: Optional[str] = None
)
```

**参数:**
- `model` (torch.nn.Module): 要包装的 PyTorch 模型
- `device` (str): 运行计算的设备 ("cpu", "cuda", "cuda:0" 等)
- `model_name` (str, optional): 模型名称。默认为模型的类名。

**示例:**
```python
model = torch.nn.Linear(768, 768)
compute_fn = ModelComputeFunction(
    model=model,
    device="cuda",
    model_name="TransformerLayer"
)
```

#### 方法

继承自 ComputeFunction 的所有方法，并实现了：
- `compute()`: 运行 model.eval() 并执行带有 no_grad() 的前向传播
- `get_info()`: 返回模型名称和设备信息

---

### TensorCodec

处理张量的序列化和反序列化。

```python
from splitlearn_comm.core import TensorCodec
```

#### 方法

##### `encode(tensor: torch.Tensor) -> Tuple[bytes, Tuple[int, ...]]`

将张量编码为二进制格式。

**参数:**
- `tensor` (torch.Tensor): 要编码的张量

**返回:**
- tuple: (binary_data, shape_tuple)

**示例:**
```python
codec = TensorCodec()
data, shape = codec.encode(torch.randn(2, 3, 4))
# data: bytes, shape: (2, 3, 4)
```

##### `decode(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor`

将二进制数据解码回张量。

**参数:**
- `data` (bytes): 二进制张量数据
- `shape` (tuple): 张量的形状

**返回:**
- torch.Tensor: 重建的张量

**示例:**
```python
tensor = codec.decode(data, shape)
```

---

### CompressedTensorCodec

带有 zlib 压缩的 TensorCodec，用于带宽受限的场景。

```python
from splitlearn_comm.core import CompressedTensorCodec
```

#### 构造函数

```python
CompressedTensorCodec(compression_level: int = 6)
```

**参数:**
- `compression_level` (int): 压缩级别 0-9 (0=无, 9=最大)。默认: 6

**示例:**
```python
codec = CompressedTensorCodec(compression_level=9)
```

---

## 服务端 (Server)

### GRPCComputeServer

用于提供计算函数服务的高级 gRPC 服务器。

```python
from splitlearn_comm import GRPCComputeServer
```

#### 构造函数

```python
GRPCComputeServer(
    compute_fn: ComputeFunction,
    host: str = "0.0.0.0",
    port: int = 50051,
    max_workers: int = 10,
    codec: Optional[TensorCodec] = None,
    version: str = "1.0.0"
)
```

**参数:**
- `compute_fn` (ComputeFunction): 要提供的计算函数
- `host` (str): 服务器主机地址。默认: "0.0.0.0"
- `port` (int): 服务器端口。默认: 50051
- `max_workers` (int): 线程池大小。默认: 10
- `codec` (TensorCodec, optional): 自定义张量编解码器。默认: TensorCodec()
- `version` (str): 服务版本。默认: "1.0.0"

**示例:**
```python
server = GRPCComputeServer(
    compute_fn=my_function,
    host="0.0.0.0",
    port=50051,
    max_workers=20
)
```

#### 方法

##### `start()`

启动 gRPC 服务器。

**示例:**
```python
server.start()
print("Server started")
```

##### `wait_for_termination()`

阻塞直到服务器停止。

**示例:**
```python
server.start()
server.wait_for_termination()  # 在此处阻塞
```

##### `stop(grace: Optional[float] = 5.0)`

优雅地停止服务器。

**参数:**
- `grace` (float, optional): 宽限期（秒）。默认: 5.0

**示例:**
```python
server.stop(grace=10.0)
```

##### 上下文管理器支持

```python
with GRPCComputeServer(compute_fn, port=50051) as server:
    # 服务器自动启动和停止
    pass
```

---

### ComputeServicer

低级 gRPC servicer 实现（高级用法）。

```python
from splitlearn_comm.server import ComputeServicer
```

大多数用户应该使用 `GRPCComputeServer`。有关详细信息，请参阅源代码。

---

## 客户端 (Client)

### GRPCComputeClient

用于连接到计算服务器的 gRPC 客户端。

```python
from splitlearn_comm import GRPCComputeClient
```

#### 构造函数

```python
GRPCComputeClient(
    server_address: str,
    timeout: float = 30.0,
    retry_strategy: Optional[RetryStrategy] = None,
    codec: Optional[TensorCodec] = None
)
```

**参数:**
- `server_address` (str): 格式为 "host:port" 的服务器地址
- `timeout` (float): 请求超时（秒）。默认: 30.0
- `retry_strategy` (RetryStrategy, optional): 重试策略。默认: ExponentialBackoff()
- `codec` (TensorCodec, optional): 自定义张量编解码器。默认: TensorCodec()

**示例:**
```python
client = GRPCComputeClient(
    server_address="localhost:50051",
    timeout=60.0
)
```

#### 方法

##### `connect() -> bool`

建立与服务器的连接。

**返回:**
- bool: 如果连接成功则为 True，否则为 False

**示例:**
```python
if client.connect():
    print("Connected!")
else:
    print("Connection failed")
```

##### `compute(input_tensor: torch.Tensor) -> torch.Tensor`

执行远程计算。

**参数:**
- `input_tensor` (torch.Tensor): 输入张量

**返回:**
- torch.Tensor: 计算后的输出张量

**引发:**
- Exception: 如果所有重试后计算失败

**示例:**
```python
input_tensor = torch.randn(1, 10, 768)
output_tensor = client.compute(input_tensor)
```

##### `health_check() -> bool`

检查服务器是否健康。

**返回:**
- bool: 如果服务器健康则为 True

**示例:**
```python
if client.health_check():
    print("Server is healthy")
```

##### `get_service_info() -> Dict[str, Any]`

获取有关远程服务的信息。

**返回:**
- dict: 服务信息，包括：
  - `service_name`: 服务名称
  - `version`: 服务版本
  - `uptime_seconds`: 服务器运行时间
  - `total_requests`: 处理的总请求数
  - `device`: 计算设备
  - `custom_info`: 额外的自定义信息

**示例:**
```python
info = client.get_service_info()
print(f"Service: {info['service_name']}")
print(f"Device: {info['device']}")
```

##### `get_statistics() -> Dict[str, Any]`

获取客户端统计信息。

**返回:**
- dict: 统计信息，包括：
  - `total_requests`: 发出的总请求数
  - `successful_requests`: 成功的请求数
  - `failed_requests`: 失败的请求数
  - `avg_network_time_ms`: 平均网络时间（毫秒）
  - `avg_compute_time_ms`: 平均服务器计算时间（毫秒）
  - `avg_total_time_ms`: 平均总时间（毫秒）

**示例:**
```python
stats = client.get_statistics()
print(f"Requests: {stats['total_requests']}")
print(f"Avg time: {stats['avg_total_time_ms']:.2f}ms")
```

##### `close()`

关闭连接。

**示例:**
```python
client.close()
```

##### 上下文管理器支持

```python
with GRPCComputeClient("localhost:50051") as client:
    output = client.compute(input_tensor)
# 连接自动关闭
```

---

## 重试策略 (Retry Strategies)

### RetryStrategy

重试策略的抽象基类。

```python
from splitlearn_comm.client import RetryStrategy
```

#### 方法

##### `execute(func: Callable, *args, **kwargs) -> Any`

执行带有重试逻辑的函数。

**参数:**
- `func` (Callable): 要执行的函数
- `*args`: func 的位置参数
- `**kwargs`: func 的关键字参数

**返回:**
- Any: func 的返回值

---

### ExponentialBackoff

带有指数退避和抖动的重试策略。

```python
from splitlearn_comm import ExponentialBackoff
```

#### 构造函数

```python
ExponentialBackoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.25
)
```

**参数:**
- `max_retries` (int): 最大重试次数。默认: 3
- `initial_delay` (float): 初始延迟（秒）。默认: 1.0
- `max_delay` (float): 最大延迟（秒）。默认: 30.0
- `jitter` (float): 抖动因子 (0-1)。默认: 0.25

**示例:**
```python
retry = ExponentialBackoff(
    max_retries=5,
    initial_delay=2.0,
    max_delay=60.0
)

client = GRPCComputeClient(
    "localhost:50051",
    retry_strategy=retry
)
```

**行为:**
- 延迟 = min(initial_delay * 2^attempt, max_delay)
- 实际延迟 = delay * (1 ± jitter * random())

---

### FixedDelay

尝试之间具有固定延迟的重试策略。

```python
from splitlearn_comm import FixedDelay
```

#### 构造函数

```python
FixedDelay(
    max_retries: int = 3,
    delay: float = 1.0
)
```

**参数:**
- `max_retries` (int): 最大重试次数。默认: 3
- `delay` (float): 固定延迟（秒）。默认: 1.0

**示例:**
```python
retry = FixedDelay(max_retries=5, delay=2.0)

client = GRPCComputeClient(
    "localhost:50051",
    retry_strategy=retry
)
```

---

## 高级用法

### 自定义计算函数

```python
from splitlearn_comm.core import ComputeFunction
import torch

class MyCustomFunction(ComputeFunction):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def setup(self):
        print("Initializing...")
        self.cache = {}

    def compute(self, input_tensor):
        # 自定义逻辑
        result = input_tensor * self.param1 + self.param2
        return result

    def get_info(self):
        return {
            "name": "MyCustomFunction",
            "param1": self.param1,
            "param2": self.param2
        }

    def teardown(self):
        print("Cleaning up...")
        self.cache.clear()

# 使用它
compute_fn = MyCustomFunction(param1=2.0, param2=1.0)
server = GRPCComputeServer(compute_fn, port=50051)
```

### 自定义重试策略

```python
from splitlearn_comm.client import RetryStrategy
import time

class CustomRetry(RetryStrategy):
    def __init__(self, max_retries=3):
        self.max_retries = max_retries

    def execute(self, func, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1.0)  # 自定义延迟逻辑

client = GRPCComputeClient(
    "localhost:50051",
    retry_strategy=CustomRetry(max_retries=5)
)
```

---

## 错误处理

### 常见异常

- `grpc.RpcError`: 网络或 RPC 错误
- `ValueError`: 无效的输入形状或数据
- `RuntimeError`: 计算错误

### 错误处理示例

```python
try:
    output = client.compute(input_tensor)
except grpc.RpcError as e:
    print(f"RPC error: {e.code()}, {e.details()}")
except ValueError as e:
    print(f"Invalid input: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## 性能提示

1. **使用二进制序列化**: 默认的 TensorCodec 已经过优化
2. **为 WAN 启用压缩**: 对远程服务器使用 CompressedTensorCodec
3. **调整线程池**: 根据 CPU 核心数调整 `max_workers`
4. **批量请求**: 发送更大的批次而不是许多小请求
5. **调整超时**: 根据预期的计算时间设置适当的超时
6. **监控统计信息**: 使用 `get_statistics()` 识别瓶颈

---

## 另请参阅

- [协议文档](protocol.md)
- [扩展指南](extending.md)
- [示例](../examples/)
