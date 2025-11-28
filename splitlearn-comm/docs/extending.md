# 扩展 splitlearn-comm

本指南向您展示如何针对特定用例自定义和扩展 splitlearn-comm。

## 目录

- [自定义计算函数](#custom-compute-functions)
- [自定义重试策略](#custom-retry-strategies)
- [自定义张量编解码器](#custom-tensor-codecs)
- [高级服务器配置](#advanced-server-configuration)
- [中间件和拦截器](#middleware-and-interceptors)
- [监控和日志记录](#monitoring-and-logging)

---

## 自定义计算函数

`ComputeFunction` 抽象允许您实现任何计算逻辑，而不仅仅是模型推理。

### 基本自定义函数

```python
from splitlearn_comm.core import ComputeFunction
import torch

class DataAugmentation(ComputeFunction):
    """应用数据增强变换。"""

    def __init__(self, augmentation_prob=0.5):
        self.augmentation_prob = augmentation_prob

    def compute(self, input_tensor):
        # 应用随机增强
        if torch.rand(1).item() < self.augmentation_prob:
            # 随机水平翻转
            input_tensor = torch.flip(input_tensor, dims=[-1])

        # 添加噪声
        noise = torch.randn_like(input_tensor) * 0.01
        return input_tensor + noise

    def get_info(self):
        return {
            "name": "DataAugmentation",
            "augmentation_prob": str(self.augmentation_prob)
        }
```

### 有状态的计算函数

```python
class CachingFunction(ComputeFunction):
    """带有缓存的计算函数。"""

    def __init__(self, model, cache_size=100):
        self.model = model
        self.cache_size = cache_size
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def setup(self):
        """服务器启动时调用。"""
        print(f"Initializing cache with size {self.cache_size}")
        self.model.eval()

    def compute(self, input_tensor):
        # 从张量哈希创建缓存键
        cache_key = hash(input_tensor.cpu().numpy().tobytes())

        # 检查缓存
        if cache_key in self.cache:
            self.hits += 1
            return self.cache[cache_key]

        # 缓存未命中 - 计算
        self.misses += 1
        with torch.no_grad():
            output = self.model(input_tensor)

        # 更新缓存 (LRU 风格)
        if len(self.cache) >= self.cache_size:
            # 移除最旧的条目
            self.cache.pop(next(iter(self.cache)))

        self.cache[cache_key] = output
        return output

    def get_info(self):
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        return {
            "name": "CachingFunction",
            "cache_size": str(self.cache_size),
            "cache_hit_rate": f"{hit_rate:.2%}",
            "cache_hits": str(self.hits),
            "cache_misses": str(self.misses)
        }

    def teardown(self):
        """服务器停止时调用。"""
        print(f"Cache stats - Hits: {self.hits}, Misses: {self.misses}")
        self.cache.clear()
```

### 多模型集成

```python
class EnsembleFunction(ComputeFunction):
    """多个模型的集成。"""

    def __init__(self, models, weights=None, device="cpu"):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.device = device

    def setup(self):
        # 将所有模型移动到设备并设置为 eval 模式
        for model in self.models:
            model.to(self.device)
            model.eval()

    def compute(self, input_tensor):
        input_tensor = input_tensor.to(self.device)

        # 计算模型输出的加权平均值
        outputs = []
        with torch.no_grad():
            for model, weight in zip(self.models, self.weights):
                output = model(input_tensor)
                outputs.append(output * weight)

        ensemble_output = torch.stack(outputs).sum(dim=0)
        return ensemble_output

    def get_info(self):
        return {
            "name": "EnsembleFunction",
            "num_models": str(len(self.models)),
            "device": self.device,
            "weights": str(self.weights)
        }
```

### 预处理管道

```python
class PreprocessingPipeline(ComputeFunction):
    """多阶段预处理管道。"""

    def __init__(self, stages):
        """
        Args:
            stages: (name, transform_fn) 元组列表
        """
        self.stages = stages

    def compute(self, input_tensor):
        x = input_tensor
        for stage_name, transform_fn in self.stages:
            x = transform_fn(x)
        return x

    def get_info(self):
        stage_names = [name for name, _ in self.stages]
        return {
            "name": "PreprocessingPipeline",
            "stages": ", ".join(stage_names),
            "num_stages": str(len(self.stages))
        }

# 用法
pipeline = PreprocessingPipeline([
    ("normalize", lambda x: (x - x.mean()) / x.std()),
    ("clamp", lambda x: torch.clamp(x, -3, 3)),
    ("scale", lambda x: x * 0.5)
])
```

---

## 自定义重试策略

针对不同的失败场景实现自定义重试逻辑。

### 自适应重试

```python
from splitlearn_comm.client import RetryStrategy
import time
import logging

class AdaptiveRetry(RetryStrategy):
    """根据错误类型进行适应的重试策略。"""

    def __init__(self, max_retries=5, base_delay=1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.logger = logging.getLogger(__name__)

    def execute(self, func, *args, **kwargs):
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)

            except grpc.RpcError as e:
                last_exception = e

                # 根据错误代码采取不同的策略
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    # 服务器宕机 - 更长的退避
                    delay = self.base_delay * (3 ** attempt)
                elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    # 超时 - 适度退避
                    delay = self.base_delay * (2 ** attempt)
                elif e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                    # 服务器过载 - 指数退避
                    delay = self.base_delay * (2 ** attempt)
                else:
                    # 其他错误 - 不重试
                    raise

                if attempt < self.max_retries - 1:
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed with {e.code()}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

        # 所有重试耗尽
        raise last_exception
```

### 熔断器 (Circuit Breaker)

```python
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"      # 正常运行
    OPEN = "open"          # 失败中，拒绝请求
    HALF_OPEN = "half_open"  # 测试是否恢复

class CircuitBreakerRetry(RetryStrategy):
    """用于容错的熔断器模式。"""

    def __init__(
        self,
        failure_threshold=5,
        timeout=60.0,
        half_open_attempts=3
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_attempts = half_open_attempts

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0

    def execute(self, func, *args, **kwargs):
        # 检查电路是否应转换为 HALF_OPEN
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)

            # 成功
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.half_open_attempts:
                    # 已恢复！
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0

            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

            raise
```

---

## 自定义张量编解码器

为专用数据类型实现自定义序列化。

### 量化张量编解码器

```python
from splitlearn_comm.core import TensorCodec
import torch
import numpy as np

class QuantizedTensorCodec(TensorCodec):
    """具有 8 位量化的编解码器，用于减少带宽。"""

    def encode(self, tensor):
        # 转换为 numpy
        array = tensor.cpu().numpy()

        # 量化为 int8
        min_val = array.min()
        max_val = array.max()
        scale = (max_val - min_val) / 255.0

        if scale == 0:
            scale = 1.0

        quantized = ((array - min_val) / scale).astype(np.uint8)

        # 序列化量化数据 + scale/offset
        data = quantized.tobytes()
        metadata = {
            'min': min_val,
            'max': max_val,
            'scale': scale
        }

        shape = tuple(tensor.shape)
        return data, shape, metadata

    def decode(self, data, shape, metadata):
        # 反序列化
        quantized = np.frombuffer(data, dtype=np.uint8).reshape(shape)

        # 反量化
        array = quantized.astype(np.float32) * metadata['scale'] + metadata['min']

        tensor = torch.from_numpy(array)
        return tensor
```

### 稀疏张量编解码器

```python
class SparseTensorCodec(TensorCodec):
    """用于稀疏张量 (COO 格式) 的编解码器。"""

    def encode(self, tensor):
        # 转换为稀疏 COO 格式
        sparse = tensor.to_sparse_coo()

        # 分别序列化索引和值
        indices = sparse.indices().cpu().numpy()
        values = sparse.values().cpu().numpy().astype(np.float32)

        indices_bytes = indices.tobytes()
        values_bytes = values.tobytes()

        # 用分隔符组合
        data = (
            len(indices_bytes).to_bytes(4, 'little') +
            indices_bytes +
            values_bytes
        )

        shape = tuple(tensor.shape)
        return data, shape

    def decode(self, data, shape):
        # 提取长度
        indices_len = int.from_bytes(data[:4], 'little')

        # 提取索引和值
        indices_bytes = data[4:4 + indices_len]
        values_bytes = data[4 + indices_len:]

        indices = np.frombuffer(indices_bytes, dtype=np.int64)
        values = np.frombuffer(values_bytes, dtype=np.float32)

        # 重建稀疏张量
        indices_tensor = torch.from_numpy(indices).view(-1, len(shape))
        values_tensor = torch.from_numpy(values)

        sparse = torch.sparse_coo_tensor(
            indices_tensor.t(),
            values_tensor,
            shape
        )

        return sparse.to_dense()
```

---

## 高级服务器配置

### 自定义 gRPC 选项

```python
from splitlearn_comm import GRPCComputeServer

server = GRPCComputeServer(
    compute_fn=compute_fn,
    port=50051,
    max_workers=20,
    options=[
        # 最大消息大小 (100MB)
        ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),

        # Keepalive 设置
        ('grpc.keepalive_time_ms', 30000),
        ('grpc.keepalive_timeout_ms', 10000),
        ('grpc.keepalive_permit_without_calls', True),

        # 性能调优
        ('grpc.http2.min_ping_interval_without_data_ms', 5000),
        ('grpc.http2.max_pings_without_data', 0),
    ]
)
```

### 多端口服务器

```python
class MultiPortServer:
    """在不同端口上提供多个计算函数服务。"""

    def __init__(self, functions_and_ports):
        """
        Args:
            functions_and_ports: (compute_fn, port) 元组列表
        """
        self.servers = [
            GRPCComputeServer(fn, port=port)
            for fn, port in functions_and_ports
        ]

    def start(self):
        for server in self.servers:
            server.start()

    def wait_for_termination(self):
        # 等待所有服务器
        for server in self.servers:
            server.wait_for_termination()

    def stop(self):
        for server in self.servers:
            server.stop()

# 用法
server = MultiPortServer([
    (preprocessing_fn, 50051),
    (model_fn, 50052),
    (postprocessing_fn, 50053)
])
```

---

## 中间件和拦截器

gRPC 支持拦截器用于横切关注点。

### 日志拦截器

```python
import grpc
import logging

class LoggingInterceptor(grpc.ServerInterceptor):
    """记录所有 RPC 调用。"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def intercept_service(self, continuation, handler_call_details):
        method = handler_call_details.method
        self.logger.info(f"RPC call: {method}")

        # 继续调用
        return continuation(handler_call_details)

# 用法
server = grpc.server(
    futures.ThreadPoolExecutor(max_workers=10),
    interceptors=[LoggingInterceptor()]
)
```

### 认证拦截器

```python
class AuthInterceptor(grpc.ServerInterceptor):
    """需要认证令牌。"""

    def __init__(self, valid_tokens):
        self.valid_tokens = valid_tokens

    def intercept_service(self, continuation, handler_call_details):
        # 提取元数据
        metadata = dict(handler_call_details.invocation_metadata)
        token = metadata.get('authorization', '')

        if token not in self.valid_tokens:
            return grpc.unary_unary_rpc_method_handler(
                lambda request, context: context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    'Invalid token'
                )
            )

        return continuation(handler_call_details)

# 客户端用法
client = GRPCComputeClient("localhost:50051")
client.channel = grpc.insecure_channel(
    "localhost:50051",
    options=[('grpc.default_authority', 'localhost')]
)

# 向调用添加认证元数据
metadata = [('authorization', 'secret-token')]
client.stub.Compute(request, metadata=metadata)
```

---

## 监控和日志记录

### Prometheus 指标

```python
from prometheus_client import Counter, Histogram, start_http_server

class MonitoredComputeFunction(ComputeFunction):
    """带有 Prometheus 指标的计算函数。"""

    def __init__(self, model):
        self.model = model

        # 定义指标
        self.request_counter = Counter(
            'compute_requests_total',
            'Total compute requests'
        )

        self.compute_duration = Histogram(
            'compute_duration_seconds',
            'Compute duration in seconds'
        )

        self.error_counter = Counter(
            'compute_errors_total',
            'Total compute errors'
        )

    def setup(self):
        # 启动 Prometheus HTTP 服务器
        start_http_server(8000)
        self.model.eval()

    def compute(self, input_tensor):
        self.request_counter.inc()

        with self.compute_duration.time():
            try:
                with torch.no_grad():
                    output = self.model(input_tensor)
                return output
            except Exception as e:
                self.error_counter.inc()
                raise
```

### 结构化日志

```python
import logging
import json
import time

class StructuredLogger:
    """JSON 结构化日志。"""

    def __init__(self, name):
        self.logger = logging.getLogger(name)

    def log_request(self, request_id, input_shape, output_shape, duration_ms, success):
        log_entry = {
            "timestamp": time.time(),
            "request_id": request_id,
            "input_shape": list(input_shape),
            "output_shape": list(output_shape),
            "duration_ms": duration_ms,
            "success": success
        }
        self.logger.info(json.dumps(log_entry))

class LoggedComputeFunction(ComputeFunction):
    """带有结构化日志的计算函数。"""

    def __init__(self, model):
        self.model = model
        self.logger = StructuredLogger(__name__)

    def compute(self, input_tensor):
        request_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            output = self.model(input_tensor)
            duration_ms = (time.time() - start_time) * 1000

            self.logger.log_request(
                request_id=request_id,
                input_shape=input_tensor.shape,
                output_shape=output.shape,
                duration_ms=duration_ms,
                success=True
            )

            return output

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            self.logger.log_request(
                request_id=request_id,
                input_shape=input_tensor.shape,
                output_shape=(),
                duration_ms=duration_ms,
                success=False
            )

            raise
```

---

## 最佳实践

1. **始终实现 get_info()**: 为调试提供有用的元数据
2. **使用 setup()/teardown()**: 正确管理资源
3. **优雅地处理错误**: 返回有意义的错误消息
4. **记录重要事件**: 在生产环境中使用结构化日志
5. **监控性能**: 跟踪指标以进行优化
6. **彻底测试**: 对自定义计算函数进行单元测试
7. **记录行为**: 为自定义类添加文档字符串

---

## 另请参阅

- [API 参考](api.md)
- [协议文档](protocol.md)
- [示例](../examples/)
