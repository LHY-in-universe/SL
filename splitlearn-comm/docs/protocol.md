# 协议文档

本文档描述了 splitlearn-comm 使用的 gRPC 协议。

## 目录

- [概览](#overview)
- [Protocol Buffer 定义](#protocol-buffer-definition)
- [消息类型](#message-types)
- [RPC 方法](#rpc-methods)
- [张量序列化](#tensor-serialization)
- [错误处理](#error-handling)

---

## 概览

splitlearn-comm 使用 gRPC 进行客户端和服务器之间的通信。协议使用 Protocol Buffers (protobuf) 版本 3 定义。

**主要特性:**
- 用于高性能的二进制张量序列化
- 支持形状和 dtype 信息的元数据
- 内置健康检查
- 通过 GetServiceInfo 进行服务发现
- 结构化错误报告

---

## Protocol Buffer 定义

### 服务定义

```protobuf
service ComputeService {
    // 对输入张量执行计算
    rpc Compute(ComputeRequest) returns (ComputeResponse);

    // 检查服务健康状况
    rpc HealthCheck(HealthRequest) returns (HealthResponse);

    // 获取服务信息
    rpc GetServiceInfo(ServiceInfoRequest) returns (ServiceInfoResponse);
}
```

### 完整 Proto 文件

位置: `src/splitlearn_comm/protocol/protos/compute_service.proto`

```protobuf
syntax = "proto3";

package compute_service;

// ============================================================================
// Compute RPC
// ============================================================================

message TensorData {
    bytes data = 1;              // 二进制张量数据 (float32)
    repeated int64 shape = 2;    // 张量形状
    string dtype = 3;            // 数据类型 (例如 "float32")
}

message ComputeRequest {
    TensorData input = 1;        // 输入张量
    string request_id = 2;       // 可选请求 ID 用于追踪
}

message ComputeResponse {
    TensorData output = 1;       // 输出张量
    double compute_time_ms = 2;  // 服务端计算时间
    bool success = 3;            // 成功标志
    string error_message = 4;    // 失败时的错误消息
}

// ============================================================================
// Health Check RPC
// ============================================================================

message HealthRequest {
    // 目前为空
}

message HealthResponse {
    bool healthy = 1;            // 服务器健康状态
    string message = 2;          // 可选状态消息
}

// ============================================================================
// Service Info RPC
// ============================================================================

message ServiceInfoRequest {
    // 目前为空
}

message ServiceInfoResponse {
    string service_name = 1;     // 服务名称
    string version = 2;          // 服务版本
    string device = 3;           // 计算设备 (cpu/cuda)
    double uptime_seconds = 4;   // 服务器运行时间
    int64 total_requests = 5;    // 处理的总请求数
    map<string, string> custom_info = 6;  // 来自 ComputeFunction 的自定义信息
}
```

---

## 消息类型

### TensorData

表示带有元数据的序列化张量。

**字段:**
- `data` (bytes): IEEE 754 float32 格式的二进制张量数据
- `shape` (repeated int64): 张量维度，例如 [2, 3, 768]
- `dtype` (string): 数据类型标识符，目前始终为 "float32"

**序列化格式:**
```
data = numpy_array.astype(np.float32).tobytes()
```

**大小计算:**
```
size_bytes = 4 * product(shape)  // 每个 float32 4 字节
```

**示例:**
```python
# Tensor: shape [2, 3, 4], dtype float32
TensorData {
    data: <96 bytes>        // 2*3*4*4 = 96 字节
    shape: [2, 3, 4]
    dtype: "float32"
}
```

---

### ComputeRequest

客户端计算请求。

**字段:**
- `input` (TensorData): 要处理的输入张量
- `request_id` (string): 用于日志/追踪的可选请求标识符

**示例:**
```python
ComputeRequest {
    input: TensorData {
        data: <binary_data>
        shape: [1, 10, 768]
        dtype: "float32"
    }
    request_id: "req-12345"  # 可选
}
```

---

### ComputeResponse

带有计算结果的服务器响应。

**字段:**
- `output` (TensorData): 计算后的输出张量
- `compute_time_ms` (double): 毫秒级的服务端计算时间
- `success` (bool): 计算是否成功
- `error_message` (string): 如果 success=false 的错误描述

**成功示例:**
```python
ComputeResponse {
    output: TensorData { ... }
    compute_time_ms: 15.3
    success: true
    error_message: ""
}
```

**错误示例:**
```python
ComputeResponse {
    output: TensorData { data: b"", shape: [], dtype: "" }
    compute_time_ms: 0.0
    success: false
    error_message: "CUDA out of memory"
}
```

---

### HealthRequest / HealthResponse

简单的健康检查机制。

**HealthRequest:** 空消息

**HealthResponse 字段:**
- `healthy` (bool): 服务器健康状态
- `message` (string): 可选状态消息

**示例:**
```python
HealthResponse {
    healthy: true
    message: "All systems operational"
}
```

---

### ServiceInfoRequest / ServiceInfoResponse

服务发现和元数据。

**ServiceInfoRequest:** 空消息

**ServiceInfoResponse 字段:**
- `service_name` (string): 来自 ComputeFunction.get_info() 的名称
- `version` (string): 服务版本
- `device` (string): 计算设备 (cpu/cuda/cuda:0/etc)
- `uptime_seconds` (double): 服务器运行时间（秒）
- `total_requests` (int64): 启动以来处理的总请求数
- `custom_info` (map<string, string>): 来自 ComputeFunction.get_info() 的额外键值对

**示例:**
```python
ServiceInfoResponse {
    service_name: "TransformerLayer"
    version: "1.0.0"
    device: "cuda:0"
    uptime_seconds: 3600.5
    total_requests: 12458
    custom_info: {
        "model_params": "1234567",
        "batch_size": "32"
    }
}
```

---

## RPC 方法

### Compute

对输入张量执行计算。

**方法签名:**
```protobuf
rpc Compute(ComputeRequest) returns (ComputeResponse);
```

**流程:**
1. 客户端使用 TensorCodec 序列化张量
2. 客户端发送 ComputeRequest
3. 服务器反序列化张量
4. 服务器调用 ComputeFunction.compute()
5. 服务器序列化输出张量
6. 服务器发送带有计时信息的 ComputeResponse

**计时:**
- `compute_time_ms`: 仅服务端计算（不包括网络）
- 客户端跟踪包括网络开销在内的总时间

**错误处理:**
- 计算错误: success=false, error_message 已填充
- 网络错误: 客户端引发 grpc.RpcError

---

### HealthCheck

检查服务器是否正常运行。

**方法签名:**
```protobuf
rpc HealthCheck(HealthRequest) returns (HealthResponse);
```

**用法:**
- 客户端用于验证连接性
- 快速、轻量级检查
- 始终立即返回

**示例用例:**
```python
if not client.health_check():
    # 重连或优雅失败
    client.reconnect()
```

---

### GetServiceInfo

检索服务元数据和统计信息。

**方法签名:**
```protobuf
rpc GetServiceInfo(ServiceInfoRequest) returns (ServiceInfoResponse);
```

**用法:**
- 服务发现
- 版本兼容性检查
- 性能监控
- 来自 ComputeFunction 的自定义元数据

**示例用例:**
```python
info = client.get_service_info()
if info['version'] != expected_version:
    raise VersionMismatchError()
```

---

## 张量序列化

### 格式

splitlearn-comm 使用 **二进制序列化** 以获得高性能：

1. **转换为 numpy:** `tensor.cpu().numpy()`
2. **转换为 float32:** `.astype(np.float32)`
3. **序列化为字节:** `.tobytes()`
4. **提取形状:** `tuple(tensor.shape)`

### 反序列化

1. **创建 numpy 数组:** `np.frombuffer(data, dtype=np.float32)`
2. **重塑:** `.reshape(shape)`
3. **转换为 torch:** `torch.from_numpy(array)`

### 性能特征

| 方法 | 速度 | 开销 |
|--------|-------|----------|
| **二进制 (bytes)** | **1x (基准)** | **~0%** |
| Protobuf repeated float | 慢 4 倍 | ~15% |
| JSON | 慢 10 倍 | ~40% |

### 示例代码

```python
# 编码
def encode(tensor):
    array = tensor.cpu().numpy().astype(np.float32)
    data = array.tobytes()
    shape = tuple(tensor.shape)
    return data, shape

# 解码
def decode(data, shape):
    array = np.frombuffer(data, dtype=np.float32).reshape(shape)
    tensor = torch.from_numpy(array)
    return tensor
```

### 压缩

对于 WAN 场景，可选 zlib 压缩：

```python
# 压缩编码
data_compressed = zlib.compress(data, level=6)

# 压缩解码
data = zlib.decompress(data_compressed)
```

**权衡:**
- **LAN**: 压缩开销 > 带宽节省 → 不使用
- **WAN**: 压缩开销 < 带宽节省 → 使用

---

## 错误处理

### 错误类型

#### 1. 计算错误

ComputeFunction.compute() 执行期间的错误。

**响应:**
```python
ComputeResponse {
    success: false
    error_message: "详细错误描述"
}
```

**客户端行为:**
- 不重试（计算逻辑错误）
- 向调用者引发异常

#### 2. 网络错误

连接失败、超时等。

**gRPC 状态码:**
- `UNAVAILABLE`: 服务器不可达
- `DEADLINE_EXCEEDED`: 请求超时
- `CANCELLED`: 请求已取消
- `UNKNOWN`: 未知错误

**客户端行为:**
- 自动重试并退避（如果配置了 RetryStrategy）
- 如果所有重试耗尽，最终引发 grpc.RpcError

#### 3. 序列化错误

无效的张量格式或形状。

**服务器响应:**
```python
ComputeResponse {
    success: false
    error_message: "Invalid tensor shape: expected 3D, got 2D"
}
```

### 错误处理示例

```python
try:
    output = client.compute(input_tensor)
except grpc.RpcError as e:
    if e.code() == grpc.StatusCode.UNAVAILABLE:
        print("Server unavailable")
    elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
        print("Request timeout")
    else:
        print(f"RPC error: {e.details()}")
except Exception as e:
    print(f"Computation error: {e}")
```

---

## 协议扩展

### 添加自定义字段

要使用自定义字段扩展协议：

1. **修改 proto 文件:**
```protobuf
message ComputeRequest {
    TensorData input = 1;
    string request_id = 2;

    // 自定义扩展
    map<string, string> metadata = 10;  // 使用 >= 10 的字段号
}
```

2. **重新编译:**
```bash
bash scripts/compile_proto.sh
```

3. **更新客户端/服务器:**
```python
# 客户端
request.metadata["custom_key"] = "custom_value"

# 服务器
metadata = request.metadata
```

### 向后兼容性

Protocol Buffers 确保向后兼容性：
- 未知字段被忽略
- 可选字段默认为零/空
- 新字段不会破坏旧的客户端/服务器

---

## 最佳实践

1. **使用 request_id 进行追踪:**
   ```python
   request.request_id = f"req-{uuid.uuid4()}"
   ```

2. **监控 compute_time_ms:**
   ```python
   if response.compute_time_ms > threshold:
       logger.warning(f"Slow computation: {response.compute_time_ms}ms")
   ```

3. **检查 success 标志:**
   ```python
   if not response.success:
       raise ComputationError(response.error_message)
   ```

4. **填充 custom_info:**
   ```python
   def get_info(self):
       return {
           "model_name": "GPT2",
           "num_params": str(self.num_params),
           "gpu_memory_mb": str(torch.cuda.memory_allocated() // 1024**2)
       }
   ```

---

## 另请参阅

- [API 参考](api.md)
- [扩展指南](extending.md)
- [Protocol Buffer 文档](https://protobuf.dev/)
- [gRPC 文档](https://grpc.io/docs/)
