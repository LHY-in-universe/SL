# SplitLearnComm 库主要功能

## 概述

**SplitLearnComm** 是一个基于 gRPC 的分布式深度学习通信库，用于在不同机器之间传输和计算张量数据。

## 核心功能

### 1. **gRPC 服务器（Server）**

提供两种服务器实现：

#### 同步服务器：`GRPCComputeServer`
- 使用 `ThreadPoolExecutor` 处理并发请求
- 适合 CPU 密集型任务
- 支持多线程并发处理

#### 异步服务器：`AsyncGRPCComputeServer`
- 使用 `grpc.aio` 和 `asyncio`
- 适合 I/O 密集型任务
- 不需要线程池，使用协程

**功能**：
- 监听客户端连接
- 接收计算请求
- 执行计算并返回结果
- 健康检查
- 服务信息查询

### 2. **gRPC 客户端（Client）**

`GRPCComputeClient` 提供：
- 连接到远程服务器
- 发送计算请求
- 接收计算结果
- 自动重试机制（指数退避）
- 超时控制
- 统计信息收集

### 3. **计算函数抽象（ComputeFunction）**

完全解耦模型依赖，支持任意计算逻辑：

#### `ComputeFunction`（抽象基类）
- 定义 `compute(input_tensor)` 接口
- 允许用户实现自定义计算逻辑

#### `ModelComputeFunction`（PyTorch 模型包装器）
- 将 PyTorch 模型包装为计算函数
- 自动处理设备转换（CPU/GPU）
- 支持 `torch.no_grad()` 模式

#### 异步版本：
- `AsyncComputeFunction`
- `AsyncModelComputeFunction`
- 支持异步计算

### 4. **张量编解码（TensorCodec）**

高效的张量序列化：

#### `TensorCodec`
- 二进制格式序列化（比 protobuf 快 4 倍）
- 零拷贝优化（numpy/torch 转换）
- 支持不同数据类型（float32, int64 等）

#### `CompressedTensorCodec`
- 可选的 zlib 压缩
- 适合带宽受限场景（WAN）

### 5. **重试策略（Retry Strategy）**

内置重试机制：

- `ExponentialBackoff`：指数退避
- `FixedDelay`：固定延迟
- 可自定义重试策略

### 6. **监控和统计（Monitoring）**

- **LogManager**：日志管理
- **MetricsManager**：性能指标收集
  - 请求计数
  - 延迟统计
  - 成功率
  - 计算时间

### 7. **用户界面（UI）**

可选的可视化界面（需要 gradio）：

- **ClientUI**：客户端界面
- **ServerMonitoringUI**：服务器监控界面

### 8. **协议定义（Protocol）**

基于 Protocol Buffers 的 gRPC 服务定义：

- `ComputeRequest`：计算请求
- `ComputeResponse`：计算响应
- `HealthRequest/Response`：健康检查
- `ServiceInfoRequest/Response`：服务信息

## 主要使用场景

### 1. **分布式模型推理**

将模型部署在服务器上，客户端发送输入数据，服务器返回推理结果。

### 2. **Split Learning（分割学习）**

在 Split Learning 场景中：
- **Bottom Model**：在客户端运行
- **Trunk Model**：在服务器运行（使用 SplitLearnComm）
- **Top Model**：在客户端运行

### 3. **模型服务化**

将 PyTorch 模型包装为 gRPC 服务，支持远程调用。

## 架构特点

### 1. **模型无关（Model Agnostic）**

通过 `ComputeFunction` 抽象，完全解耦模型依赖：
- 可以包装任意 PyTorch 模型
- 可以自定义计算逻辑
- 不依赖特定的模型架构

### 2. **高性能**

- 二进制张量序列化（比 protobuf 快 4 倍）
- 零拷贝优化
- 可选的压缩支持

### 3. **可靠性**

- 自动重试机制
- 健康检查
- 错误处理
- 超时控制

### 4. **灵活性**

- 支持同步和异步两种模式
- 可自定义重试策略
- 可自定义编解码器
- 支持单机、LAN、WAN 部署

## 核心模块结构

```
splitlearn_comm/
├── server/          # 服务器实现
│   ├── grpc_server.py          # 同步服务器
│   ├── async_grpc_server.py   # 异步服务器
│   ├── servicer.py             # 同步 Servicer
│   └── async_servicer.py       # 异步 Servicer
├── client/          # 客户端实现
│   ├── grpc_client.py          # gRPC 客户端
│   └── retry.py                # 重试策略
├── core/            # 核心抽象
│   ├── compute_function.py     # 计算函数接口
│   ├── async_compute_function.py  # 异步计算函数
│   └── tensor_codec.py        # 张量编解码
├── protocol/        # 协议定义
│   └── compute_service.proto   # gRPC 服务定义
├── monitoring/      # 监控功能
│   ├── log_manager.py          # 日志管理
│   └── metrics_manager.py      # 指标收集
└── ui/              # 用户界面（可选）
    ├── client_ui.py            # 客户端 UI
    └── server_ui.py            # 服务器 UI
```

## 快速使用示例

### 服务器端

```python
from splitlearn_comm import GRPCComputeServer
from splitlearn_comm.core import ModelComputeFunction

# 创建计算函数
compute_fn = ModelComputeFunction(model, device="cuda")

# 启动服务器
server = GRPCComputeServer(
    compute_fn=compute_fn,
    host="0.0.0.0",
    port=50051,
    max_workers=1  # 单线程模式
)

server.start()
server.wait_for_termination()
```

### 客户端

```python
from splitlearn_comm import GRPCComputeClient

# 连接服务器
client = GRPCComputeClient("localhost:50051")
client.connect()

# 发送计算请求
output = client.compute(input_tensor)

# 关闭连接
client.close()
```

## 总结

**SplitLearnComm** 是一个专门为分布式深度学习设计的通信库，主要功能包括：

1. ✅ **gRPC 服务器和客户端**：提供远程计算服务
2. ✅ **计算函数抽象**：模型无关的设计
3. ✅ **高效张量编解码**：优化的序列化
4. ✅ **重试和错误处理**：提高可靠性
5. ✅ **监控和统计**：性能追踪
6. ✅ **同步和异步支持**：灵活的部署方式

它是 Split Learning 框架中的通信层，负责在不同组件之间传输张量数据和执行远程计算。

