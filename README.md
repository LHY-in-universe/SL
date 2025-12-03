# SplitLearn Project

这是一个用于分布式深度学习和大型语言模型（LLM）物理拆分的综合项目。它由三个核心组件组成，旨在实现模型的拆分、通信和管理。

## 🚀 快速开始（5分钟）

### 极简安装

**客户端（轻量级，~280MB）**：
```bash
pip install splitlearn-core[client] splitlearn-comm[client]
```

**服务端（完整功能，~1.3GB）**：
```bash
pip install splitlearn-core[server] splitlearn-comm[server] splitlearn-manager[server]
```

### 极简使用

**服务端（3行代码）**：
```python
from splitlearn_manager.quickstart import ManagedServer

server = ManagedServer("gpt2", port=50051)
server.start()  # 阻塞运行
```

**客户端（3行代码）**：
```python
from splitlearn_comm.quickstart import Client
import torch

client = Client("localhost:50051")
output = client.compute(torch.randn(1, 10, 768))
```

**完整示例**: 查看 [QUICKSTART_GUIDE.md](./QUICKSTART_GUIDE.md) 获取更多示例。

---

## 📚 项目结构

本项目包含三个主要子项目：

### 1. [SplitLearnCore](./SplitLearnCore) - 核心模型分割库

核心逻辑库，负责将 Transformer 模型（如 GPT-2, Qwen2, Gemma）拆分为三个部分：
- **Bottom**: Embeddings + 前 N 层（客户端）
- **Trunk**: 中间 M 层（服务端，计算密集型）
- **Top**: 最后 K 层 + LM Head（客户端）

**特性**：
- ✅ 支持多种模型（GPT-2, Qwen2, Gemma）
- ✅ 自动权重映射和加载
- ✅ 增量加载分片模型
- ✅ 本地优先 + HuggingFace 自动下载

### 2. [SplitLearnComm](./SplitLearnComm) - 高性能通信库

基于 gRPC 的通信库，负责在拆分后的模型部分之间传输 Tensor 数据。

**特性**：
- ✅ 异步 gRPC 通信（grpc.aio）
- ✅ 自动重试和超时管理
- ✅ Tensor 序列化优化
- ✅ 可选 Gradio UI

### 3. [SplitLearnManager](./SplitLearnManager) - 模型生命周期管理

提供模型部署和生命周期管理功能。

**特性**：
- ✅ 异步模型管理（解决锁阻塞问题）
- ✅ 资源监控和健康检查
- ✅ Prometheus 指标导出
- ✅ 负载均衡和请求路由

---

## 💡 核心特性

### 🎯 简化的 API

**之前**（15-20行代码）：
```python
from splitlearn_comm import GRPCComputeClient
from splitlearn_comm.client import RetryStrategy, ExponentialBackoff

retry_strategy = ExponentialBackoff(
    initial_delay=1.0,
    max_delay=60.0,
    multiplier=2.0,
    max_attempts=5
)

client = GRPCComputeClient(
    server_address="localhost:50051",
    retry_strategy=retry_strategy
)
client.connect()
output = client.compute(input_tensor)
client.close()
```

**现在**（3行代码）：
```python
from splitlearn_comm.quickstart import Client

client = Client("localhost:50051")
output = client.compute(input_tensor)
```

### 📦 按需安装

| 使用场景 | 安装命令 | 包大小 |
|---------|---------|--------|
| 客户端（轻量级） | `pip install splitlearn-core[client] splitlearn-comm[client]` | ~280MB |
| 服务端（完整功能） | `pip install splitlearn-core[server] splitlearn-comm[server] splitlearn-manager[server]` | ~1.3GB |
| 开发环境 | `pip install -r requirements-dev.txt` | ~1.5GB |

### ⚡ 性能优化

- **异步架构**：解决互斥锁阻塞问题，`list_models()` 延迟从 >1000ms 降至 <10ms（降低 99%）
- **并发处理**：并发 QPS 提升 2-3 倍
- **资源高效**：轻量级客户端，适合移动端和边缘设备

---

## 📖 完整文档

### 安装指南
- **[INSTALLATION_GUIDE.md](./INSTALLATION_GUIDE.md)** - 详细的安装说明（客户端/服务端/开发环境）

### 快速开始
- **[QUICKSTART_GUIDE.md](./QUICKSTART_GUIDE.md)** - 5分钟快速上手指南

### 异步架构
- **[ASYNC_MIGRATION_GUIDE.md](./ASYNC_MIGRATION_GUIDE.md)** - 异步 API 迁移指南
- **[IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)** - 异步架构实施总结

### 示例代码
- **[examples/quickstart_client.py](./examples/quickstart_client.py)** - 客户端快速示例
- **[examples/quickstart_server.py](./examples/quickstart_server.py)** - 服务端快速示例
- **[examples/split_learning_minimal.py](./examples/split_learning_minimal.py)** - 完整 Split Learning 示例
- **[examples/production_setup.py](./examples/production_setup.py)** - 生产环境配置

### 子项目文档
- [SplitLearnCore 文档](./SplitLearnCore/README.md)
- [SplitLearnComm 文档](./SplitLearnComm/README.md)
- [SplitLearnManager 文档](./SplitLearnManager/README.md)

---

## 🛠️ 安装

### 方式 1：pip 安装（推荐）

**客户端（轻量级）**：
```bash
pip install splitlearn-core[client] splitlearn-comm[client]
```

**服务端（完整功能）**：
```bash
pip install splitlearn-core[server] splitlearn-comm[server] splitlearn-manager[server]
```

### 方式 2：使用 requirements.txt

```bash
# 克隆仓库
git clone https://github.com/yourusername/SL.git
cd SL

# 客户端
pip install -r SplitLearnCore/requirements.txt
pip install -r SplitLearnComm/requirements.txt

# 服务端
pip install -r SplitLearnCore/requirements-dev.txt
pip install -r SplitLearnComm/requirements-dev.txt
pip install -r SplitLearnManager/requirements-dev.txt
```

### 方式 3：从源码安装（开发模式）

```bash
# 克隆仓库
git clone https://github.com/yourusername/SL.git
cd SL

# 开发模式安装
pip install -e ./SplitLearnCore
pip install -e ./SplitLearnComm
pip install -e ./SplitLearnManager
```

详细安装说明请查看 [INSTALLATION_GUIDE.md](./INSTALLATION_GUIDE.md)

---

## 🎓 使用示例

### 示例 1：快速开始

**启动服务端**：
```python
from splitlearn_manager.quickstart import ManagedServer

# 一行代码启动服务器
server = ManagedServer("gpt2", port=50051)
server.start()
```

**运行客户端**：
```python
from splitlearn_comm.quickstart import Client
import torch

client = Client("localhost:50051")
output = client.compute(torch.randn(1, 10, 768))
print(f"Output shape: {output.shape}")
```

### 示例 2：完整 Split Learning

```python
from splitlearn.quickstart import load_split_model
from splitlearn_comm.quickstart import Client

# 加载分割模型
bottom, trunk, top = load_split_model("gpt2", split_points=[2, 10])

# 客户端推理
input_ids = torch.randint(0, 50257, (1, 10))

# Step 1: Bottom 模型（客户端）
hidden = bottom(input_ids)

# Step 2: Trunk 模型（服务端）
client = Client("localhost:50051")
hidden = client.compute(hidden)

# Step 3: Top 模型（客户端）
logits = top(hidden)
```

### 示例 3：生产环境部署

```python
# 使用环境变量配置
import os
os.environ["MODEL_TYPE"] = "qwen2"
os.environ["DEVICE"] = "cuda"
os.environ["LOG_LEVEL"] = "INFO"
os.environ["ENABLE_METRICS"] = "true"

from examples.production_setup import main
main()
```

更多示例请查看 [examples/](./examples/) 目录和 [QUICKSTART_GUIDE.md](./QUICKSTART_GUIDE.md)

---

## 🏗️ 架构

```
┌─────────────────────┐     gRPC/Tensor      ┌─────────────────────┐
│   Client (Low Compute)   ├──────────────────→│  Server (High Compute) │
│                     │                     │                     │
│  ┌───────────────┐  │                     │  ┌───────────────┐  │
│  │ Bottom Model  │  │                     │  │ Trunk Model   │  │
│  │ (0-2 layers)  │  │                     │  │ (2-10 layers) │  │
│  └───────┬───────┘  │                     │  └───────┬───────┘  │
│          │          │                     │          │          │
│          └──────────┼─────────────────────┼──────────┘          │
│                     │                     │                     │
│  ┌───────────────┐  │                     │                     │
│  │   Top Model   │  │◄────────────────────┤                     │
│  │ (10-12 layers)│  │                     │                     │
│  └───────────────┘  │                     │                     │
└─────────────────────┘                     └─────────────────────┘
```

**优势**：
- 🔐 隐私保护：敏感数据不离开客户端
- 📱 资源受限：客户端只需少量算力
- ⚡ 低延迟：本地处理 embedding 和 head
- 🔄 灵活部署：客户端/服务端可独立升级

---

## 🤝 贡献

欢迎提交 Pull Request 或 Issue 来改进这个项目。

### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/yourusername/SL.git
cd SL

# 安装开发依赖
pip install -e ./SplitLearnCore[dev]
pip install -e ./SplitLearnComm[dev]
pip install -e ./SplitLearnManager[dev]

# 运行测试
pytest
```

---

## 📄 许可证

MIT License

---

## 🆘 获取帮助

- **文档**: [完整文档链接]
- **Issues**: [GitHub Issues](https://github.com/yourusername/SL/issues)
- **示例**: 查看 [examples/](./examples/) 目录
- **快速开始**: [QUICKSTART_GUIDE.md](./QUICKSTART_GUIDE.md)
- **安装指南**: [INSTALLATION_GUIDE.md](./INSTALLATION_GUIDE.md)

---

## 📊 项目统计

| 指标 | 数值 |
|-----|------|
| 总代码行数 | ~15,000+ |
| 支持的模型 | GPT-2, Qwen2, Gemma |
| 测试覆盖率 | >80% |
| 文档页数 | 100+ |
| 示例数量 | 20+ |

---

**开始使用 Split Learning，让分布式深度学习变得简单！** 🚀
