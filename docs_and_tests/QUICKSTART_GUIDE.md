# Split Learning Quick Start Guide

本指南帮助您在 5 分钟内快速上手 Split Learning 框架。

---

## 目录

- [5分钟快速开始](#5分钟快速开始)
- [客户端示例](#客户端示例)
- [服务端示例](#服务端示例)
- [完整的 Split Learning 示例](#完整的-split-learning-示例)
- [常见使用场景](#常见使用场景)
- [常见问题](#常见问题)

---

## 5分钟快速开始

### 步骤 1: 安装

```bash
# 客户端（轻量级）
pip install splitlearn-core[client] splitlearn-comm[client]

# 服务端（完整功能）
pip install splitlearn-core[server] splitlearn-comm[server] splitlearn-manager[server]
```

### 步骤 2: 运行服务端

```python
# server.py
from splitlearn_manager.quickstart import ManagedServer

# 一行代码启动服务器！
server = ManagedServer("gpt2", port=50051)
server.start()  # 阻塞运行
```

```bash
# 运行服务端
python server.py
```

### 步骤 3: 运行客户端

```python
# client.py
from splitlearn_comm.quickstart import Client
import torch

# 连接到服务器
client = Client("localhost:50051")

# 发送推理请求
input_tensor = torch.randn(1, 10, 768)  # (batch, seq_len, hidden_size)
output = client.compute(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
```

```bash
# 运行客户端
python client.py
```

### 完成！

您已经成功运行了第一个 Split Learning 应用！

---

## 客户端示例

### 示例 1: 基础使用

```python
from splitlearn_comm.quickstart import Client
import torch

# 创建客户端（自动连接）
client = Client("localhost:50051")

# 准备输入
input_tensor = torch.randn(1, 10, 768)

# 发送计算请求
output = client.compute(input_tensor)

print(f"计算完成！输出形状: {output.shape}")

# 关闭连接
client.close()
```

### 示例 2: 使用上下文管理器

```python
from splitlearn_comm.quickstart import Client
import torch

# 使用 with 语句自动管理连接
with Client("localhost:50051") as client:
    input_tensor = torch.randn(1, 10, 768)
    output = client.compute(input_tensor)
    print(f"输出: {output.shape}")

# 连接自动关闭
```

### 示例 3: 配置重试和超时

```python
from splitlearn_comm.quickstart import Client

# 自定义配置
client = Client(
    server_address="remote-server:50051",
    max_retries=10,       # 最多重试10次
    timeout=60.0,         # 超时60秒
    auto_connect=True     # 自动连接
)

# 使用客户端...
```

---

## 服务端示例

### 示例 1: 基础服务端

```python
from splitlearn_manager.quickstart import ManagedServer

# 创建并启动服务器（阻塞）
server = ManagedServer(
    model_type="gpt2",
    component="trunk",  # 服务端通常运行 trunk
    port=50051
)

server.start()  # 阻塞，直到 Ctrl+C
```

### 示例 2: 自定义配置

```python
from splitlearn_manager.quickstart import ManagedServer

# 自定义服务器配置
server = ManagedServer(
    model_type="qwen2",
    model_path="Qwen/Qwen2.5-0.5B-Instruct",
    component="trunk",
    port=50051,
    host="0.0.0.0",
    device="cuda",  # 使用 GPU
    max_models=10,   # 最多管理10个模型
    # 传递给模型的额外参数
    start_layer=4,
    end_layer=20
)

server.start()
```

### 示例 3: 使用纯模型服务

```python
import torch.nn as nn
from splitlearn_comm.quickstart import serve

# 定义您自己的模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 768)

    def forward(self, x):
        return self.linear(x)

# 一行代码启动服务！
serve(MyModel(), port=50051)  # 阻塞运行
```

---

## 完整的 Split Learning 示例

### 场景：GPT-2 分布式推理

**架构**:
- **客户端**: 运行 Bottom (0-2层) + Top (10-12层)
- **服务端**: 运行 Trunk (2-10层)

#### 1. 准备模型（仅需一次）

```python
# prepare_models.py
from splitlearn_core.quickstart import load_split_model

# 下载并分割模型
bottom, trunk, top = load_split_model(
    model_type="gpt2",
    split_points=[2, 10],  # Bottom: 0-2, Trunk: 2-10, Top: 10-end
    cache_dir="./models"   # 保存到本地
)

print("✓ 模型已准备好")
print(f"  Bottom: {sum(p.numel() for p in bottom.parameters())/1e6:.2f}M 参数")
print(f"  Trunk:  {sum(p.numel() for p in trunk.parameters())/1e6:.2f}M 参数")
print(f"  Top:    {sum(p.numel() for p in top.parameters())/1e6:.2f}M 参数")
```

#### 2. 启动服务端（Trunk）

```python
# server.py
from splitlearn_core.quickstart import load_split_model
from splitlearn_comm.quickstart import Server

# 加载 Trunk 模型
_, trunk, _ = load_split_model(
    "gpt2",
    split_points=[2, 10],
    cache_dir="./models"
)

# 启动服务器
server = Server(
    model=trunk,
    port=50051,
    device="cuda"  # 或 "cpu"
)

print("服务端启动，监听端口 50051...")
server.start()
server.wait_for_termination()
```

#### 3. 运行客户端（Bottom + Top）

```python
# client.py
from splitlearn_core.quickstart import load_split_model
from splitlearn_comm.quickstart import Client
import torch

# 加载 Bottom 和 Top 模型
bottom, _, top = load_split_model(
    "gpt2",
    split_points=[2, 10],
    cache_dir="./models"
)

# 连接到服务端
client = Client("localhost:50051")

# 准备输入（示例：tokenized text）
input_ids = torch.randint(0, 50257, (1, 10))  # (batch=1, seq_len=10)

# === Split Learning 推理流程 ===

# 步骤 1: 客户端 - Bottom 模型前向传播
bottom_output = bottom(input_ids)
print(f"Bottom 输出形状: {bottom_output.shape}")

# 步骤 2: 发送到服务端 - Trunk 模型计算
trunk_output = client.compute(bottom_output)
print(f"Trunk 输出形状: {trunk_output.shape}")

# 步骤 3: 客户端 - Top 模型前向传播
final_output = top(trunk_output)
print(f"Final 输出形状: {final_output.shape}")

# 获取预测结果
logits = final_output
predicted_ids = torch.argmax(logits, dim=-1)
print(f"预测的 token IDs: {predicted_ids}")

# 关闭连接
client.close()
```

#### 4. 运行

```bash
# 终端 1: 启动服务端
python server.py

# 终端 2: 运行客户端
python client.py
```

---

## 常见使用场景

### 场景 1: 低延迟推理（本地 Bottom + Top）

**适用于**: 需要快速响应的应用（聊天机器人、实时翻译）

```python
# 客户端保留 embedding 和 head，延迟最低
bottom, _, top = load_split_model("gpt2", split_points=[2, 10])

# 快速前向传播
def fast_infer(input_ids):
    hidden = bottom(input_ids)
    hidden = client.compute(hidden)  # 仅此步骤需要网络
    return top(hidden)
```

### 场景 2: 隐私保护（敏感数据不离开客户端）

**适用于**: 医疗、金融等隐私敏感场景

```python
# 原始输入（敏感）仅在客户端处理
sensitive_input = load_medical_data()

# Bottom 模型在本地处理，提取特征
features = bottom(sensitive_input)  # 特征已脱敏

# 仅发送特征到服务端（不含原始数据）
result = client.compute(features)
```

### 场景 3: 资源受限设备（移动端、边缘设备）

**适用于**: 手机、IoT 设备等算力有限的环境

```python
# 客户端仅运行轻量级 Bottom 模型
bottom = load_bottom_model("gpt2", end_layer=2)
bottom.eval()  # 推理模式

# 服务端运行重量级 Trunk + Top
# 客户端设备只需要少量内存和计算
```

### 场景 4: 批处理优化（服务端批处理多个请求）

**适用于**: 高吞吐量场景

```python
# 服务端可以批处理来自多个客户端的请求
# 客户端代码不变，服务端自动优化批处理
```

---

## 常见问题

### Q1: 如何选择 split_points？

**A**: 根据您的需求权衡：

- **低延迟**: 更多层放在客户端（如 [4, 8]）
- **低带宽**: 更少层在客户端（如 [1, 11]）
- **隐私保护**: 确保敏感处理在客户端（Bottom 包含 embedding）
- **平衡**: 推荐 GPT-2 使用 [2, 10]，Qwen2 使用 [4, 20]

### Q2: 模型加载很慢怎么办？

**A**: 使用本地缓存：

```python
# 第一次下载后，模型保存在本地
bottom, trunk, top = load_split_model(
    "gpt2",
    split_points=[2, 10],
    cache_dir="./models"  # 保存到本地
)

# 后续加载直接从本地读取，非常快
```

### Q3: 如何处理多个客户端连接？

**A**: 服务端自动处理并发：

```python
# 服务端代码（支持多客户端）
server = Server(
    model=trunk,
    port=50051,
    max_workers=10  # 最多10个并发请求
)
server.start()

# 多个客户端可以同时连接和发送请求
```

### Q4: 如何监控服务端性能？

**A**: 使用 ManagedServer：

```python
from splitlearn_manager.quickstart import ManagedServer

# 自动包含性能监控
server = ManagedServer("gpt2", port=50051)
server.start()

# 查看 Prometheus 指标
# 访问 http://localhost:9090/metrics
```

### Q5: 客户端和服务端必须在同一台机器吗？

**A**: 不需要！

```python
# 客户端连接到远程服务器
client = Client("remote-server.example.com:50051")

# 或使用 IP 地址
client = Client("192.168.1.100:50051")
```

### Q6: 如何实现负载均衡？

**A**: 部署多个服务端实例：

```python
# 客户端轮询多个服务器
servers = [
    "server1:50051",
    "server2:50051",
    "server3:50051"
]

import random
client = Client(random.choice(servers))
```

---

## 下一步

- **查看完整示例**: `examples/` 目录包含更多示例
- **查看 API 文档**: 详细的 API 参考文档
- **性能调优**: 学习如何优化性能
- **部署指南**: 生产环境部署最佳实践

## 获取帮助

- **文档**: [完整文档](https://splitlearn.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/yourusername/SL/issues)
- **社区**: [Discord](https://discord.gg/splitlearn)

---

**Happy Split Learning!** 🎉
