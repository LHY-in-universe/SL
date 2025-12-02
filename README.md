# SplitLearn Project

这是一个用于分布式深度学习和大型语言模型（LLM）物理拆分的综合项目。它由三个核心组件组成，旨在实现模型的拆分、通信和管理。

## 项目结构

本项目包含三个主要子项目：

### 1. [SplitLearnCore](./SplitLearnCore)
核心逻辑库，负责将 Transformer 模型（如 GPT-2, Qwen2, Gemma）拆分为三个部分：
- **Bottom**: Embeddings + 前 N 层
- **Trunk**: 中间 M 层（计算密集型）
- **Top**: 最后 K 层 + LM Head

### 2. [SplitLearnComm](./SplitLearnComm)
高性能通信库，基于 gRPC。
- 负责在拆分后的模型部分之间传输 Tensor 数据。
- 优化了序列化性能，支持压缩和重试机制。
- 支持异步通信（grpc.aio）

### 3. [SplitLearnManager](./SplitLearnManager)
模型部署和生命周期管理库。
- 负责模型的加载、卸载和资源监控。
- 提供负载均衡和健康检查功能。
- 支持异步模型管理，解决并发锁阻塞问题

## 快速开始

### 安装

建议按以下顺序以开发模式安装所有组件：

```bash
# 1. 安装核心库
cd SplitLearnCore
pip install -e .

# 2. 安装通信库
cd ../SplitLearnComm
pip install -e .

# 3. 安装管理库
cd ../SplitLearnManager
pip install -e .
```

### 使用示例

每个子目录中都包含详细的文档和示例代码：

- **模型拆分**: 查看 `SplitLearnCore/examples/`
- **通信测试**: 查看 `SplitLearnComm/examples/`
- **服务管理**: 查看 `SplitLearnManager/examples/`

## 文档

所有详细文档均已翻译为中文，位于各子项目的 `docs/` 目录下。

- [SplitLearnCore 文档](./SplitLearnCore/docs)
- [SplitLearnComm 文档](./SplitLearnComm/docs)
- [SplitLearnManager 文档](./SplitLearnManager/docs)

## 贡献

欢迎提交 Pull Request 或 Issue 来改进这个项目。

## 许可证

MIT License
