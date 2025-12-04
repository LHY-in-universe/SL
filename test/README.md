# Split Learning 测试环境

这是一个完整的 Split Learning 测试环境，包含服务端和客户端。

## 目录结构

```
test/
├── README.md              # 本文件
├── server/                # 服务端
│   └── trunk_server.py    # Trunk 模型服务器 (端口 50052)
├── client/                # 客户端
│   └── test_client.py     # 测试客户端（本地加载 Bottom 和 Top 模型）
├── start_all.sh           # 启动 Trunk 服务器
├── stop_all.sh            # 停止 Trunk 服务器
└── logs/                  # 日志目录（自动创建）
```

**注意**: `server/bottom_server.py` 和 `server/top_server.py` 已不再使用，因为 Bottom 和 Top 模型现在在客户端本地加载。

## 快速开始

### 方式 1: 使用启动脚本（推荐）

**1. 启动 Trunk 服务器**
```bash
bash test/start_all.sh
```

服务器将在后台运行，日志保存在 `test/logs/` 目录。

**2. 运行客户端测试**
```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 test/client/test_client.py
```

客户端会自动：
- 从 `models/bottom/` 加载 Bottom 模型
- 从 `models/top/` 加载 Top 模型
- 连接到 Trunk 服务器 (localhost:50052)

**3. 停止服务器**
```bash
bash test/stop_all.sh
```

### 方式 2: 手动启动（用于调试）

需要打开 **2 个终端窗口**。

**终端 1: Trunk Server**
```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 test/server/trunk_server.py
```

**终端 2: 客户端**
```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 test/client/test_client.py
```

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                         客户端                               │
│                    (test_client.py)                         │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Bottom 模型 (本地)                                │    │
│  │  GPT-2 Layers 0-1                                 │    │
│  │  输入: token IDs → 输出: hidden states (768-dim)   │    │
│  └────────────────────────────────────────────────────┘    │
│                          │                                  │
│                          │ 1. Hidden States                 │
│                          ↓                                  │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           │ 2. 传输: Hidden States (gRPC)
                           ↓
┌───────────────────────────────────────────────────────────┐
│                  Trunk Server (50052)                     │
│                 GPT-2 Layers 2-9                          │
│  输入: hidden states → 输出: hidden states (768-dim)      │
└───────────┬───────────────────────────────────────────────┘
            │
            │ 3. 传输: Hidden States (gRPC)
            ↓
┌───────────────────────────────────────────────────────────┐
│                         客户端                             │
│                    (test_client.py)                        │
│                                                             │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Top 模型 (本地)                                   │   │
│  │  GPT-2 Layers 10-11                               │   │
│  │  输入: hidden states → 输出: logits (50257-dim)   │   │
│  └────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ↓                                  │
│                  解码预测结果                               │
└─────────────────────────────────────────────────────────────┘
```

**架构说明**:
- **客户端**: 本地加载 Bottom 和 Top 模型，处理输入和输出
- **服务端**: 只运行 Trunk 服务器，处理中间层计算
- **数据流**: Token IDs → (本地 Bottom) → Hidden States → (远程 Trunk) → Hidden States → (本地 Top) → Logits

## 测试流程

客户端测试脚本 (`test_client.py`) 执行以下步骤：

1. **加载本地模型** - 从 `models/bottom/` 和 `models/top/` 加载 Bottom 和 Top 模型
2. **连接服务器** - 建立到 Trunk 服务器的 gRPC 连接
3. **准备输入** - 将文本转换为 token IDs
4. **分布式推理**:
   - **本地 Bottom 模型** 处理 token IDs → hidden states
   - **远程 Trunk 服务器** 处理 hidden states → hidden states
   - **本地 Top 模型** 处理 hidden states → logits
5. **解析结果** - 从 logits 解码预测的下一个词
6. **性能统计** - 显示每个阶段的耗时（区分本地和远程）
7. **清理** - 关闭连接并释放本地模型

## 配置说明

### 模型配置

- **模型**: GPT-2 (124M 参数)
- **拆分点**: [2, 10]
  - Bottom: layers 0-1 (53.6M 参数) - **客户端本地加载**
  - Trunk: layers 2-9 (56.7M 参数) - **服务端运行**
  - Top: layers 10-11 (52.8M 参数) - **客户端本地加载**
- **设备**: CPU
- **并发**: 单线程模式 (`max_workers=1`)

### 模型文件位置

客户端从以下路径加载模型：
- Bottom 模型: `models/bottom/gpt2_2-10_bottom.pt`
- Top 模型: `models/top/gpt2_2-10_top.pt`

确保这些文件存在，否则客户端将无法运行。

### 端口配置

- Trunk Server: `50052`

如需修改端口，编辑 `test/server/trunk_server.py` 和 `test/client/test_client.py` 中的端口号。

## 常见问题

### Q1: 首次运行时间很长

**A:** 首次运行需要下载 GPT-2 模型（约 500MB），可能需要几分钟。模型会缓存在 `~/.cache/huggingface/` 目录，之后运行会快得多。

### Q2: 连接被拒绝 (ConnectionRefusedError)

**A:** 确保所有三个服务器都在运行。如果使用启动脚本，检查 `test/logs/` 目录中的日志文件查看错误信息。

### Q3: 如何查看服务器日志

**A:** 日志文件位于 `test/logs/` 目录：
```bash
# 实时查看 Bottom Server 日志
tail -f test/logs/bottom.log

# 查看所有日志
cat test/logs/*.log
```

### Q4: 端口被占用

**A:** 检查端口使用情况：
```bash
lsof -i :50051
lsof -i :50052
lsof -i :50053
```

终止占用端口的进程：
```bash
kill -9 <PID>
```

### Q5: 使用 Anaconda 环境失败

**A:** 本项目必须使用 Framework Python，不支持 Anaconda Python。详见根目录的 `ANACONDA环境无法使用的最终结论.md`。

## 环境要求

- **Python**: Framework Python 3.11 (`/Library/Frameworks/Python.framework/Versions/3.11/bin/python3`)
- **依赖**:
  - `torch` >= 2.0
  - `transformers` >= 4.30
  - `grpcio` >= 1.50
  - `protobuf` >= 4.21
- **内存**: 建议 4GB 以上
- **网络**: 首次运行需要互联网连接下载模型

## 性能参考

在 MacBook (Apple Silicon M1/M2) 上的典型性能：

- **模型加载**: 5-10 秒（首次），2-5 秒（后续）
- **单次推理**: 0.1-0.5 秒
  - Bottom: ~0.05 秒
  - Trunk: ~0.1 秒
  - Top: ~0.05 秒
- **网络延迟**: < 1ms (本地)

## 进阶使用

### 修改测试文本

编辑 `test/client/test_client.py`，修改 `TEST_TEXT` 变量：

```python
TEST_TEXT = "Your custom text here"
```

### 修改拆分点

编辑服务器文件中的 `split_points` 参数：

```python
server = ManagedServer(
    model_type="gpt2",
    component="trunk",
    port=50052,
    split_points=[4, 8],  # 修改拆分点
    device="cpu",
    max_workers=1
)
```

### 使用不同的模型

支持的模型类型：
- `gpt2`: GPT-2 (124M)
- `gemma-2b`: Gemma 2B
- `qwen2-0.5b`: Qwen2 0.5B

修改 `model_type` 参数即可。

## 故障排除

### 服务器启动失败

1. 检查 Python 路径是否正确
2. 检查依赖是否安装完整
3. 查看 `test/logs/` 中的错误日志

### 推理结果异常

1. 确认三个服务器的 `split_points` 参数一致
2. 确认模型已完全加载（查看日志）
3. 尝试重启所有服务器

### 内存不足

1. 减小 batch size（修改输入的 batch 维度）
2. 使用更小的模型（如 `qwen2-0.5b`）
3. 关闭其他占用内存的程序

## 许可证

本测试代码遵循项目主许可证。
