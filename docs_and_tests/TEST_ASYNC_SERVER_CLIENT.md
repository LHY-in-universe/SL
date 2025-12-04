# 异步服务器和客户端测试说明

## 📁 已创建的文件

1. **`testcode/server_async_with_model.py`**
   - 异步 gRPC 服务器
   - 使用 `AsyncGRPCComputeServer`
   - 使用 `AsyncModelComputeFunction`
   - 加载 `testcode/gpt2_trunk_full.pt` 模型

2. **`testcode/client_async_with_model.py`**
   - 异步 gRPC 客户端
   - 连接到异步服务器
   - 显示详细的数据传输信息

## 🚀 使用方式

### 方法 1：手动在两个终端中运行（推荐）

**终端 1 - 启动服务器：**
```bash
cd /Users/lhy/Desktop/Git/SL
python testcode/server_async_with_model.py
```

**等待看到以下输出：**
```
✅ 服务器运行中，等待客户端连接...
📡 服务器地址: localhost:50061
```

**终端 2 - 运行客户端：**
```bash
cd /Users/lhy/Desktop/Git/SL
python testcode/client_async_with_model.py
```

### 方法 2：后台运行服务器

```bash
# 启动服务器（后台）
cd /Users/lhy/Desktop/Git/SL
python testcode/server_async_with_model.py > /tmp/server.log 2>&1 &

# 等待 60-90 秒（模型加载需要时间）
sleep 60

# 检查服务器状态
ps aux | grep server_async_with_model
lsof -i :50061

# 运行客户端
python testcode/client_async_with_model.py
```

## ⏱️ 注意事项

1. **模型加载时间**：
   - `gpt2_trunk_full.pt` 文件大小：224 MB
   - 加载时间可能需要 **1-2 分钟**
   - 在加载过程中会看到 `[mutex.cc : 452]` 警告（这是正常的）

2. **服务器启动流程**：
   - 检查模型文件是否存在
   - 加载模型到内存（`torch.load()`）
   - 创建异步计算函数
   - 创建异步 gRPC 服务器
   - 启动服务器并监听端口

3. **如果连接失败**：
   - 确保服务器已完全启动（看到 "服务器运行中" 消息）
   - 检查端口是否被监听：`lsof -i :50061`
   - 等待更长时间（模型加载可能需要 1-2 分钟）

## ✅ 成功标志

**服务器端：**
```
✅ 服务器运行中，等待客户端连接...
📡 服务器地址: localhost:50061
```

**客户端：**
```
📥 收到响应
📊 接收到的数据:
   形状: torch.Size([1, 10, 768])
   ...
✅ 输出形状正确
```

## 🔍 故障排查

1. **端口未监听**：
   - 等待更长时间（模型加载中）
   - 检查服务器日志

2. **连接被拒绝**：
   - 确保服务器已完全启动
   - 检查防火墙设置

3. **模型加载卡住**：
   - 这是正常的，大模型加载需要时间
   - 等待 1-2 分钟

## 📊 测试内容

客户端会执行以下测试：
1. 连接测试
2. 服务信息查询
3. 单次计算请求（显示详细数据传输信息）
4. 多次请求测试（5次）

每次请求都会显示：
- 输入/输出张量的形状、大小、统计信息
- 数据传输统计（发送/接收数据大小、吞吐量）

