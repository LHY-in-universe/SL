# 测试指南

## 测试方式

有两种测试方式：
1. **简单版本**（推荐）- 不使用模型，只测试通信功能
2. **模型版本** - 使用真实模型，测试完整流程

---

## 方式 1：简单版本测试（推荐）

### 特点
- ✅ 不需要模型文件
- ✅ 快速启动（瞬间）
- ✅ 无 mutex 警告
- ✅ 专注于测试通信功能

### 步骤

#### 步骤 1：打开第一个终端，启动服务器

```bash
cd /Users/lhy/Desktop/Git/SL
python testcode/server_comm_simple.py
```

**你会看到：**
```
🚀 gRPC 服务器启动（简单版本 - 不使用模型）
======================================================================
🔧 创建计算函数...
✓ 简单计算函数创建（不使用模型）
🌐 创建 gRPC 服务器...
   监听地址: 0.0.0.0:50056
▶️  启动服务器...
   ✓ 服务器已启动
✅ 服务器运行中，等待客户端连接...
📡 服务器地址: localhost:50056
```

#### 步骤 2：打开第二个终端，运行客户端

```bash
cd /Users/lhy/Desktop/Git/SL
python testcode/client_comm_simple.py
```

**你会看到：**
- 连接信息
- 发送的数据详情
- 接收的数据详情
- 传输统计
- 计算结果验证

#### 步骤 3：观察数据传输

**服务器端会显示：**
```
📥 服务器收到请求 #1
📊 输入数据信息:
   形状: torch.Size([1, 10, 768])
   数据大小: 30.00 KB
   ...
📤 输出数据信息:
   形状: torch.Size([1, 10, 768])
   数据大小: 30.00 KB
   ...
📡 数据传输统计:
   接收数据: 30.00 KB
   发送数据: 30.00 KB
   总传输: 60.00 KB
```

**客户端会显示：**
```
📤 发送请求 #1
📊 准备发送的数据:
   形状: torch.Size([1, 10, 768])
   数据大小: 30.00 KB
   ...
📥 收到响应
📊 接收到的数据:
   形状: torch.Size([1, 10, 768])
   数据大小: 30.00 KB
   ...
✅ 计算结果正确: output = input * 2 + 1
```

---

## 方式 2：模型版本测试

### 特点
- ⚠️ 需要模型文件（`testcode/gpt2_trunk_full.pt`）
- ⚠️ 加载时间较长（几秒）
- ⚠️ 可能有 mutex 警告（正常现象）
- ✅ 测试完整的模型推理流程

### 前提条件

确保模型文件存在：
```bash
ls testcode/gpt2_trunk_full.pt
```

如果不存在，先准备模型：
```bash
python testcode/prepare_models.py
```

### 步骤

#### 步骤 1：打开第一个终端，启动服务器

```bash
cd /Users/lhy/Desktop/Git/SL
python testcode/server_comm_test.py
```

**你会看到：**
- 模型加载过程
- 服务器启动信息
- 等待客户端连接

#### 步骤 2：打开第二个终端，运行客户端

```bash
cd /Users/lhy/Desktop/Git/SL
python testcode/client_comm_test.py
```

**你会看到：**
- 连接信息
- 发送的数据详情
- 接收的数据详情
- 传输统计
- 模型推理结果

---

## 快速测试命令

### 一键测试（简单版本）

**终端 1（服务器）：**
```bash
cd /Users/lhy/Desktop/Git/SL && python testcode/server_comm_simple.py
```

**终端 2（客户端）：**
```bash
cd /Users/lhy/Desktop/Git/SL && python testcode/client_comm_simple.py
```

### 一键测试（模型版本）

**终端 1（服务器）：**
```bash
cd /Users/lhy/Desktop/Git/SL && python testcode/server_comm_test.py
```

**终端 2（客户端）：**
```bash
cd /Users/lhy/Desktop/Git/SL && python testcode/client_comm_test.py
```

---

## 测试内容

### 通信功能测试

1. **连接测试**
   - 客户端能否连接到服务器
   - 连接是否稳定

2. **数据传输测试**
   - 数据能否正确发送
   - 数据能否正确接收
   - 数据大小和形状是否正确

3. **计算功能测试**
   - 服务器能否执行计算
   - 计算结果是否正确
   - 计算耗时是否合理

4. **多次请求测试**
   - 连续发送多个请求
   - 验证稳定性
   - 统计成功率

5. **统计信息测试**
   - 获取传输统计
   - 获取性能指标
   - 验证数据准确性

---

## 停止测试

### 停止服务器

在服务器终端按 `Ctrl+C`

### 停止所有 Python 进程

```bash
pkill -f python
```

---

## 常见问题

### 1. 连接失败

**问题：** 客户端无法连接到服务器

**解决：**
- 确保服务器已启动
- 检查端口是否正确（简单版本：50056，模型版本：50055）
- 检查防火墙设置

### 2. 模型文件不存在

**问题：** `gpt2_trunk_full.pt` 不存在

**解决：**
```bash
python testcode/prepare_models.py
```

### 3. mutex 警告

**问题：** 出现 `[mutex.cc : 452] RAW: Lock blocking` 警告

**说明：**
- 如果只在模型加载时出现一次 → 正常，可忽略
- 如果频繁出现 → 需要进一步调查

**解决：**
- 使用简单版本测试（无 mutex 警告）
- 或者忽略单次警告（不影响功能）

---

## 测试脚本说明

| 脚本 | 用途 | 端口 | 需要模型 |
|------|------|------|---------|
| `server_comm_simple.py` | 简单服务器 | 50056 | ❌ |
| `client_comm_simple.py` | 简单客户端 | 50056 | ❌ |
| `server_comm_test.py` | 模型服务器 | 50055 | ✅ |
| `client_comm_test.py` | 模型客户端 | 50055 | ✅ |

---

## 推荐测试流程

1. **先测试简单版本**（验证通信功能）
   ```bash
   # 终端 1
   python testcode/server_comm_simple.py
   
   # 终端 2
   python testcode/client_comm_simple.py
   ```

2. **再测试模型版本**（验证完整流程）
   ```bash
   # 终端 1
   python testcode/server_comm_test.py
   
   # 终端 2
   python testcode/client_comm_test.py
   ```

---

## 预期结果

### 成功标志

- ✅ 服务器成功启动
- ✅ 客户端成功连接
- ✅ 数据成功传输
- ✅ 计算结果正确
- ✅ 多次请求成功
- ✅ 统计信息准确

### 失败标志

- ❌ 连接失败
- ❌ 数据传输错误
- ❌ 计算结果不正确
- ❌ 请求频繁失败

---

## 下一步

测试成功后，你可以：
1. 查看详细的数据传输信息
2. 分析性能指标
3. 测试不同的数据大小
4. 测试不同的请求频率

