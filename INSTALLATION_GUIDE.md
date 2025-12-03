# Split Learning Installation Guide

本指南介绍如何安装 Split Learning 框架的各个组件，针对不同的使用场景提供不同的安装方式。

---

## 目录

- [快速安装](#快速安装)
- [客户端安装](#客户端安装轻量级)
- [服务端安装](#服务端安装完整功能)
- [开发环境安装](#开发环境安装)
- [从源码安装](#从源码安装)
- [依赖说明](#依赖说明)
- [常见问题](#常见问题)

---

## 快速安装

### 前提条件

- Python 3.8 或更高版本
- pip 21.0 或更高版本
- （可选）CUDA 工具包用于 GPU 支持

### 验证 Python 版本

```bash
python --version  # 应显示 3.8 或更高
pip --version     # 应显示 21.0 或更高
```

---

## 客户端安装（轻量级）

客户端通常只需要运行 Bottom 和 Top 模型，因此可以使用轻量级安装。

### 方式 1：使用 pip extras（推荐）

```bash
# 安装轻量级客户端依赖
pip install splitlearn-core[client] splitlearn-comm[client]
```

### 方式 2：使用 requirements.txt

```bash
# 克隆仓库
git clone https://github.com/yourusername/SL.git
cd SL

# 安装最小依赖
pip install -r SplitLearnCore/requirements.txt
pip install -r SplitLearnComm/requirements.txt
```

### 安装内容

| 包 | 大小（约） | 说明 |
|----|-----------|------|
| torch | ~200MB | PyTorch 核心 |
| numpy | ~15MB | 数值计算 |
| grpcio | ~10MB | gRPC 通信 |
| protobuf | ~5MB | 协议缓冲区 |
| transformers | ~50MB | Transformers 库（最小安装） |
| safetensors | ~5MB | 安全张量存储 |
| **总计** | **~285MB** | 轻量级安装 |

---

## 服务端安装（完整功能）

服务端需要完整的模型管理和监控功能。

### 方式 1：使用 pip extras（推荐）

```bash
# 安装完整服务端依赖
pip install splitlearn-core[server] \
            splitlearn-comm[server] \
            splitlearn-manager[server]
```

### 方式 2：使用 requirements-dev.txt

```bash
# 克隆仓库
git clone https://github.com/yourusername/SL.git
cd SL

# 安装完整依赖
pip install -r SplitLearnCore/requirements-dev.txt
pip install -r SplitLearnComm/requirements-dev.txt
pip install -r SplitLearnManager/requirements-dev.txt
```

### 安装内容

| 包 | 大小（约） | 说明 |
|----|-----------|------|
| torch | ~200MB | PyTorch 核心 |
| transformers | ~1GB | 完整 Transformers 库 |
| safetensors | ~5MB | 模型存储 |
| huggingface-hub | ~10MB | 模型下载 |
| grpcio | ~10MB | gRPC 通信 |
| pyyaml | ~1MB | 配置管理 |
| psutil | ~5MB | 系统监控 |
| prometheus-client | ~5MB | 指标收集 |
| **总计** | **~1.24GB** | 完整安装 |

---

## 开发环境安装

开发环境需要额外的测试和代码质量工具。

```bash
# 克隆仓库
git clone https://github.com/yourusername/SL.git
cd SL

# 安装开发依赖
pip install -e SplitLearnCore[dev]
pip install -e SplitLearnComm[dev]
pip install -e SplitLearnManager[dev]

# 或者使用 requirements-dev.txt
pip install -r SplitLearnCore/requirements-dev.txt
pip install -r SplitLearnComm/requirements-dev.txt
pip install -r SplitLearnManager/requirements-dev.txt
```

### 开发依赖包含

- **测试工具**: pytest, pytest-cov, pytest-asyncio, pytest-timeout
- **代码质量**: black, isort, flake8, mypy
- **文档工具**: sphinx, sphinx-rtd-theme
- **构建工具**: build, twine
- **调试工具**: ipython, ipdb

---

## 从源码安装

### 开发模式安装（推荐用于开发）

```bash
# 克隆仓库
git clone https://github.com/yourusername/SL.git
cd SL

# 开发模式安装（代码更改立即生效）
cd SplitLearnCore
pip install -e .

cd ../SplitLearnComm
pip install -e .

cd ../SplitLearnManager
pip install -e .
```

### 标准安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/SL.git
cd SL

# 标准安装
pip install ./SplitLearnCore
pip install ./SplitLearnComm
pip install ./SplitLearnManager
```

---

## 依赖说明

### 核心依赖（所有安装都需要）

```
torch>=2.0.0           # PyTorch 深度学习框架
numpy>=1.24.0          # 数值计算库
grpcio>=1.50.0         # gRPC 通信框架
protobuf>=4.0.0        # 协议缓冲区
```

### 可选依赖

#### Client 额外依赖

```
transformers>=4.30.0   # Transformer 模型支持
safetensors>=0.3.0     # 安全模型存储
```

#### Server 额外依赖

```
huggingface-hub>=0.16.0    # HuggingFace 模型下载
tqdm>=4.65.0               # 进度条
psutil>=5.9.0              # 系统资源监控
pyyaml>=6.0                # YAML 配置
prometheus-client>=0.16.0  # Prometheus 指标
```

#### UI 额外依赖

```
gradio>=3.50.0,<5.0.0  # Web UI 框架
pandas>=1.5.0          # 数据处理
```

---

## 常见问题

### Q1: 如何选择安装方式？

**A**: 根据您的使用场景选择：

- **只需要客户端**（运行 Bottom/Top 模型）：
  ```bash
  pip install splitlearn-core[client] splitlearn-comm[client]
  ```

- **只需要服务端**（运行 Trunk 模型）：
  ```bash
  pip install splitlearn-core[server] splitlearn-comm[server] splitlearn-manager[server]
  ```

- **开发或需要完整功能**：
  ```bash
  pip install -r requirements-dev.txt
  ```

### Q2: 安装时间太长怎么办？

**A**: 使用国内镜像源加速：

```bash
# 使用清华镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    splitlearn-core[client] splitlearn-comm[client]

# 或者配置全局镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q3: 如何验证安装成功？

**A**: 运行以下命令验证：

```python
# 验证 SplitLearnCore
python -c "from splitlearn import ModelFactory; print('✓ SplitLearnCore OK')"

# 验证 SplitLearnComm
python -c "from splitlearn_comm import GRPCComputeClient; print('✓ SplitLearnComm OK')"

# 验证 SplitLearnManager
python -c "from splitlearn_manager import AsyncModelManager; print('✓ SplitLearnManager OK')"

# 验证 Quickstart API
python -c "from splitlearn.quickstart import load_split_model; print('✓ Quickstart API OK')"
```

### Q4: 客户端是否必须安装 transformers？

**A**: 是的，因为 Bottom/Top 模型代码依赖 transformers 的 Block 和 Config 类。但您可以：
- 使用轻量级的 `[client]` extras，避免安装完整的 transformers
- 使用预下载的模型文件，避免网络下载

### Q5: 如何卸载？

**A**: 使用 pip 卸载：

```bash
pip uninstall splitlearn-core splitlearn-comm splitlearn-manager
```

### Q6: 安装后占用空间太大怎么办？

**A**:
- 客户端只需 ~285MB，确保没有安装 `[server]` extras
- 清理 pip 缓存：`pip cache purge`
- 清理 HuggingFace 缓存：删除 `~/.cache/huggingface/`

### Q7: GPU 支持如何配置？

**A**:

```bash
# 安装 PyTorch with CUDA（根据您的 CUDA 版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证 GPU 可用性
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Q8: 离线安装怎么办？

**A**:

```bash
# 在有网络的机器上下载依赖
pip download -r requirements.txt -d ./packages

# 在离线机器上安装
pip install --no-index --find-links=./packages -r requirements.txt
```

---

## 升级到最新版本

```bash
# 升级所有包
pip install --upgrade splitlearn-core splitlearn-comm splitlearn-manager

# 或从源码升级（开发模式）
cd SL
git pull
pip install -e . --upgrade
```

---

## 下一步

安装完成后，请查看 [QUICKSTART_GUIDE.md](./QUICKSTART_GUIDE.md) 快速开始使用。

## 获取帮助

- **文档**: [完整文档](https://splitlearn.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/yourusername/SL/issues)
- **示例**: 查看 `examples/` 目录

---

**祝使用愉快！** 🚀
