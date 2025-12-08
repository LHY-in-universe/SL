# Python 环境说明

## 为什么会有多个 Python 环境？

你的系统中有多个 Python 环境，这是正常的，因为不同的安装方式会创建不同的 Python 环境：

### 1. **系统 Python** (`/usr/bin/python3`)
- macOS 系统自带的 Python
- 版本通常较旧（Python 3.9）
- 不建议直接修改，可能影响系统功能

### 2. **Homebrew Python** (`/opt/homebrew/bin/python3`)
- 通过 Homebrew 包管理器安装
- 版本较新（Python 3.13.4）
- 独立于系统 Python，更安全

### 3. **Python.org Framework** (`/Library/Frameworks/Python.framework/Versions/3.11/bin/python3`)
- 从 python.org 官方安装包安装
- 当前使用：Python 3.11.0
- **这是本项目使用的 Python 环境**

### 4. **Anaconda Python** (`/Users/lhy/anaconda3/bin/python3`)
- Anaconda 发行版
- 包含大量科学计算库
- 可能导致 mutex 锁问题

### 5. **虚拟环境** (`venv/`)
- 项目特定的隔离环境
- 基于某个基础 Python 创建
- 本项目使用 Python.framework 3.11 创建

## 当前项目配置

本项目使用：
- **基础 Python**: `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3`
- **虚拟环境**: `gpt2_interactive/venv/`
- **运行方式**: `./run.sh`（自动使用虚拟环境）

## 如何更新其他 Python 环境

### 更新 Homebrew Python
```bash
brew upgrade python@3.13
# 或
brew upgrade python@3.12
```

### 更新 Python.org Framework
```bash
# 访问 https://www.python.org/downloads/
# 下载最新版本并安装
```

### 更新 Anaconda Python
```bash
conda update python
conda update --all
```

### 更新 pip 和包
```bash
# 对于 Python.framework 3.11
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pip install --upgrade pip

# 对于 Homebrew Python
/opt/homebrew/bin/python3 -m pip install --upgrade pip

# 对于 Anaconda
conda update pip
```

## 建议

1. **项目开发**: 使用虚拟环境（venv），避免污染全局环境
2. **系统 Python**: 不要修改，保持原样
3. **主要使用**: Python.framework 或 Homebrew Python
4. **Anaconda**: 仅在需要科学计算环境时使用

## 检查 Python 版本

```bash
# 查看所有 Python 版本
which -a python3 | xargs -I {} sh -c 'echo "{}:" && {} --version'

# 查看当前使用的 Python
python3 --version
which python3
```

