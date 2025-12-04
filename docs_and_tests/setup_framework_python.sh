#!/bin/bash
# 快速设置 Framework Python 环境来使用 SplitLearnCore

set -e  # 遇到错误立即退出

echo "=========================================="
echo "设置 Framework Python 环境"
echo "=========================================="
echo ""

# 检查 Framework Python
PYTHON_PATH="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"

if [ ! -f "$PYTHON_PATH" ]; then
    echo "❌ Framework Python 未找到"
    echo "   预期位置: $PYTHON_PATH"
    echo ""
    echo "请先安装 Python 3.11："
    echo "   https://www.python.org/downloads/"
    exit 1
fi

echo "✓ 找到 Framework Python"
$PYTHON_PATH --version
echo ""

# 虚拟环境路径
VENV_PATH="$HOME/venv-splitlearn"

echo "创建虚拟环境: $VENV_PATH"
if [ -d "$VENV_PATH" ]; then
    echo "⚠️  虚拟环境已存在，将删除并重新创建"
    rm -rf "$VENV_PATH"
fi

$PYTHON_PATH -m venv "$VENV_PATH"
echo "✓ 虚拟环境创建完成"
echo ""

# 激活虚拟环境
echo "激活虚拟环境..."
source "$VENV_PATH/bin/activate"
echo "✓ 虚拟环境已激活"
echo ""

# 升级 pip
echo "升级 pip..."
pip install --upgrade pip -q
echo "✓ pip 已升级"
echo ""

# 安装依赖
echo "安装 PyTorch 和 transformers..."
pip install torch torchvision transformers -q
echo "✓ 依赖安装完成"
echo ""

# 安装 SplitLearnCore
SPLITLEARN_PATH="$(cd "$(dirname "$0")/SplitLearnCore" && pwd)"
if [ -d "$SPLITLEARN_PATH" ]; then
    echo "安装 SplitLearnCore (开发模式)..."
    pip install -e "$SPLITLEARN_PATH" -q
    echo "✓ SplitLearnCore 安装完成"
else
    echo "⚠️  未找到 SplitLearnCore 目录: $SPLITLEARN_PATH"
    echo "   跳过安装，你可以稍后手动安装"
fi
echo ""

# 测试
echo "=========================================="
echo "测试环境"
echo "=========================================="
python << 'EOF'
import sys
import torch
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")

try:
    from splitlearn_core import ModelFactory
    print("SplitLearnCore: ✓ 已安装")
except ImportError:
    print("SplitLearnCore: ⚠️  未安装")
EOF
echo ""

# 完成
echo "=========================================="
echo "✅ 设置完成！"
echo "=========================================="
echo ""
echo "使用方法："
echo ""
echo "1. 激活虚拟环境："
echo "   source $VENV_PATH/bin/activate"
echo ""
echo "2. 运行你的脚本："
echo "   python your_script.py"
echo ""
echo "3. 退出虚拟环境："
echo "   deactivate"
echo ""
echo "或者，创建别名（添加到 ~/.zshrc 或 ~/.bashrc）："
echo "   alias splitlearn-python='source $VENV_PATH/bin/activate'"
echo ""
