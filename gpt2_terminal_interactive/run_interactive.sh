#!/bin/bash
# GPT-2 终端交互式生成工具启动脚本

cd "$(dirname "$0")/.."

# 使用 Framework 的 Python（避免 mutex 锁问题）
PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.11/Resources/Python.app/Contents/MacOS/Python"

# 如果 Framework Python 不存在，尝试使用 Framework 的 python3.11
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11"
    if [ ! -f "$PYTHON_BIN" ]; then
        PYTHON_BIN="python3"
        echo "⚠ 警告: Framework Python 未找到，使用系统 Python"
    else
        echo "✓ 使用 Framework Python: $PYTHON_BIN"
    fi
else
    echo "✓ 使用 Framework Python: $PYTHON_BIN"
fi

# 设置 Python 路径
export PYTHONPATH="./SplitLearnCore/src:./SplitLearnComm/src:$PYTHONPATH"

# 进入目录并运行
cd gpt2_terminal_interactive
exec "$PYTHON_BIN" gpt2_interactive.py
