#!/bin/bash
# 使用虚拟环境运行脚本（避免 Anaconda mutex 问题和 Bus error）

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
PYTHON_FRAMEWORK="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"

# 设置 PYTHONPATH 以包含 SplitLearnCore
SPLITLEARN_CORE_PATH="$SCRIPT_DIR/../SplitLearnCore/src"
if [ -d "$SPLITLEARN_CORE_PATH" ]; then
    export PYTHONPATH="$SPLITLEARN_CORE_PATH:${PYTHONPATH:-}"
fi

# 检查虚拟环境是否存在
if [ -f "$VENV_PYTHON" ]; then
    # 使用虚拟环境，优先使用 SplitLearnCore 兼容版本
    if [ -f "$SCRIPT_DIR/interactive_gpt2_splitlearn.py" ]; then
        "$VENV_PYTHON" "$SCRIPT_DIR/interactive_gpt2_splitlearn.py" "$@"
    else
        "$VENV_PYTHON" "$SCRIPT_DIR/interactive_gpt2.py" "$@"
    fi
elif [ -f "$PYTHON_FRAMEWORK" ]; then
    # 如果没有虚拟环境，使用 Python.framework 3.11
    "$PYTHON_FRAMEWORK" "$SCRIPT_DIR/interactive_gpt2.py" "$@"
elif [ -f "/opt/homebrew/bin/python3" ]; then
    # 如果没有 Python.framework，尝试使用 Homebrew Python
    /opt/homebrew/bin/python3 "$SCRIPT_DIR/interactive_gpt2.py" "$@"
else
    # 最后使用系统 Python
    /usr/bin/python3 "$SCRIPT_DIR/interactive_gpt2.py" "$@"
fi
