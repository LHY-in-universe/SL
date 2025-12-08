#!/bin/bash
# 使用系统自带的 Python 运行脚本

# 使用系统 Python（避免 Anaconda mutex 问题）
/usr/bin/python3 "$(dirname "$0")/interactive_gpt2.py" "$@"

