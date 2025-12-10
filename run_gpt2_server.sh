#!/bin/bash
# GPT-2 服务端启动脚本

echo "========================================"
echo "启动 GPT-2 Trunk 服务端..."
echo "========================================"

# 设置环境变量
export PYTHONPATH=./SplitLearnCore/src:./SplitLearnComm/src
export CUDA_VISIBLE_DEVICES=0  # 使用第一张 GPU（如有多张GPU可调整）

# 使用 framework Python
PYTHON=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3

# 创建日志目录
mkdir -p logs

echo ""
echo "环境配置:"
echo "  Python: $PYTHON"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# 启动服务端
$PYTHON gpt2_server_grpc.py
