#!/bin/bash
# GPT-2 完整模型启动脚本

echo "========================================"
echo "启动 GPT-2 完整模型（对照组）..."
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
echo "注意: 完整模型在端口 7861 运行（与分拆客户端的7860不同）"
echo ""

# 启动完整模型
$PYTHON gpt2_full_model_gradio.py
