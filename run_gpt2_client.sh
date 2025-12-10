#!/bin/bash
# GPT-2 客户端启动脚本

echo "========================================"
echo "启动 GPT-2 客户端（Gradio）..."
echo "========================================"

# 设置环境变量
export PYTHONPATH=./SplitLearnCore/src:./SplitLearnComm/src

# 远程服务地址（可根据实际情况修改）
export GPT2_TRUNK_SERVER="${GPT2_TRUNK_SERVER:-localhost:50051}"

# 使用 framework Python
PYTHON=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3

echo ""
echo "环境配置:"
echo "  Python: $PYTHON"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  远程服务: $GPT2_TRUNK_SERVER"
echo ""
echo "提示: 如需连接远程服务器，请设置环境变量:"
echo "  export GPT2_TRUNK_SERVER=\"<远程IP>:50051\""
echo ""

# 启动客户端
$PYTHON gpt2_client_gradio_grpc.py
