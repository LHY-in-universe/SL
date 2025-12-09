#!/bin/bash
# 使用 Framework3 Python 启动 Qwen3-VL gRPC 服务端

# 设置环境变量抑制警告
export GRPC_VERBOSITY=ERROR
export GLOG_minloglevel=2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Framework3 Python 路径
PYTHON_FRAMEWORK3="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"

# 检查 Python 是否存在
if [ ! -f "$PYTHON_FRAMEWORK3" ]; then
    echo "错误: 找不到 Framework3 Python: $PYTHON_FRAMEWORK3"
    exit 1
fi

# 进入项目目录
cd "$(dirname "$0")"

# 设置 PYTHONPATH
export PYTHONPATH=./SplitLearnCore/src:./SplitLearnComm/src

# 启动服务端
echo "使用 Framework3 Python 启动服务端..."
echo "Python 路径: $PYTHON_FRAMEWORK3"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

$PYTHON_FRAMEWORK3 qwen3_server_grpc.py

