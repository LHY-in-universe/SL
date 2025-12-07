#!/bin/bash
# 快速启动 LoRA 微调测试脚本

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Python 路径
PYTHON="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"

echo "========================================================================"
echo "Split Learning LoRA 微调测试"
echo "========================================================================"
echo ""

# 检查服务器是否运行
echo "检查 Trunk 服务器状态..."
if [ -f "$SCRIPT_DIR/.trunk.pid" ]; then
    TRUNK_PID=$(cat "$SCRIPT_DIR/.trunk.pid")
    if kill -0 $TRUNK_PID 2>/dev/null; then
        echo "  ✓ Trunk 服务器正在运行 (PID: $TRUNK_PID)"
    else
        echo "  ⚠️  PID 文件存在但进程未运行，启动服务器..."
        bash "$SCRIPT_DIR/start_all.sh"
        sleep 3
    fi
else
    echo "  ⚠️  Trunk 服务器未运行，启动服务器..."
    bash "$SCRIPT_DIR/start_all.sh"
    sleep 3
fi

echo ""
echo "========================================================================"
echo "运行 LoRA 微调测试"
echo "========================================================================"
echo ""

# 运行训练脚本
cd "$PROJECT_ROOT"
$PYTHON "$SCRIPT_DIR/client/train_lora_simple.py" "$@"

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "========================================================================"
    echo "✓ 测试完成"
    echo "========================================================================"
else
    echo "========================================================================"
    echo "❌ 测试失败 (退出码: $exit_code)"
    echo "========================================================================"
fi

exit $exit_code
