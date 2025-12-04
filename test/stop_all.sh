#!/bin/bash
# 停止 Split Learning Trunk 服务器

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================================"
echo "停止 Split Learning Trunk 服务器"
echo "========================================================================"
echo ""

stopped=0

# 停止 Trunk Server
if [ -f "$SCRIPT_DIR/.trunk.pid" ]; then
    TRUNK_PID=$(cat "$SCRIPT_DIR/.trunk.pid")
    if kill -0 $TRUNK_PID 2>/dev/null; then
        echo "[1/1] 停止 Trunk Server (PID: $TRUNK_PID)..."
        kill $TRUNK_PID
        rm "$SCRIPT_DIR/.trunk.pid"
        echo "  ✓ 已停止"
        stopped=$((stopped + 1))
    else
        echo "[1/1] Trunk Server 未运行"
        rm "$SCRIPT_DIR/.trunk.pid"
    fi
else
    echo "[1/1] Trunk Server 未找到 PID 文件"
fi

echo ""
if [ $stopped -gt 0 ]; then
    echo "✓ 已停止 Trunk 服务器"
else
    echo "⚠️  Trunk 服务器未运行"
fi
echo ""
