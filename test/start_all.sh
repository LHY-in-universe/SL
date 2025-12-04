#!/bin/bash
# 启动 Split Learning Trunk 服务器

set -e

# 检查 Python 路径
PYTHON="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"

if [ ! -f "$PYTHON" ]; then
    echo "❌ Framework Python 未找到: $PYTHON"
    echo "请先安装 Framework Python 或修改此脚本中的 PYTHON 变量"
    exit 1
fi

echo "========================================================================"
echo "启动 Split Learning Trunk 服务器"
echo "========================================================================"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$SCRIPT_DIR/server"

# 创建 logs 目录
mkdir -p "$SCRIPT_DIR/logs"

# 启动 Trunk Server
echo "[1/1] 启动 Trunk Server (端口 50052)..."
$PYTHON "$SERVER_DIR/trunk_server.py" > "$SCRIPT_DIR/logs/trunk.log" 2>&1 &
TRUNK_PID=$!
echo "  ✓ PID: $TRUNK_PID"
sleep 2

# 保存 PID 到文件
echo "$TRUNK_PID" > "$SCRIPT_DIR/.trunk.pid"

echo ""
echo "========================================================================"
echo "✓ Trunk 服务器已启动"
echo "========================================================================"
echo ""
echo "服务器信息:"
echo "  Trunk Server:  localhost:50052 (PID: $TRUNK_PID)"
echo ""
echo "日志文件:"
echo "  Trunk:  $SCRIPT_DIR/logs/trunk.log"
echo ""
echo "运行测试客户端:"
echo "  $PYTHON test/client/test_client.py"
echo ""
echo "注意: 客户端将本地加载 Bottom 和 Top 模型，只连接 Trunk 服务器"
echo ""
echo "停止服务器:"
echo "  bash test/stop_all.sh"
echo ""
