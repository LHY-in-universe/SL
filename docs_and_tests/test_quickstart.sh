#!/bin/bash

echo "=== SplitLearn 快速测试脚本 ==="
echo ""

# 1. 关闭所有进程
echo "1. 关闭所有 Python 进程..."
pkill -f "python.*quickstart" 2>/dev/null
sleep 2
echo "✓ 完成"
echo ""

# 2. 启动服务器
echo "2. 启动服务器..."
cd /Users/lhy/Desktop/Git/SL
python examples/quickstart_server.py > /tmp/test_server.log 2>&1 &
SERVER_PID=$!
echo "服务器 PID: $SERVER_PID"
echo "等待 40 秒让服务器完全启动（包括模型加载）..."
sleep 40
echo ""

# 3. 检查服务器状态
echo "3. 检查服务器状态..."
if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "✓ 服务器进程正在运行"
    if lsof -i :50051 > /dev/null 2>&1; then
        echo "✓ 端口 50051 正在监听"
        SERVER_OK=true
    else
        echo "✗ 端口 50051 未监听"
        SERVER_OK=false
    fi
else
    echo "✗ 服务器进程已停止"
    SERVER_OK=false
fi
echo ""

# 4. 显示服务器日志
echo "4. 服务器日志（最后 20 行）："
tail -20 /tmp/test_server.log
echo ""

# 5. 测试客户端（如果服务器正常）
if [ "$SERVER_OK" = true ]; then
    echo "5. 测试客户端连接..."
    timeout 10 python examples/quickstart_client.py 2>&1
    echo ""
fi

# 6. 检查 mutex 警告
echo "6. 检查 mutex 警告..."
MUTEX_COUNT=$(grep -c "mutex" /tmp/test_server.log 2>/dev/null || echo "0")
if [ "$MUTEX_COUNT" -gt 0 ]; then
    echo "⚠️  发现 $MUTEX_COUNT 个 mutex 警告"
    echo "前 3 个警告："
    grep "mutex" /tmp/test_server.log | head -3
else
    echo "✓ 没有发现 mutex 警告"
fi
echo ""

# 7. 清理
echo "7. 清理..."
kill $SERVER_PID 2>/dev/null
echo "✓ 测试完成"

