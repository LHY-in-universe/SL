#!/bin/bash
# 启动 Split Learning 客户端并统计所有运行信息
# 用法: ./start_client_with_stats.sh [服务器IP:端口]
# 例如: ./start_client_with_stats.sh 192.168.0.144:50052

# 默认服务器地址
DEFAULT_SERVER="192.168.0.144:50052"

# 获取服务器地址（命令行参数优先，否则使用默认值）
if [ -n "$1" ]; then
    SERVER="$1"
else
    SERVER="$DEFAULT_SERVER"
fi

echo "=========================================="
echo "启动 Split Learning 客户端"
echo "=========================================="
echo "服务器地址: $SERVER"
echo ""
echo "功能说明:"
echo "  - 自动收集客户端资源使用（CPU、内存）"
echo "  - 自动收集服务器资源使用（如果服务器支持）"
echo "  - 自动生成合并监控报告（客户端+服务器统计）"
echo "  - 报告保存在: test/reports/"
echo ""
echo "输入 'quit' 或 'exit' 退出程序"
echo "=========================================="
echo ""

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 使用 Framework Python 3.11 启动客户端
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 test/client/interactive_client.py "$SERVER"

