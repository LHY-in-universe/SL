"""
测试增强的监控功能

这个脚本演示了新增的监控功能：
1. 详细的延迟统计（P95/P99等）
2. 实时日志显示
3. 延迟分布和趋势图
"""

import os
import sys
import torch
import time

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearnComm', 'src'))

from splitlearn_comm.core import ModelComputeFunction
from splitlearn_comm.server import ComputeServicer
from splitlearn_comm.ui import ServerMonitoringUI


def main():
    print("=== 测试增强的监控功能 ===\n")

    # 1. 创建一个简单的测试模型
    print("1. 创建测试模型...")
    test_model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 10)
    )
    test_model.eval()
    print("   ✓ 模型创建完成\n")

    # 2. 创建 ComputeFunction
    print("2. 创建 ComputeFunction...")
    compute_fn = ModelComputeFunction(model=test_model, device="cpu")
    print("   ✓ ComputeFunction 创建完成\n")

    # 3. 创建 Servicer（会自动初始化监控管理器）
    print("3. 创建 ComputeServicer（集成监控）...")
    servicer = ComputeServicer(compute_fn, history_size=100)
    print("   ✓ Servicer 创建完成")
    print("   ✓ MetricsManager 已初始化")
    print("   ✓ LogManager 已初始化\n")

    # 4. 模拟一些请求来生成监控数据
    print("4. 模拟请求生成监控数据...")
    print("   生成 20 个请求...")

    for i in range(20):
        # 创建测试输入
        test_input = torch.randn(1, 10)

        # 编码
        encoded_data, shape = servicer.codec.encode(test_input)

        # 模拟 gRPC 请求对象
        class MockRequest:
            def __init__(self, data, shape):
                self.data = data
                self.shape = shape

            def HasField(self, field):
                return False

        class MockContext:
            def set_code(self, code):
                pass

            def set_details(self, details):
                pass

        request = MockRequest(encoded_data, shape)
        context = MockContext()

        # 调用 Compute（会自动记录延迟和日志）
        response = servicer.Compute(request, context)

        # 随机延迟模拟不同的请求时间
        time.sleep(0.1)

        if (i + 1) % 5 == 0:
            print(f"   已完成 {i + 1} 个请求")

    print("   ✓ 20 个请求完成\n")

    # 5. 测试 get_metrics() 方法
    print("5. 测试增强的 get_metrics() 方法...")
    metrics = servicer.get_metrics()

    print(f"   总请求数: {metrics['total_requests']}")
    print(f"   成功率: {metrics['success_rate']*100:.1f}%")

    latency_stats = metrics['latency_stats']
    print(f"\n   延迟统计:")
    print(f"   - 平均: {latency_stats['mean']:.2f} ms")
    print(f"   - P50 (中位数): {latency_stats['p50']:.2f} ms")
    print(f"   - P95: {latency_stats['p95']:.2f} ms")
    print(f"   - P99: {latency_stats['p99']:.2f} ms")
    print(f"   - 最小: {latency_stats['min']:.2f} ms")
    print(f"   - 最大: {latency_stats['max']:.2f} ms")

    print(f"\n   吞吐量: {metrics['current_rps']:.2f} RPS")
    print("   ✓ get_metrics() 测试完成\n")

    # 6. 测试 get_logs() 方法
    print("6. 测试 get_logs() 方法...")
    logs = servicer.get_logs(level_filter="INFO", limit=5)
    print(f"   获取到 {len(logs)} 条 INFO 级别日志")
    if logs:
        print("   最新的一条日志:")
        latest = logs[0]
        print(f"   - 时间: {latest['timestamp'].strftime('%H:%M:%S')}")
        print(f"   - 级别: {latest['level']}")
        print(f"   - 消息: {latest['message']}")
    print("   ✓ get_logs() 测试完成\n")

    # 7. 启动监控 UI
    print("7. 启动增强的监控 UI...")
    print("\n" + "="*60)
    print("📊 监控 UI 包含 3 个标签页:")
    print("   1. 📈 概览 - 服务器统计和请求历史")
    print("   2. 📝 实时日志 - 支持级别过滤的日志查看")
    print("   3. ⏱️ 延迟分析 - P95/P99统计、分布图、趋势图")
    print("="*60)
    print("\n🌐 UI 将在浏览器中打开: http://127.0.0.1:7862")
    print("\n⚡ 功能特性:")
    print("   • 每 2 秒自动刷新")
    print("   • 支持日志级别过滤 (DEBUG/INFO/WARNING/ERROR)")
    print("   • 交互式 Plotly 图表")
    print("   • 详细的延迟百分位统计")
    print("\n按 Ctrl+C 停止\n")

    # 创建并启动 UI
    ui = ServerMonitoringUI(
        servicer=servicer,
        theme="default",
        refresh_interval=2
    )

    try:
        ui.launch(
            share=False,
            server_port=7862,
            inbrowser=True,
            blocking=True
        )
    except KeyboardInterrupt:
        print("\n\n✓ 测试完成！")


if __name__ == "__main__":
    main()
