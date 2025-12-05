#!/usr/bin/env python3
"""
合并监控示例

演示如何使用客户端和服务端合并报告功能
"""
import time
import torch


def simulate_split_learning_workflow():
    """
    模拟 Split Learning 工作流程

    这个示例展示了如何在实际的 Split Learning 场景中使用合并监控。
    """
    print("=" * 80)
    print("合并监控示例（模拟）")
    print("=" * 80)
    print()

    # 注意：这是一个模拟示例
    # 在实际使用中，你需要：
    # 1. 启动 trunk 服务器（会自动启动 ServerMonitor）
    # 2. 客户端连接到服务器
    # 3. 客户端启动 ClientMonitor
    # 4. 执行 Split Learning 推理
    # 5. 获取服务端监控数据并生成合并报告

    print("【实际使用步骤】")
    print()
    print("1. 启动服务端（会自动启动监控）:")
    print("   ```python")
    print("   from splitlearn_manager import ManagedServer")
    print()
    print("   server = ManagedServer(")
    print("       model_type='gpt2',")
    print("       component='trunk',")
    print("       port=50052,")
    print("       split_points=[2, 10],")
    print("       device='cpu'")
    print("   )")
    print("   server.start()  # 服务端会自动启用监控")
    print("   ```")
    print()

    print("2. 客户端连接并启动监控:")
    print("   ```python")
    print("   from splitlearn_monitor import ClientMonitor")
    print("   from splitlearn_comm import Client")
    print()
    print("   # 初始化客户端监控")
    print("   monitor = ClientMonitor('my_session', enable_gpu=True)")
    print("   monitor.start()")
    print()
    print("   # 连接服务端")
    print("   trunk_client = Client('localhost:50052')")
    print("   ```")
    print()

    print("3. 执行推理（自动收集服务端监控数据）:")
    print("   ```python")
    print("   for i in range(10):")
    print("       with monitor.track_phase('bottom_model'):")
    print("           hidden1 = bottom(input_ids)")
    print()
    print("       with monitor.track_phase('trunk_remote'):")
    print("           # 服务端监控数据会自动附加到响应中")
    print("           hidden2 = trunk_client.compute(hidden1)")
    print()
    print("       with monitor.track_phase('top_model'):")
    print("           output = top(hidden2)")
    print("   ```")
    print()

    print("4. 停止监控并生成合并报告:")
    print("   ```python")
    print("   monitor.stop()")
    print()
    print("   # 获取服务端监控数据")
    print("   server_data = trunk_client._client.get_server_monitoring_data()")
    print()
    print("   # 生成合并报告")
    print("   report_path = monitor.save_merged_report(server_data)")
    print("   print(f'合并报告: {report_path}')")
    print("   ```")
    print()

    print("【合并报告包含的内容】")
    print()
    print("✓ 客户端资源监控（CPU、内存、GPU）")
    print("✓ 服务端资源监控（CPU、内存、GPU）")
    print("✓ 客户端 vs 服务端对比")
    print("✓ 性能分析（bottom、trunk、top 各阶段耗时）")
    print("✓ 时间序列图表和统计数据")
    print()

    print("=" * 80)
    print()

    # 模拟服务端监控数据（实际场景中从 gRPC 响应获取）
    print("【模拟数据示例】")
    print()
    print("模拟生成服务端监控数据...")

    # 创建模拟的服务端监控快照
    server_snapshots = []
    base_time = time.time()

    for i in range(10):
        snapshot = {
            "timestamp": base_time + i * 0.1,
            "cpu_percent": 45.0 + torch.rand(1).item() * 10,
            "memory_mb": 2048.0 + torch.rand(1).item() * 512,
            "memory_percent": 50.0 + torch.rand(1).item() * 10,
            "gpu_available": False,
            "gpu_utilization": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
        }
        server_snapshots.append(snapshot)

    print(f"生成了 {len(server_snapshots)} 个服务端监控快照")
    print()
    print("示例快照:")
    print(f"  时间戳: {server_snapshots[0]['timestamp']:.2f}")
    print(f"  CPU: {server_snapshots[0]['cpu_percent']:.1f}%")
    print(f"  内存: {server_snapshots[0]['memory_mb']:.1f} MB")
    print()

    print("=" * 80)
    print("示例完成！")
    print()
    print("要在实际环境中使用，请参考上面的步骤。")
    print("=" * 80)


if __name__ == "__main__":
    simulate_split_learning_workflow()
