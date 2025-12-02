#!/usr/bin/env python3
"""
Simple Client Example

演示如何使用 splitlearn-comm 连接到 gRPC 服务器并执行计算。
"""

import logging
import torch

from splitlearn_comm import GRPCComputeClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    print("=" * 70)
    print("Simple Client Example - splitlearn-comm")
    print("=" * 70)

    # 1. 创建客户端
    print("\n[1] Creating client...")
    client = GRPCComputeClient(
        server_address="localhost:50051",
        timeout=30.0
    )

    # 2. 连接到服务器
    print("\n[2] Connecting to server...")
    if not client.connect():
        print("✗ Failed to connect to server")
        print("  Make sure the server is running (python examples/simple_server.py)")
        return

    # 3. 获取服务器信息
    print("\n[3] Getting server info...")
    info = client.get_service_info()
    if info:
        print(f"  Service: {info['service_name']}")
        print(f"  Version: {info['version']}")
        print(f"  Device: {info['device']}")
        print(f"  Uptime: {info['uptime_seconds']:.1f}s")

    # 4. 执行计算
    print("\n[4] Performing computations...")
    for i in range(5):
        # 创建随机输入
        input_tensor = torch.randn(1, 10, 768)

        # 远程计算
        output_tensor = client.compute(input_tensor)

        print(f"  Request {i+1}: {input_tensor.shape} → {output_tensor.shape}")

    # 5. 获取统计信息
    print("\n[5] Statistics:")
    stats = client.get_statistics()
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Avg network time: {stats['avg_network_time_ms']:.2f}ms")
    print(f"  Avg compute time: {stats['avg_compute_time_ms']:.2f}ms")
    print(f"  Avg total time: {stats['avg_total_time_ms']:.2f}ms")

    # 6. 关闭连接
    print("\n[6] Closing connection...")
    client.close()
    print("✓ Done!")


if __name__ == "__main__":
    main()
