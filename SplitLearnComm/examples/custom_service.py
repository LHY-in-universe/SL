#!/usr/bin/env python3
"""
Custom Service Example

演示如何创建自定义的 ComputeFunction 实现。
"""

import logging
import torch

from splitlearn_comm import GRPCComputeServer, GRPCComputeClient
from splitlearn_comm.core import ComputeFunction

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ImageProcessingFunction(ComputeFunction):
    """
    自定义计算函数示例：图像处理

    这个例子展示了如何实现自定义的计算逻辑，
    而不仅仅是模型前向传播。
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.request_count = 0

    def compute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        应用简单的图像处理操作：
        1. 标准化
        2. 高斯模糊
        3. 对比度增强
        """
        self.request_count += 1

        # 移动到目标设备
        x = input_tensor.to(self.device)

        # 1. 标准化
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        x = (x - mean) / (std + 1e-6)

        # 2. 简单的卷积（模拟高斯模糊）
        kernel_size = 3
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.device)
        kernel = kernel / kernel.numel()

        # 对每个通道应用卷积
        if x.dim() == 4:  # [B, C, H, W]
            padding = kernel_size // 2
            x = torch.nn.functional.conv2d(
                x, kernel.repeat(x.shape[1], 1, 1, 1),
                padding=padding, groups=x.shape[1]
            )

        # 3. 对比度增强
        x = torch.tanh(x * 1.5) / 1.5

        return x

    def get_info(self):
        """返回服务信息"""
        return {
            "name": "ImageProcessing",
            "device": self.device,
            "operations": ["normalize", "gaussian_blur", "contrast_enhance"],
            "processed_images": self.request_count
        }

    def setup(self):
        """初始化设置"""
        print(f"ImageProcessingFunction initialized on {self.device}")

    def teardown(self):
        """清理资源"""
        print(f"Processed {self.request_count} images in total")


def run_server():
    """启动服务器"""
    print("\n" + "=" * 70)
    print("Custom Service - Server")
    print("=" * 70)

    # 创建自定义计算函数
    compute_fn = ImageProcessingFunction(device="cpu")

    # 启动服务器
    server = GRPCComputeServer(
        compute_fn=compute_fn,
        host="0.0.0.0",
        port=50051
    )

    try:
        server.start()
        print("\n✓ Server is running. Press Ctrl+C to stop.\n")
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        server.stop()


def run_client():
    """运行客户端测试"""
    print("\n" + "=" * 70)
    print("Custom Service - Client")
    print("=" * 70)

    # 创建客户端
    client = GRPCComputeClient("localhost:50051")

    if not client.connect():
        print("✗ Failed to connect. Make sure server is running.")
        return

    # 获取服务信息
    info = client.get_service_info()
    print(f"\nService Info:")
    print(f"  Name: {info['service_name']}")
    print(f"  Operations: {info['custom_info'].get('operations', 'N/A')}")

    # 创建模拟图像数据
    print("\nProcessing images...")
    for i in range(3):
        # 模拟图像 [batch=1, channels=3, height=64, width=64]
        image = torch.randn(1, 3, 64, 64)

        # 远程处理
        processed = client.compute(image)

        print(f"  Image {i+1}: {image.shape} → {processed.shape}")
        print(f"    Input range: [{image.min():.2f}, {image.max():.2f}]")
        print(f"    Output range: [{processed.min():.2f}, {processed.max():.2f}]")

    # 统计信息
    stats = client.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Avg time: {stats['avg_total_time_ms']:.2f}ms")

    client.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "server":
        run_server()
    elif len(sys.argv) > 1 and sys.argv[1] == "client":
        run_client()
    else:
        print("Usage:")
        print("  python custom_service.py server  # Start server")
        print("  python custom_service.py client  # Run client")
