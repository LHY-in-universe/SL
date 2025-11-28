#!/usr/bin/env python3
"""
Simple Server Example

演示如何使用 splitlearn-comm 创建一个简单的 gRPC 服务器。
"""

import logging
import torch

from splitlearn_comm import GRPCComputeServer
from splitlearn_comm.core import ModelComputeFunction

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    print("=" * 70)
    print("Simple Server Example - splitlearn-comm")
    print("=" * 70)

    # 1. 创建一个简单的 PyTorch 模型
    print("\n[1] Creating model...")
    model = torch.nn.Sequential(
        torch.nn.Linear(768, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(1024, 768)
    )
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 2. 包装为 ComputeFunction
    print("\n[2] Wrapping model in ComputeFunction...")
    compute_fn = ModelComputeFunction(
        model=model,
        device="cpu",  # 使用 "cuda" 如果有 GPU
        model_name="SimpleTransformerLayer"
    )

    # 3. 创建并启动服务器
    print("\n[3] Starting gRPC server...")
    server = GRPCComputeServer(
        compute_fn=compute_fn,
        host="0.0.0.0",
        port=50051,
        max_workers=10
    )

    try:
        server.start()
        print("\n✓ Server is running. Press Ctrl+C to stop.\n")
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        server.stop()


if __name__ == "__main__":
    main()
