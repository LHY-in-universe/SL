#!/usr/bin/env python3
"""
GPT-2 Split Learning Server
加载 GPT-2 trunk 模型并提供 gRPC 服务
"""

import os
import sys

# 抑制 gRPC 和 protobuf 警告 (必须在导入 grpc 之前设置)
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'  # 使用 Python 实现

import torch
import logging

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearnComm', 'src'))
sys.path.insert(0, os.path.join(project_root, 'SplitLearnCore', 'src'))

from splitlearn_comm import GRPCComputeServer
from splitlearn_comm.core import ModelComputeFunction

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    print("=" * 70)
    print("GPT-2 Split Learning Server")
    print("=" * 70)

    # 1. 导入 splitlearn 模块（torch.load 需要这些类来反序列化）
    print("\n[1] 导入 splitlearn 模块...")
    try:
        from splitlearn_core.models.gpt2 import GPT2TrunkModel
        print("    ✓ GPT2TrunkModel 导入成功")
    except ImportError as e:
        print(f"    ❌ 导入失败: {e}")
        print("\n请确保 SplitLearning 包已正确安装")
        return

    # 2. 加载 GPT-2 trunk 模型
    trunk_path = os.path.join(current_dir, "gpt2_trunk_full.pt")

    if not os.path.exists(trunk_path):
        print(f"\n❌ 模型文件不存在: {trunk_path}")
        print("\n请先运行: python testcode/prepare_models.py")
        return

    print(f"\n[2] 加载 GPT-2 Trunk 模型...")
    print(f"    路径: {trunk_path}")
    trunk_model = torch.load(trunk_path, map_location='cpu', weights_only=False)
    trunk_model.eval()

    # 计算参数量
    total_params = sum(p.numel() for p in trunk_model.parameters())
    print(f"    ✓ 模型加载完成")
    print(f"    ✓ 参数量: {total_params:,}")

    # 3. 包装为 ComputeFunction
    print("\n[3] 创建 ComputeFunction...")
    compute_fn = ModelComputeFunction(
        model=trunk_model,
        device="cpu",  # 使用 "cuda" 如果有 GPU
        model_name="gpt2-trunk"
    )
    print("    ✓ ComputeFunction 创建完成")

    # 4. 创建并启动服务器
    print("\n[4] 启动 gRPC 服务器...")
    server = GRPCComputeServer(
        compute_fn=compute_fn,
        host="0.0.0.0",
        port=50051,
        max_workers=10
    )

    print("\n" + "=" * 70)
    print("✅ 服务器启动成功！")
    print("=" * 70)
    print(f"📡 监听地址: 0.0.0.0:50051")
    print(f"🔗 本地连接: localhost:50051")
    print(f"🔗 局域网连接: <你的IP>:50051")
    print("\n💡 提示:")
    print("   • 保持此窗口运行")
    print("   • 在另一个终端运行客户端")
    print("   • 按 Ctrl+C 停止服务器")
    print("=" * 70)
    print()

    try:
        server.start()
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n\n停止服务器...")
        server.stop()
        print("✓ 服务器已关闭")


if __name__ == "__main__":
    main()
