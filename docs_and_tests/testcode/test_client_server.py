#!/usr/bin/env python3
"""
简单的客户端-服务器集成测试
测试 gRPC 通信是否正常工作
"""
import os
import sys

# 设置环境变量（必须在导入 grpc 之前）
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import torch

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearnComm', 'src'))

from splitlearn_comm import GRPCComputeClient

def test_connection():
    """测试服务器连接"""
    print("=" * 70)
    print("测试 1: 服务器连接")
    print("=" * 70)

    client = GRPCComputeClient("localhost:50051", timeout=5.0)

    if client.connect():
        print("✅ 连接成功！")
        return client
    else:
        print("❌ 连接失败！")
        sys.exit(1)

def test_compute(client):
    """测试计算功能"""
    print("\n" + "=" * 70)
    print("测试 2: 计算功能")
    print("=" * 70)

    # 创建一个测试张量（模拟 GPT-2 hidden state）
    test_tensor = torch.randn(1, 10, 768)  # [batch, seq_len, hidden_dim]
    print(f"输入张量形状: {test_tensor.shape}")

    try:
        # 发送计算请求
        result = client.compute(test_tensor, model_id="gpt2-trunk")
        print(f"✅ 计算成功！")
        print(f"输出张量形状: {result.shape}")
        print(f"输出张量类型: {result.dtype}")

        # 验证输出形状
        expected_shape = (1, 10, 768)  # GPT-2 保持相同形状
        if result.shape == expected_shape:
            print(f"✅ 输出形状正确: {result.shape}")
        else:
            print(f"⚠️  输出形状不符合预期: {result.shape} (期望: {expected_shape})")

        return True
    except Exception as e:
        print(f"❌ 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_requests(client):
    """测试多次请求"""
    print("\n" + "=" * 70)
    print("测试 3: 多次请求")
    print("=" * 70)

    num_requests = 5
    successes = 0

    for i in range(num_requests):
        test_tensor = torch.randn(1, 5, 768)
        try:
            result = client.compute(test_tensor, model_id="gpt2-trunk")
            successes += 1
            print(f"  请求 {i+1}/{num_requests}: ✅ 成功")
        except Exception as e:
            print(f"  请求 {i+1}/{num_requests}: ❌ 失败 ({e})")

    print(f"\n总结: {successes}/{num_requests} 请求成功")
    return successes == num_requests

def main():
    print("\n🧪 Split Learning 客户端-服务器集成测试\n")

    # 测试 1: 连接
    client = test_connection()

    # 测试 2: 单次计算
    compute_ok = test_compute(client)

    # 测试 3: 多次请求
    multiple_ok = test_multiple_requests(client)

    # 总结
    print("\n" + "=" * 70)
    print("📊 测试总结")
    print("=" * 70)
    print(f"✅ 连接测试: 通过")
    print(f"{'✅' if compute_ok else '❌'} 计算测试: {'通过' if compute_ok else '失败'}")
    print(f"{'✅' if multiple_ok else '❌'} 多次请求测试: {'通过' if multiple_ok else '失败'}")

    if compute_ok and multiple_ok:
        print("\n🎉 所有测试通过！系统运行正常！")
        return 0
    else:
        print("\n❌ 部分测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
