#!/usr/bin/env python3
"""
gRPC 客户端测试脚本 - 简单版本

连接简单服务器（不使用模型），测试通信功能
"""

import os
import sys
import time
import torch
import logging

# 设置环境变量（必须在导入 grpc 之前）
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearnComm', 'src'))

# 配置详细日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

from splitlearn_comm import GRPCComputeClient

# 测试配置
SERVER_ADDRESS = "localhost:50056"  # 简单服务器端口
TIMEOUT = 30.0


def print_tensor_info(tensor, name, prefix="   "):
    """打印张量详细信息"""
    print(f"{prefix}形状: {tensor.shape}")
    print(f"{prefix}数据类型: {tensor.dtype}")
    print(f"{prefix}数据大小: {tensor.numel() * 4 / 1024:.2f} KB")
    print(f"{prefix}最小值: {tensor.min().item():.6f}")
    print(f"{prefix}最大值: {tensor.max().item():.6f}")
    print(f"{prefix}平均值: {tensor.mean().item():.6f}")


def test_connection():
    """测试连接"""
    print("\n" + "=" * 70)
    print("🔌 连接测试")
    print("=" * 70)
    
    client = GRPCComputeClient(
        server_address=SERVER_ADDRESS,
        timeout=TIMEOUT
    )
    
    print(f"\n📡 连接服务器: {SERVER_ADDRESS}")
    print("   正在连接...")
    
    if client.connect():
        print("   ✓ 连接成功！")
        return client
    else:
        print("   ❌ 连接失败！")
        print(f"\n💡 请确保服务器正在运行:")
        print(f"   python testcode/server_comm_simple.py")
        return None


def test_compute(client, request_num=1):
    """测试计算并显示详细信息"""
    print("\n" + "=" * 70)
    print(f"📤 发送请求 #{request_num}")
    print("=" * 70)
    
    # 创建测试输入
    test_input = torch.randn(1, 10, 768)
    
    print(f"\n📊 准备发送的数据:")
    print_tensor_info(test_input, "输入数据")
    
    # 计算数据大小
    input_size_kb = test_input.numel() * 4 / 1024
    
    # 发送请求
    print(f"\n🚀 发送计算请求...")
    print(f"   数据大小: {input_size_kb:.2f} KB")
    print(f"   正在传输...")
    
    start_time = time.time()
    
    try:
        output = client.compute(test_input)
        
        total_time = (time.time() - start_time) * 1000
        
        print(f"\n📥 收到响应")
        print("=" * 70)
        
        # 显示输出数据信息
        print(f"\n📊 接收到的数据:")
        print_tensor_info(output, "输出数据")
        
        # 验证计算结果（应该是 input * 2 + 1）
        expected = test_input * 2 + 1
        if torch.allclose(output, expected, atol=1e-5):
            print(f"\n✅ 计算结果正确: output = input * 2 + 1")
        else:
            print(f"\n⚠️  计算结果不符合预期")
        
        # 计算传输统计
        output_size_kb = output.numel() * 4 / 1024
        total_size_kb = input_size_kb + output_size_kb
        
        print(f"\n📡 传输统计:")
        print(f"   发送数据: {input_size_kb:.2f} KB")
        print(f"   接收数据: {output_size_kb:.2f} KB")
        print(f"   总传输: {total_size_kb:.2f} KB")
        print(f"   总耗时: {total_time:.2f} ms")
        if total_time > 0:
            print(f"   吞吐量: {total_size_kb / (total_time / 1000):.2f} KB/s")
        
        # 验证输出形状
        if output.shape == test_input.shape:
            print(f"\n✅ 输出形状正确: {output.shape}")
        else:
            print(f"\n⚠️  输出形状不符合预期: {output.shape} (期望: {test_input.shape})")
        
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 请求失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_requests(client, num_requests=5):
    """测试多次请求"""
    print("\n" + "=" * 70)
    print(f"🔄 多次请求测试 ({num_requests} 次)")
    print("=" * 70)
    
    successes = 0
    total_time = 0.0
    total_data = 0.0
    
    for i in range(num_requests):
        print(f"\n--- 请求 {i+1}/{num_requests} ---")
        
        test_input = torch.randn(1, 5, 768)
        input_size_kb = test_input.numel() * 4 / 1024
        
        start_time = time.time()
        try:
            output = client.compute(test_input)
            elapsed = (time.time() - start_time) * 1000
            
            # 验证结果
            expected = test_input * 2 + 1
            if torch.allclose(output, expected, atol=1e-5):
                output_size_kb = output.numel() * 4 / 1024
                request_data = input_size_kb + output_size_kb
                
                total_time += elapsed
                total_data += request_data
                successes += 1
                
                print(f"   ✓ 成功 (耗时: {elapsed:.2f} ms, 数据: {request_data:.2f} KB)")
            else:
                print(f"   ⚠️  结果不正确")
            
        except Exception as e:
            print(f"   ❌ 失败: {e}")
    
    print(f"\n📊 总结:")
    print(f"   成功: {successes}/{num_requests}")
    if successes > 0:
        print(f"   总耗时: {total_time:.2f} ms")
        print(f"   平均耗时: {total_time/successes:.2f} ms")
        print(f"   总传输: {total_data:.2f} KB")
        if total_time > 0:
            print(f"   平均吞吐量: {total_data / (total_time / 1000):.2f} KB/s")
    
    return successes == num_requests


def main():
    print("\n" + "=" * 70)
    print("💻 gRPC 客户端测试（简单版本 - 测试通信功能）")
    print("=" * 70)
    print(f"\n📡 服务器地址: {SERVER_ADDRESS}")
    print(f"⏱️  超时时间: {TIMEOUT} 秒")
    print(f"💡 服务器执行: output = input * 2 + 1")
    print()
    
    # 连接服务器
    client = test_connection()
    if client is None:
        return 1
    
    try:
        # 测试单次计算
        test_compute(client, request_num=1)
        
        # 测试多次请求
        test_multiple_requests(client, num_requests=5)
        
        print("\n" + "=" * 70)
        print("✅ 所有测试完成")
        print("=" * 70)
        
        # 获取统计信息
        print("\n📊 客户端统计:")
        stats = client.get_statistics()
        print(f"   总请求数: {stats.get('total_requests', 0)}")
        print(f"   成功请求: {stats.get('successful_requests', 0)}")
        print(f"   失败请求: {stats.get('failed_requests', 0)}")
        print(f"   平均网络时间: {stats.get('avg_network_time_ms', 0):.2f} ms")
        print(f"   平均计算时间: {stats.get('avg_compute_time_ms', 0):.2f} ms")
        
    finally:
        print("\n🔌 关闭连接...")
        client.close()
        print("   ✓ 连接已关闭")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

