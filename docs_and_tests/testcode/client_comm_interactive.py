#!/usr/bin/env python3
"""
gRPC 客户端交互式测试脚本

允许用户手动输入信息进行测试：
- 自定义输入数据
- 选择不同的测试场景
- 实时查看结果
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

# 注意：交互式客户端可以使用同步版本，因为它只是发送请求
# 如果需要异步版本，可以使用 client_comm_simple_async.py
from splitlearn_comm import GRPCComputeClient

# 测试配置
DEFAULT_SERVER = "localhost:50056"  # 简单服务器端口
TIMEOUT = 30.0


def print_separator():
    """打印分隔线"""
    print("\n" + "=" * 70)


def print_tensor_info(tensor, name="张量"):
    """打印张量详细信息"""
    print(f"\n📊 {name}信息:")
    print(f"   形状: {tensor.shape}")
    print(f"   数据类型: {tensor.dtype}")
    print(f"   数据大小: {tensor.numel() * 4 / 1024:.2f} KB")
    print(f"   最小值: {tensor.min().item():.6f}")
    print(f"   最大值: {tensor.max().item():.6f}")
    print(f"   平均值: {tensor.mean().item():.6f}")
    print(f"   标准差: {tensor.std().item():.6f}")


def create_custom_tensor():
    """创建自定义张量"""
    print_separator()
    print("📝 创建自定义张量")
    print_separator()
    
    print("\n请选择创建方式：")
    print("1. 随机张量（指定形状）")
    print("2. 全零张量（指定形状）")
    print("3. 全一张量（指定形状）")
    print("4. 手动输入数值（1D 张量）")
    print("5. 使用预设形状")
    
    choice = input("\n请选择 (1-5): ").strip()
    
    if choice == "1":
        # 随机张量
        print("\n请输入形状（用空格分隔，例如: 1 10 768）:")
        shape_str = input("形状: ").strip()
        try:
            shape = tuple(map(int, shape_str.split()))
            tensor = torch.randn(*shape)
            print(f"✓ 创建随机张量: {shape}")
            return tensor
        except Exception as e:
            print(f"❌ 输入错误: {e}")
            return None
    
    elif choice == "2":
        # 全零张量
        print("\n请输入形状（用空格分隔，例如: 1 10 768）:")
        shape_str = input("形状: ").strip()
        try:
            shape = tuple(map(int, shape_str.split()))
            tensor = torch.zeros(*shape)
            print(f"✓ 创建全零张量: {shape}")
            return tensor
        except Exception as e:
            print(f"❌ 输入错误: {e}")
            return None
    
    elif choice == "3":
        # 全一张量
        print("\n请输入形状（用空格分隔，例如: 1 10 768）:")
        shape_str = input("形状: ").strip()
        try:
            shape = tuple(map(int, shape_str.split()))
            tensor = torch.ones(*shape)
            print(f"✓ 创建全一张量: {shape}")
            return tensor
        except Exception as e:
            print(f"❌ 输入错误: {e}")
            return None
    
    elif choice == "4":
        # 手动输入数值
        print("\n请输入数值（用空格分隔，例如: 1.0 2.0 3.0）:")
        values_str = input("数值: ").strip()
        try:
            values = list(map(float, values_str.split()))
            tensor = torch.tensor(values)
            print(f"✓ 创建张量: {tensor.shape}")
            return tensor
        except Exception as e:
            print(f"❌ 输入错误: {e}")
            return None
    
    elif choice == "5":
        # 预设形状
        print("\n请选择预设形状：")
        print("1. (1, 10, 768)  - 小张量")
        print("2. (1, 20, 768)  - 中等张量")
        print("3. (2, 10, 768)  - 批量=2")
        print("4. (1, 5, 768)   - 短序列")
        print("5. (1, 50, 768)  - 长序列")
        
        preset = input("请选择 (1-5): ").strip()
        presets = {
            "1": (1, 10, 768),
            "2": (1, 20, 768),
            "3": (2, 10, 768),
            "4": (1, 5, 768),
            "5": (1, 50, 768),
        }
        
        if preset in presets:
            shape = presets[preset]
            tensor = torch.randn(*shape)
            print(f"✓ 创建随机张量: {shape}")
            return tensor
        else:
            print("❌ 无效选择")
            return None
    
    else:
        print("❌ 无效选择")
        return None


def send_single_request(client, input_tensor):
    """发送单个请求"""
    print_separator()
    print("📤 发送请求")
    print_separator()
    
    # 显示输入数据
    print_tensor_info(input_tensor, "输入")
    
    # 发送请求
    print(f"\n🚀 正在发送请求...")
    start_time = time.time()
    
    try:
        output = client.compute(input_tensor)
        total_time = (time.time() - start_time) * 1000
        
        print(f"✓ 请求成功！")
        
        # 显示输出数据
        print_tensor_info(output, "输出")
        
        # 验证结果（简单服务器：output = input * 2 + 1）
        expected = input_tensor * 2 + 1
        if torch.allclose(output, expected, atol=1e-5):
            print(f"\n✅ 计算结果正确: output = input * 2 + 1")
        else:
            print(f"\n⚠️  计算结果不符合预期")
            print(f"   预期范围: [{expected.min().item():.6f}, {expected.max().item():.6f}]")
            print(f"   实际范围: [{output.min().item():.6f}, {output.max().item():.6f}]")
        
        # 传输统计
        input_size_kb = input_tensor.numel() * 4 / 1024
        output_size_kb = output.numel() * 4 / 1024
        total_size_kb = input_size_kb + output_size_kb
        
        print(f"\n📡 传输统计:")
        print(f"   发送数据: {input_size_kb:.2f} KB")
        print(f"   接收数据: {output_size_kb:.2f} KB")
        print(f"   总传输: {total_size_kb:.2f} KB")
        print(f"   总耗时: {total_time:.2f} ms")
        if total_time > 0:
            print(f"   吞吐量: {total_size_kb / (total_time / 1000):.2f} KB/s")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 请求失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def send_multiple_requests(client):
    """发送多个请求"""
    print_separator()
    print("🔄 多次请求测试")
    print_separator()
    
    num_requests = input("\n请输入请求数量 (默认 5): ").strip()
    num_requests = int(num_requests) if num_requests else 5
    
    print(f"\n将发送 {num_requests} 个请求...")
    
    # 创建测试输入
    print("\n请选择输入数据：")
    print("1. 每次使用相同的随机张量")
    print("2. 每次使用不同的随机张量")
    
    choice = input("请选择 (1-2): ").strip()
    
    if choice == "1":
        # 相同的张量
        test_input = torch.randn(1, 10, 768)
        print(f"✓ 使用固定张量: {test_input.shape}")
    elif choice == "2":
        # 不同的张量
        test_input = None
        print("✓ 每次使用不同的随机张量")
    else:
        print("❌ 无效选择，使用默认")
        test_input = torch.randn(1, 10, 768)
    
    successes = 0
    total_time = 0.0
    
    for i in range(num_requests):
        print(f"\n--- 请求 {i+1}/{num_requests} ---")
        
        if test_input is None:
            # 每次创建新的随机张量
            current_input = torch.randn(1, 10, 768)
        else:
            current_input = test_input
        
        start_time = time.time()
        try:
            output = client.compute(current_input)
            elapsed = (time.time() - start_time) * 1000
            total_time += elapsed
            successes += 1
            print(f"   ✓ 成功 (耗时: {elapsed:.2f} ms)")
        except Exception as e:
            print(f"   ❌ 失败: {e}")
    
    print(f"\n📊 总结:")
    print(f"   成功: {successes}/{num_requests}")
    if successes > 0:
        print(f"   总耗时: {total_time:.2f} ms")
        print(f"   平均耗时: {total_time/successes:.2f} ms")


def show_statistics(client):
    """显示统计信息"""
    print_separator()
    print("📊 客户端统计信息")
    print_separator()
    
    stats = client.get_statistics()
    
    print(f"\n总请求数: {stats.get('total_requests', 0)}")
    print(f"平均网络时间: {stats.get('avg_network_time_ms', 0):.2f} ms")
    print(f"平均计算时间: {stats.get('avg_compute_time_ms', 0):.2f} ms")
    print(f"平均总时间: {stats.get('avg_total_time_ms', 0):.2f} ms")


def show_service_info(client):
    """显示服务信息"""
    print_separator()
    print("ℹ️  服务器信息")
    print_separator()
    
    try:
        info = client.get_service_info()
        
        if info:
            print(f"\n服务名: {info.get('service_name', 'N/A')}")
            print(f"版本: {info.get('version', 'N/A')}")
            print(f"设备: {info.get('device', 'N/A')}")
            print(f"总请求数: {info.get('total_requests', 0)}")
            print(f"运行时间: {info.get('uptime_seconds', 0):.1f} 秒")
        else:
            print("❌ 无法获取服务器信息")
    except Exception as e:
        print(f"❌ 获取服务器信息失败: {e}")


def main_menu(client):
    """主菜单"""
    while True:
        print_separator()
        print("📋 主菜单")
        print_separator()
        
        print("\n请选择操作：")
        print("1. 发送单个请求（使用自定义输入）")
        print("2. 发送单个请求（使用预设输入）")
        print("3. 发送多个请求")
        print("4. 查看客户端统计")
        print("5. 查看服务器信息")
        print("6. 退出")
        
        choice = input("\n请选择 (1-6): ").strip()
        
        if choice == "1":
            # 自定义输入
            input_tensor = create_custom_tensor()
            if input_tensor is not None:
                send_single_request(client, input_tensor)
        
        elif choice == "2":
            # 预设输入
            input_tensor = torch.randn(1, 10, 768)
            print(f"\n✓ 使用预设输入: {input_tensor.shape}")
            send_single_request(client, input_tensor)
        
        elif choice == "3":
            # 多次请求
            send_multiple_requests(client)
        
        elif choice == "4":
            # 统计信息
            show_statistics(client)
        
        elif choice == "5":
            # 服务器信息
            show_service_info(client)
        
        elif choice == "6":
            # 退出
            print("\n👋 再见！")
            break
        
        else:
            print("❌ 无效选择，请重试")
        
        # 等待用户确认
        input("\n按 Enter 继续...")


def main():
    print("\n" + "=" * 70)
    print("💻 gRPC 客户端交互式测试")
    print("=" * 70)
    
    # 获取服务器地址
    server_address = input(f"\n请输入服务器地址 (默认: {DEFAULT_SERVER}): ").strip()
    if not server_address:
        server_address = DEFAULT_SERVER
    
    print(f"\n📡 连接服务器: {server_address}")
    print("   正在连接...")
    
    # 创建客户端
    client = GRPCComputeClient(
        server_address=server_address,
        timeout=TIMEOUT
    )
    
    # 连接服务器
    if not client.connect():
        print("   ❌ 连接失败！")
        print(f"\n💡 请确保服务器正在运行:")
        print(f"   python testcode/server_comm_simple.py")
        return 1
    
    print("   ✓ 连接成功！")
    
    try:
        # 显示主菜单
        main_menu(client)
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

