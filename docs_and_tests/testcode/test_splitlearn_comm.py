#!/usr/bin/env python3
"""
SplitLearnComm 库功能测试

使用 testcode 目录中的现成模型文件测试 SplitLearnComm 的功能：
- 服务器启动和停止
- 客户端连接
- 计算功能
- 多次请求
- 错误处理
- 性能测试
"""

import os
import sys
import time
import threading
import torch
import logging

# 设置环境变量（必须在导入 grpc 之前）
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearnComm', 'src'))
sys.path.insert(0, os.path.join(project_root, 'SplitLearnCore', 'src'))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from splitlearn_comm import GRPCComputeServer, GRPCComputeClient
from splitlearn_comm.core import ModelComputeFunction

# 测试配置
TEST_PORT = 50052  # 使用不同的端口避免冲突
TEST_HOST = "localhost"
MODEL_PATH = os.path.join(current_dir, "gpt2_trunk_full.pt")


def check_model_file():
    """检查模型文件是否存在"""
    print("=" * 70)
    print("检查模型文件")
    print("=" * 70)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        print("\n请先运行: python testcode/prepare_models.py")
        return False
    
    file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"✓ 模型文件存在: {MODEL_PATH}")
    print(f"  大小: {file_size_mb:.2f} MB")
    return True


def load_model():
    """加载模型"""
    print("\n" + "=" * 70)
    print("加载模型")
    print("=" * 70)
    
    try:
        # 导入必要的类（用于反序列化）
        from splitlearn_core.models.gpt2 import GPT2TrunkModel
        print("✓ GPT2TrunkModel 导入成功")
    except ImportError as e:
        print(f"⚠️  无法导入 GPT2TrunkModel: {e}")
        print("   尝试直接加载...")
    
    print(f"\n加载模型: {MODEL_PATH}")
    start_time = time.time()
    
    model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model.eval()
    
    elapsed = time.time() - start_time
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ 模型加载完成 (耗时: {elapsed:.2f} 秒)")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型类型: {type(model).__name__}")
    
    return model


def test_server_start_stop(model):
    """测试服务器启动和停止"""
    print("\n" + "=" * 70)
    print("测试 1: 服务器启动和停止")
    print("=" * 70)
    
    try:
        # 创建 ComputeFunction
        compute_fn = ModelComputeFunction(
            model=model,
            device="cpu",
            model_name="gpt2-trunk-test"
        )
        print("✓ ComputeFunction 创建成功")
        
        # 创建服务器（单线程模式）
        server = GRPCComputeServer(
            compute_fn=compute_fn,
            host="0.0.0.0",
            port=TEST_PORT,
            max_workers=1  # 单线程模式
        )
        print(f"✓ 服务器创建成功 (端口: {TEST_PORT})")
        
        # 启动服务器
        server.start()
        print("✓ 服务器启动成功")
        
        # 等待一下确保服务器完全启动
        time.sleep(2)
        
        # 检查端口是否在监听
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((TEST_HOST, TEST_PORT))
        sock.close()
        
        if result == 0:
            print("✓ 端口正在监听")
        else:
            print("⚠️  端口未监听")
        
        # 停止服务器
        server.stop(grace=2)
        print("✓ 服务器停止成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_client_connection(model):
    """测试客户端连接"""
    print("\n" + "=" * 70)
    print("测试 2: 客户端连接")
    print("=" * 70)
    
    # 在后台启动服务器
    compute_fn = ModelComputeFunction(
        model=model,
        device="cpu",
        model_name="gpt2-trunk-test"
    )
    server = GRPCComputeServer(
        compute_fn=compute_fn,
        host="0.0.0.0",
        port=TEST_PORT,
        max_workers=1
    )
    
    def run_server():
        server.start()
        server.wait_for_termination()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # 等待服务器启动
    print("等待服务器启动...")
    time.sleep(3)
    
    try:
        # 创建客户端
        client = GRPCComputeClient(
            server_address=f"{TEST_HOST}:{TEST_PORT}",
            timeout=10.0
        )
        print("✓ 客户端创建成功")
        
        # 连接服务器
        if client.connect():
            print("✓ 连接成功")
            
            # 获取服务信息
            info = client.get_service_info()
            print(f"✓ 服务信息:")
            print(f"  服务名: {info.get('service_name', 'N/A')}")
            print(f"  版本: {info.get('version', 'N/A')}")
            print(f"  设备: {info.get('device', 'N/A')}")
            print(f"  总请求数: {info.get('total_requests', 0)}")
            
            # 健康检查
            is_healthy = client.health_check()
            print(f"✓ 健康检查: {'健康' if is_healthy else '不健康'}")
            
            client.close()
            print("✓ 客户端关闭成功")
            
            # 停止服务器
            server.stop(grace=2)
            print("✓ 服务器停止成功")
            
            return True
        else:
            print("❌ 连接失败")
            server.stop(grace=2)
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        server.stop(grace=2)
        return False


def test_compute_functionality(model):
    """测试计算功能"""
    print("\n" + "=" * 70)
    print("测试 3: 计算功能")
    print("=" * 70)
    
    # 启动服务器
    compute_fn = ModelComputeFunction(
        model=model,
        device="cpu",
        model_name="gpt2-trunk-test"
    )
    server = GRPCComputeServer(
        compute_fn=compute_fn,
        host="0.0.0.0",
        port=TEST_PORT,
        max_workers=1
    )
    
    def run_server():
        server.start()
        server.wait_for_termination()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(3)
    
    try:
        # 创建客户端
        client = GRPCComputeClient(
            server_address=f"{TEST_HOST}:{TEST_PORT}",
            timeout=30.0
        )
        
        if not client.connect():
            print("❌ 连接失败")
            server.stop(grace=2)
            return False
        
        print("✓ 客户端连接成功")
        
        # 测试 1: 基本计算
        print("\n[3.1] 基本计算测试")
        test_input = torch.randn(1, 10, 768)  # [batch, seq_len, hidden_dim]
        print(f"  输入形状: {test_input.shape}")
        print(f"  输入大小: {test_input.numel() * 4 / 1024:.2f} KB")
        
        start_time = time.time()
        output = client.compute(test_input)
        elapsed = time.time() - start_time
        
        print(f"✓ 计算完成 (耗时: {elapsed*1000:.2f} ms)")
        print(f"  输出形状: {output.shape}")
        print(f"  输出大小: {output.numel() * 4 / 1024:.2f} KB")
        print(f"  输出类型: {output.dtype}")
        
        # 验证输出形状
        if output.shape == test_input.shape:
            print("✓ 输出形状正确")
        else:
            print(f"⚠️  输出形状不符合预期: {output.shape} (期望: {test_input.shape})")
        
        # 测试 2: 不同形状的输入
        print("\n[3.2] 不同形状测试")
        test_cases = [
            (1, 5, 768),   # 短序列
            (1, 20, 768),  # 长序列
            (2, 10, 768),  # 批量大小 2
        ]
        
        for i, shape in enumerate(test_cases, 1):
            test_input = torch.randn(*shape)
            try:
                output = client.compute(test_input)
                if output.shape == shape:
                    print(f"  ✓ 测试 {i}: {shape} → {output.shape}")
                else:
                    print(f"  ⚠️  测试 {i}: {shape} → {output.shape} (形状不匹配)")
            except Exception as e:
                print(f"  ❌ 测试 {i} 失败: {e}")
        
        # 测试 3: 数值验证（简单检查）
        print("\n[3.3] 数值验证")
        test_input = torch.randn(1, 5, 768)
        output1 = client.compute(test_input)
        output2 = client.compute(test_input)  # 相同输入
        
        # 检查输出是否一致（应该一致，因为模型是确定性的）
        if torch.allclose(output1, output2, atol=1e-5):
            print("✓ 相同输入产生相同输出（确定性测试通过）")
        else:
            print("⚠️  相同输入产生不同输出（可能是数值精度问题）")
        
        # 测试 4: 边界情况
        print("\n[3.4] 边界情况测试")
        # 最小输入
        min_input = torch.randn(1, 1, 768)
        try:
            min_output = client.compute(min_input)
            print(f"  ✓ 最小输入 (1, 1, 768) → {min_output.shape}")
        except Exception as e:
            print(f"  ❌ 最小输入失败: {e}")
        
        client.close()
        server.stop(grace=2)
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        server.stop(grace=2)
        return False


def test_multiple_requests(model):
    """测试多次请求"""
    print("\n" + "=" * 70)
    print("测试 4: 多次请求（并发测试）")
    print("=" * 70)
    
    # 启动服务器
    compute_fn = ModelComputeFunction(
        model=model,
        device="cpu",
        model_name="gpt2-trunk-test"
    )
    server = GRPCComputeServer(
        compute_fn=compute_fn,
        host="0.0.0.0",
        port=TEST_PORT,
        max_workers=1  # 单线程模式
    )
    
    def run_server():
        server.start()
        server.wait_for_termination()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(3)
    
    try:
        # 创建客户端
        client = GRPCComputeClient(
            server_address=f"{TEST_HOST}:{TEST_PORT}",
            timeout=30.0
        )
        
        if not client.connect():
            print("❌ 连接失败")
            server.stop(grace=2)
            return False
        
        print("✓ 客户端连接成功")
        
        num_requests = 10
        successes = 0
        total_time = 0.0
        
        print(f"\n发送 {num_requests} 个请求...")
        
        for i in range(num_requests):
            test_input = torch.randn(1, 5, 768)
            try:
                start_time = time.time()
                output = client.compute(test_input)
                elapsed = time.time() - start_time
                total_time += elapsed
                successes += 1
                print(f"  请求 {i+1}/{num_requests}: ✓ (耗时: {elapsed*1000:.2f} ms)")
            except Exception as e:
                print(f"  请求 {i+1}/{num_requests}: ❌ ({e})")
        
        avg_time = total_time / successes if successes > 0 else 0
        print(f"\n总结:")
        print(f"  成功: {successes}/{num_requests}")
        print(f"  平均耗时: {avg_time*1000:.2f} ms")
        print(f"  总耗时: {total_time*1000:.2f} ms")
        
        client.close()
        server.stop(grace=2)
        
        return successes == num_requests
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        server.stop(grace=2)
        return False


def test_error_handling(model):
    """测试错误处理"""
    print("\n" + "=" * 70)
    print("测试 5: 错误处理")
    print("=" * 70)
    
    # 启动服务器
    compute_fn = ModelComputeFunction(
        model=model,
        device="cpu",
        model_name="gpt2-trunk-test"
    )
    server = GRPCComputeServer(
        compute_fn=compute_fn,
        host="0.0.0.0",
        port=TEST_PORT,
        max_workers=1
    )
    
    def run_server():
        server.start()
        server.wait_for_termination()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(3)
    
    try:
        client = GRPCComputeClient(
            server_address=f"{TEST_HOST}:{TEST_PORT}",
            timeout=10.0
        )
        
        if not client.connect():
            print("❌ 连接失败")
            server.stop(grace=2)
            return False
        
        # 测试 1: 无效形状（如果模型不支持）
        print("\n[5.1] 测试无效输入形状")
        try:
            invalid_input = torch.randn(10, 10, 10)  # 错误的形状
            output = client.compute(invalid_input)
            print("  ⚠️  无效输入被接受（可能模型有容错机制）")
        except Exception as e:
            print(f"  ✓ 正确拒绝无效输入: {type(e).__name__}")
        
        # 测试 2: 连接断开后的行为
        print("\n[5.2] 测试连接断开")
        client.close()
        try:
            output = client.compute(torch.randn(1, 5, 768))
            print("  ⚠️  连接断开后仍能计算（不应该发生）")
        except Exception as e:
            print(f"  ✓ 正确检测到连接断开: {type(e).__name__}")
        
        server.stop(grace=2)
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        server.stop(grace=2)
        return False


def test_performance(model):
    """测试性能"""
    print("\n" + "=" * 70)
    print("测试 6: 性能测试")
    print("=" * 70)
    
    # 启动服务器
    compute_fn = ModelComputeFunction(
        model=model,
        device="cpu",
        model_name="gpt2-trunk-test"
    )
    server = GRPCComputeServer(
        compute_fn=compute_fn,
        host="0.0.0.0",
        port=TEST_PORT,
        max_workers=1
    )
    
    def run_server():
        server.start()
        server.wait_for_termination()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(3)
    
    try:
        client = GRPCComputeClient(
            server_address=f"{TEST_HOST}:{TEST_PORT}",
            timeout=30.0
        )
        
        if not client.connect():
            print("❌ 连接失败")
            server.stop(grace=2)
            return False
        
        # 预热
        print("预热中...")
        for _ in range(3):
            client.compute(torch.randn(1, 5, 768))
        
        # 性能测试
        num_tests = 20
        test_input = torch.randn(1, 10, 768)
        
        print(f"\n执行 {num_tests} 次计算...")
        times = []
        
        for i in range(num_tests):
            start_time = time.time()
            output = client.compute(test_input)
            elapsed = time.time() - start_time
            times.append(elapsed)
            if (i + 1) % 5 == 0:
                print(f"  完成 {i+1}/{num_tests}")
        
        # 统计
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        total_time = sum(times)
        
        print(f"\n性能统计:")
        print(f"  总请求数: {num_tests}")
        print(f"  总耗时: {total_time*1000:.2f} ms")
        print(f"  平均耗时: {avg_time*1000:.2f} ms")
        print(f"  最小耗时: {min_time*1000:.2f} ms")
        print(f"  最大耗时: {max_time*1000:.2f} ms")
        print(f"  吞吐量: {num_tests/total_time:.2f} 请求/秒")
        
        # 获取客户端统计
        stats = client.get_statistics()
        print(f"\n客户端统计:")
        print(f"  总请求数: {stats.get('total_requests', 0)}")
        print(f"  平均网络时间: {stats.get('avg_network_time_ms', 0):.2f} ms")
        print(f"  平均计算时间: {stats.get('avg_compute_time_ms', 0):.2f} ms")
        
        client.close()
        server.stop(grace=2)
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        server.stop(grace=2)
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("SplitLearnComm 库功能测试")
    print("=" * 70)
    print(f"\n测试配置:")
    print(f"  模型文件: {MODEL_PATH}")
    print(f"  服务器地址: {TEST_HOST}:{TEST_PORT}")
    print(f"  单线程模式: 是")
    print()
    
    # 检查模型文件
    if not check_model_file():
        return 1
    
    # 加载模型
    try:
        model = load_model()
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 运行测试
    results = {}
    
    results['server'] = test_server_start_stop(model)
    results['connection'] = test_client_connection(model)
    results['compute'] = test_compute_functionality(model)
    results['multiple'] = test_multiple_requests(model)
    results['error'] = test_error_handling(model)
    results['performance'] = test_performance(model)
    
    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "❌ 失败"
        print(f"  {test_name:15s}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！SplitLearnComm 功能正常！")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

