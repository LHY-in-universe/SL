#!/usr/bin/env python3
"""
测试 gRPC 客户端功能

测试内容：
1. 客户端创建和连接
2. 健康检查
3. 服务信息查询
4. 计算请求
5. 多次请求
6. 错误处理
7. 统计信息
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

from splitlearn_comm import GRPCComputeClient, GRPCComputeServer
from splitlearn_comm.core import ModelComputeFunction

# 测试配置
TEST_PORT = 50054
TEST_HOST = "localhost"
MODEL_PATH = os.path.join(current_dir, "gpt2_trunk_full.pt")

# 全局服务器变量（用于后台运行）
_server = None
_server_thread = None


def start_test_server():
    """启动测试服务器（后台）"""
    global _server, _server_thread
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        return False
    
    # 加载模型
    model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model.eval()
    
    # 创建计算函数
    compute_fn = ModelComputeFunction(
        model=model,
        device="cpu",
        model_name="gpt2-trunk-test"
    )
    
    # 创建服务器
    _server = GRPCComputeServer(
        compute_fn=compute_fn,
        host="0.0.0.0",
        port=TEST_PORT,
        max_workers=1
    )
    
    def run_server():
        _server.start()
        _server.wait_for_termination()
    
    _server_thread = threading.Thread(target=run_server, daemon=True)
    _server_thread.start()
    
    # 等待服务器启动
    time.sleep(3)
    return True


def stop_test_server():
    """停止测试服务器"""
    global _server
    if _server:
        try:
            _server.stop(grace=2)
        except:
            pass


def test_client_creation():
    """测试 1: 客户端创建"""
    print("=" * 70)
    print("测试 1: 客户端创建")
    print("=" * 70)
    
    try:
        # 创建客户端
        print(f"\n创建 gRPC 客户端 (服务器: {TEST_HOST}:{TEST_PORT})...")
        client = GRPCComputeClient(
            server_address=f"{TEST_HOST}:{TEST_PORT}",
            timeout=10.0
        )
        print("✓ 客户端创建成功")
        
        # 检查客户端属性
        print(f"\n客户端信息:")
        print(f"  服务器地址: {client.server_address}")
        print(f"  超时时间: {client.timeout} 秒")
        print(f"  最大消息长度: {client.max_message_length / (1024*1024):.1f} MB")
        
        return client
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_client_connection(client):
    """测试 2: 客户端连接"""
    print("\n" + "=" * 70)
    print("测试 2: 客户端连接")
    print("=" * 70)
    
    try:
        # 连接服务器
        print("\n连接服务器...")
        if client.connect():
            print("✓ 连接成功")
            
            # 检查连接状态
            print(f"\n连接状态:")
            print(f"  已连接: {client.channel is not None}")
            print(f"  Stub 已创建: {client.stub is not None}")
            
            return True
        else:
            print("❌ 连接失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_health_check(client):
    """测试 3: 健康检查"""
    print("\n" + "=" * 70)
    print("测试 3: 健康检查")
    print("=" * 70)
    
    try:
        print("\n执行健康检查...")
        is_healthy = client.health_check()
        
        if is_healthy:
            print("✓ 服务器健康")
        else:
            print("⚠️  服务器不健康")
        
        return is_healthy
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_service_info(client):
    """测试 4: 服务信息查询"""
    print("\n" + "=" * 70)
    print("测试 4: 服务信息查询")
    print("=" * 70)
    
    try:
        print("\n获取服务信息...")
        info = client.get_service_info()
        
        print("✓ 服务信息获取成功")
        print(f"\n服务信息:")
        print(f"  服务名: {info.get('service_name', 'N/A')}")
        print(f"  版本: {info.get('version', 'N/A')}")
        print(f"  设备: {info.get('device', 'N/A')}")
        print(f"  总请求数: {info.get('total_requests', 0)}")
        print(f"  运行时间: {info.get('uptime_seconds', 0):.1f} 秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compute_request(client):
    """测试 5: 计算请求"""
    print("\n" + "=" * 70)
    print("测试 5: 计算请求")
    print("=" * 70)
    
    try:
        # 创建测试输入
        test_input = torch.randn(1, 10, 768)
        print(f"\n测试输入:")
        print(f"  形状: {test_input.shape}")
        print(f"  大小: {test_input.numel() * 4 / 1024:.2f} KB")
        
        # 发送计算请求
        print("\n发送计算请求...")
        start_time = time.time()
        output = client.compute(test_input)
        elapsed = time.time() - start_time
        
        print(f"✓ 计算完成")
        print(f"  输出形状: {output.shape}")
        print(f"  输出大小: {output.numel() * 4 / 1024:.2f} KB")
        print(f"  总耗时: {elapsed*1000:.2f} ms")
        
        # 验证输出
        if output.shape == test_input.shape:
            print("✓ 输出形状正确")
        else:
            print(f"⚠️  输出形状不符合预期: {output.shape} (期望: {test_input.shape})")
        
        # 测试不同形状
        print("\n测试不同形状的输入...")
        test_cases = [
            (1, 5, 768),   # 短序列
            (1, 20, 768),  # 长序列
        ]
        
        for i, shape in enumerate(test_cases, 1):
            test_input = torch.randn(*shape)
            try:
                output = client.compute(test_input)
                if output.shape == shape:
                    print(f"  ✓ 测试 {i}: {shape} → {output.shape}")
                else:
                    print(f"  ⚠️  测试 {i}: {shape} → {output.shape}")
            except Exception as e:
                print(f"  ❌ 测试 {i} 失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_requests(client):
    """测试 6: 多次请求"""
    print("\n" + "=" * 70)
    print("测试 6: 多次请求")
    print("=" * 70)
    
    try:
        num_requests = 5
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
        
        return successes == num_requests
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_statistics(client):
    """测试 7: 统计信息"""
    print("\n" + "=" * 70)
    print("测试 7: 统计信息")
    print("=" * 70)
    
    try:
        print("\n获取客户端统计信息...")
        stats = client.get_statistics()
        
        print("✓ 统计信息获取成功")
        print(f"\n统计信息:")
        print(f"  总请求数: {stats.get('total_requests', 0)}")
        print(f"  成功请求: {stats.get('successful_requests', 0)}")
        print(f"  失败请求: {stats.get('failed_requests', 0)}")
        print(f"  平均网络时间: {stats.get('avg_network_time_ms', 0):.2f} ms")
        print(f"  平均计算时间: {stats.get('avg_compute_time_ms', 0):.2f} ms")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling(client):
    """测试 8: 错误处理"""
    print("\n" + "=" * 70)
    print("测试 8: 错误处理")
    print("=" * 70)
    
    try:
        # 测试连接断开后的行为
        print("\n测试连接断开...")
        client.close()
        
        try:
            output = client.compute(torch.randn(1, 5, 768))
            print("⚠️  连接断开后仍能计算（不应该发生）")
            return False
        except Exception as e:
            print(f"✓ 正确检测到连接断开: {type(e).__name__}")
            return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_manager():
    """测试 9: 上下文管理器"""
    print("\n" + "=" * 70)
    print("测试 9: 客户端上下文管理器")
    print("=" * 70)
    
    try:
        print("\n使用上下文管理器...")
        with GRPCComputeClient(
            server_address=f"{TEST_HOST}:{TEST_PORT}",
            timeout=10.0
        ) as client:
            if client.connect():
                print("✓ 客户端在上下文中连接成功")
                
                # 执行一次计算
                test_input = torch.randn(1, 5, 768)
                output = client.compute(test_input)
                print(f"✓ 计算成功: {test_input.shape} → {output.shape}")
            else:
                print("❌ 连接失败")
                return False
        
        print("✓ 上下文管理器退出，客户端自动关闭")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("gRPC 客户端功能测试")
    print("=" * 70)
    print(f"\n测试配置:")
    print(f"  服务器地址: {TEST_HOST}:{TEST_PORT}")
    print(f"  模型文件: {MODEL_PATH}")
    print()
    
    # 启动测试服务器
    print("启动测试服务器...")
    if not start_test_server():
        print("❌ 无法启动测试服务器")
        return 1
    
    print("✓ 测试服务器已启动")
    time.sleep(2)
    
    results = {}
    
    try:
        # 测试 1: 客户端创建
        client = test_client_creation()
        if client is None:
            print("\n❌ 客户端创建失败，无法继续测试")
            return 1
        
        results['creation'] = client is not None
        
        # 测试 2: 客户端连接
        results['connection'] = test_client_connection(client)
        
        # 测试 3: 健康检查
        results['health_check'] = test_health_check(client)
        
        # 测试 4: 服务信息
        results['service_info'] = test_service_info(client)
        
        # 测试 5: 计算请求
        results['compute'] = test_compute_request(client)
        
        # 测试 6: 多次请求
        results['multiple'] = test_multiple_requests(client)
        
        # 测试 7: 统计信息
        results['statistics'] = test_statistics(client)
        
        # 测试 8: 错误处理
        results['error_handling'] = test_error_handling(client)
        
        # 测试 9: 上下文管理器
        results['context_manager'] = test_context_manager()
        
    finally:
        # 停止测试服务器
        print("\n停止测试服务器...")
        stop_test_server()
        time.sleep(1)
        print("✓ 测试服务器已停止")
    
    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "❌ 失败"
        print(f"  {test_name:20s}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！gRPC 客户端功能正常！")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        stop_test_server()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        stop_test_server()
        sys.exit(1)

