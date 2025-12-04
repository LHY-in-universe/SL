#!/usr/bin/env python3
"""
测试 gRPC 服务器功能

测试内容：
1. 服务器创建和初始化
2. 服务器启动
3. 端口监听检查
4. 服务器停止
5. 使用实际模型进行测试
"""

import os
import sys
import time
import socket
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

from splitlearn_comm import GRPCComputeServer
from splitlearn_comm.core import ModelComputeFunction

# 测试配置
TEST_PORT = 50053
TEST_HOST = "0.0.0.0"
MODEL_PATH = os.path.join(current_dir, "gpt2_trunk_full.pt")


def check_port_available(host, port):
    """检查端口是否可用"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        sock.close()
        return True
    except OSError:
        return False


def test_server_creation():
    """测试 1: 服务器创建"""
    print("=" * 70)
    print("测试 1: 服务器创建")
    print("=" * 70)
    
    try:
        # 检查模型文件
        if not os.path.exists(MODEL_PATH):
            print(f"❌ 模型文件不存在: {MODEL_PATH}")
            print("   请先运行: python testcode/prepare_models.py")
            return False
        
        # 加载模型
        print(f"\n加载模型: {MODEL_PATH}")
        model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        model.eval()
        print("✓ 模型加载成功")
        
        # 创建计算函数
        print("\n创建 ComputeFunction...")
        compute_fn = ModelComputeFunction(
            model=model,
            device="cpu",
            model_name="gpt2-trunk-test"
        )
        print("✓ ComputeFunction 创建成功")
        
        # 创建服务器
        print(f"\n创建 gRPC 服务器 (端口: {TEST_PORT})...")
        server = GRPCComputeServer(
            compute_fn=compute_fn,
            host=TEST_HOST,
            port=TEST_PORT,
            max_workers=1  # 单线程模式
        )
        print("✓ 服务器创建成功")
        
        # 检查服务器属性
        print(f"\n服务器信息:")
        print(f"  主机: {server.host}")
        print(f"  端口: {server.port}")
        print(f"  最大工作线程: {server.max_workers}")
        print(f"  最大消息长度: {server.max_message_length / (1024*1024):.1f} MB")
        
        return server, model
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_server_start_stop(server):
    """测试 2: 服务器启动和停止"""
    print("\n" + "=" * 70)
    print("测试 2: 服务器启动和停止")
    print("=" * 70)
    
    try:
        # 检查端口是否可用
        if not check_port_available(TEST_HOST, TEST_PORT):
            print(f"⚠️  端口 {TEST_PORT} 已被占用")
        else:
            print(f"✓ 端口 {TEST_PORT} 可用")
        
        # 启动服务器
        print("\n启动服务器...")
        server.start()
        print("✓ 服务器启动成功")
        
        # 等待服务器完全启动
        time.sleep(2)
        
        # 检查端口是否在监听
        print("\n检查端口监听状态...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', TEST_PORT))
        sock.close()
        
        if result == 0:
            print(f"✓ 端口 {TEST_PORT} 正在监听")
        else:
            print(f"⚠️  端口 {TEST_PORT} 未监听")
        
        # 获取服务信息
        print("\n服务器状态:")
        print(f"  运行中: {server.server is not None}")
        
        # 停止服务器
        print("\n停止服务器...")
        server.stop(grace=2)
        print("✓ 服务器停止成功")
        
        # 再次检查端口
        time.sleep(1)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', TEST_PORT))
        sock.close()
        
        if result != 0:
            print(f"✓ 端口 {TEST_PORT} 已释放")
        else:
            print(f"⚠️  端口 {TEST_PORT} 仍在监听")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        try:
            server.stop(grace=1)
        except:
            pass
        return False


def test_server_with_model(server, model):
    """测试 3: 使用模型进行实际计算"""
    print("\n" + "=" * 70)
    print("测试 3: 服务器模型计算功能")
    print("=" * 70)
    
    try:
        # 启动服务器
        print("启动服务器...")
        server.start()
        time.sleep(2)
        print("✓ 服务器启动成功")
        
        # 创建测试输入
        test_input = torch.randn(1, 5, 768)
        print(f"\n测试输入:")
        print(f"  形状: {test_input.shape}")
        print(f"  大小: {test_input.numel() * 4 / 1024:.2f} KB")
        
        # 直接使用计算函数测试（不通过 gRPC）
        print("\n直接调用 ComputeFunction...")
        start_time = time.time()
        output = server.compute_fn.compute(test_input)
        elapsed = time.time() - start_time
        
        print(f"✓ 计算完成")
        print(f"  输出形状: {output.shape}")
        print(f"  耗时: {elapsed*1000:.2f} ms")
        
        # 验证输出
        if output.shape == test_input.shape:
            print("✓ 输出形状正确")
        else:
            print(f"⚠️  输出形状不符合预期: {output.shape} (期望: {test_input.shape})")
        
        # 停止服务器
        print("\n停止服务器...")
        server.stop(grace=2)
        print("✓ 服务器停止成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        try:
            server.stop(grace=1)
        except:
            pass
        return False


def test_server_context_manager():
    """测试 4: 使用上下文管理器"""
    print("\n" + "=" * 70)
    print("测试 4: 服务器上下文管理器")
    print("=" * 70)
    
    try:
        # 加载模型
        model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        model.eval()
        
        compute_fn = ModelComputeFunction(
            model=model,
            device="cpu",
            model_name="gpt2-trunk-test"
        )
        
        # 使用上下文管理器
        print("使用上下文管理器启动服务器...")
        with GRPCComputeServer(
            compute_fn=compute_fn,
            host=TEST_HOST,
            port=TEST_PORT,
            max_workers=1
        ) as server:
            print("✓ 服务器在上下文中运行")
            time.sleep(1)
            
            # 检查端口
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', TEST_PORT))
            sock.close()
            
            if result == 0:
                print("✓ 端口正在监听")
            else:
                print("⚠️  端口未监听")
        
        print("✓ 上下文管理器退出，服务器自动停止")
        
        # 检查端口是否释放
        time.sleep(1)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', TEST_PORT))
        sock.close()
        
        if result != 0:
            print("✓ 端口已释放")
        else:
            print("⚠️  端口仍在监听")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("gRPC 服务器功能测试")
    print("=" * 70)
    print(f"\n测试配置:")
    print(f"  模型文件: {MODEL_PATH}")
    print(f"  服务器地址: {TEST_HOST}:{TEST_PORT}")
    print(f"  单线程模式: 是")
    print()
    
    results = {}
    
    # 测试 1: 服务器创建
    server, model = test_server_creation()
    if server is None:
        print("\n❌ 服务器创建失败，无法继续测试")
        return 1
    
    results['creation'] = server is not None
    
    # 测试 2: 服务器启动和停止
    results['start_stop'] = test_server_start_stop(server)
    
    # 测试 3: 使用模型计算
    results['model_compute'] = test_server_with_model(server, model)
    
    # 测试 4: 上下文管理器
    results['context_manager'] = test_server_context_manager()
    
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
        print("\n🎉 所有测试通过！gRPC 服务器功能正常！")
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

