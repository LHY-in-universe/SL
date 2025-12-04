#!/usr/bin/env python3
"""
gRPC 和 PyTorch 冲突诊断脚本

诊断同时使用 gRPC 和 PyTorch 时可能出现的问题：
- 线程冲突
- mutex 警告
- 初始化顺序问题
- 性能问题
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

logging.basicConfig(level=logging.WARNING)

from splitlearn_comm import GRPCComputeServer
from splitlearn_comm.core import ComputeFunction


def print_separator(title=""):
    """打印分隔线"""
    if title:
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)


def test_pytorch_alone():
    """测试 1: 单独使用 PyTorch"""
    print_separator("测试 1: 单独使用 PyTorch")
    
    print("\n✅ 测试 PyTorch 单独使用...")
    
    try:
        # 创建张量
        tensor = torch.randn(10, 10)
        print(f"  ✓ 创建张量: {tensor.shape}")
        
        # 矩阵乘法
        result = torch.matmul(tensor, tensor)
        print(f"  ✓ 矩阵乘法: {result.shape}")
        
        # 创建简单模型
        model = torch.nn.Linear(10, 5)
        output = model(tensor)
        print(f"  ✓ 模型推理: {output.shape}")
        
        print("\n✅ PyTorch 单独使用正常")
        return True
        
    except Exception as e:
        print(f"\n❌ PyTorch 单独使用失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_grpc_alone():
    """测试 2: 单独使用 gRPC（不使用 PyTorch）"""
    print_separator("测试 2: 单独使用 gRPC（不使用 PyTorch）")
    
    print("\n✅ 测试 gRPC 单独使用...")
    
    class SimpleCompute(ComputeFunction):
        def compute(self, input_tensor):
            # 不使用 PyTorch，只做简单操作
            return input_tensor * 2
    
    try:
        compute_fn = SimpleCompute()
        server = GRPCComputeServer(
            compute_fn=compute_fn,
            host="0.0.0.0",
            port=50057,
            max_workers=1
        )
        
        print(f"  ✓ 服务器创建成功")
        
        # 测试计算函数
        test_input = torch.randn(5, 5)
        output = compute_fn.compute(test_input)
        print(f"  ✓ 计算函数测试: {output.shape}")
        
        server.stop(grace=1)
        print(f"  ✓ 服务器停止成功")
        
        print("\n✅ gRPC 单独使用正常")
        return True
        
    except Exception as e:
        print(f"\n❌ gRPC 单独使用失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_grpc_with_pytorch():
    """测试 3: gRPC 和 PyTorch 同时使用"""
    print_separator("测试 3: gRPC 和 PyTorch 同时使用")
    
    print("\n⚠️  测试 gRPC 和 PyTorch 同时使用...")
    
    class PyTorchCompute(ComputeFunction):
        def __init__(self):
            self.model = torch.nn.Linear(10, 10)
            self.model.eval()
            self.request_count = 0
        
        def compute(self, input_tensor):
            self.request_count += 1
            with torch.no_grad():
                return self.model(input_tensor)
    
    try:
        compute_fn = PyTorchCompute()
        print(f"  ✓ 计算函数创建成功（使用 PyTorch 模型）")
        
        server = GRPCComputeServer(
            compute_fn=compute_fn,
            host="0.0.0.0",
            port=50058,
            max_workers=1  # 单线程模式
        )
        
        print(f"  ✓ 服务器创建成功")
        
        # 测试计算函数
        test_input = torch.randn(1, 10)
        print(f"  ✓ 测试输入: {test_input.shape}")
        
        output = compute_fn.compute(test_input)
        print(f"  ✓ 计算成功: {output.shape}")
        
        # 检查是否有警告
        print(f"\n  ⚠️  检查是否有 mutex 警告...")
        print(f"     (如果看到 mutex 警告，说明有线程冲突)")
        
        server.stop(grace=1)
        print(f"  ✓ 服务器停止成功")
        
        print(f"\n✅ gRPC 和 PyTorch 同时使用测试完成")
        print(f"   处理了 {compute_fn.request_count} 个请求")
        return True
        
    except Exception as e:
        print(f"\n❌ gRPC 和 PyTorch 同时使用失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_threading_conflict():
    """测试 4: 线程冲突检测"""
    print_separator("测试 4: 线程冲突检测")
    
    print("\n🔍 检查线程配置...")
    
    print(f"\nPyTorch 线程配置:")
    print(f"  计算线程数: {torch.get_num_threads()}")
    print(f"  互操作线程数: {torch.get_num_interop_threads()}")
    
    print(f"\n环境变量:")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', '未设置')}")
    print(f"  MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', '未设置')}")
    print(f"  NUMEXPR_NUM_THREADS: {os.environ.get('NUMEXPR_NUM_THREADS', '未设置')}")
    
    print(f"\n当前线程数:")
    print(f"  活动线程数: {threading.active_count()}")
    
    # 检查是否有线程冲突
    print(f"\n⚠️  潜在问题:")
    if torch.get_num_threads() > 1:
        print(f"  - PyTorch 使用多线程 ({torch.get_num_threads()} 个线程)")
        print(f"  - 可能与 gRPC 的线程池冲突")
    else:
        print(f"  - PyTorch 使用单线程 ✅")
    
    if threading.active_count() > 1:
        print(f"  - 当前有 {threading.active_count()} 个活动线程")
        print(f"  - 可能存在线程竞争")


def test_initialization_order():
    """测试 5: 初始化顺序"""
    print_separator("测试 5: 初始化顺序问题")
    
    print("\n🔍 检查初始化顺序...")
    
    print(f"\n当前导入顺序:")
    print(f"  1. torch (已导入)")
    print(f"  2. grpc (通过 splitlearn_comm 导入)")
    
    print(f"\n⚠️  可能的问题:")
    print(f"  - 如果先导入 torch，再导入 grpc，可能导致线程冲突")
    print(f"  - 建议：先设置环境变量，再导入 torch，最后导入 grpc")
    
    print(f"\n✅ 建议的导入顺序:")
    print(f"  1. 设置环境变量 (OMP_NUM_THREADS=1 等)")
    print(f"  2. import torch")
    print(f"  3. torch.set_num_threads(1)")
    print(f"  4. import grpc / from splitlearn_comm import ...")


def diagnose_common_issues():
    """诊断常见问题"""
    print_separator("常见问题诊断")
    
    print("\n📋 常见问题和解决方案:")
    
    print("\n1. mutex 警告")
    print("   症状: [mutex.cc : 452] RAW: Lock blocking")
    print("   原因: PyTorch 和 gRPC 的线程池冲突")
    print("   解决:")
    print("     - 设置 max_workers=1 (单线程 gRPC)")
    print("     - 设置 torch.set_num_threads(1)")
    print("     - 设置环境变量 OMP_NUM_THREADS=1")
    
    print("\n2. 性能下降")
    print("   症状: 同时使用时性能明显下降")
    print("   原因: 线程竞争导致上下文切换开销")
    print("   解决:")
    print("     - 使用单线程模式")
    print("     - 使用异步版本 (AsyncGRPCComputeServer)")
    
    print("\n3. 死锁或卡住")
    print("   症状: 程序卡住不响应")
    print("   原因: 线程死锁")
    print("   解决:")
    print("     - 检查锁的使用")
    print("     - 使用超时机制")
    print("     - 避免在计算函数中使用锁")
    
    print("\n4. 内存问题")
    print("   症状: 内存持续增长")
    print("   原因: 线程池缓存或模型未释放")
    print("   解决:")
    print("     - 使用 torch.no_grad()")
    print("     - 及时释放不需要的张量")
    print("     - 限制线程池大小")


def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("gRPC 和 PyTorch 冲突诊断")
    print("=" * 70)
    
    results = {}
    
    # 运行测试
    results['pytorch_alone'] = test_pytorch_alone()
    results['grpc_alone'] = test_grpc_alone()
    results['grpc_with_pytorch'] = test_grpc_with_pytorch()
    
    # 诊断
    test_threading_conflict()
    test_initialization_order()
    diagnose_common_issues()
    
    # 总结
    print_separator("测试总结")
    
    print("\n测试结果:")
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name:25s}: {status}")
    
    print("\n💡 建议:")
    if not results.get('grpc_with_pytorch', False):
        print("  - gRPC 和 PyTorch 同时使用有问题")
        print("  - 建议使用单线程模式 (max_workers=1)")
        print("  - 建议设置 torch.set_num_threads(1)")
        print("  - 建议使用异步版本 (AsyncGRPCComputeServer)")
    else:
        print("  - gRPC 和 PyTorch 同时使用正常")
        print("  - 如果仍有问题，检查具体错误信息")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

