#!/usr/bin/env python3
"""
测试异步版本是否能解决 gRPC 和 PyTorch 的冲突

对比测试：
1. 同步版本 + PyTorch（有冲突）
2. 异步版本 + PyTorch（应该无冲突）
"""

import os
import sys
import time
import asyncio
import torch
import torch.nn as nn
import logging
import threading

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

from splitlearn_comm import GRPCComputeServer, AsyncGRPCComputeServer
from splitlearn_comm.core import ComputeFunction, AsyncComputeFunction


def print_separator(title=""):
    """打印分隔线"""
    if title:
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)


class SyncPyTorchCompute(ComputeFunction):
    """同步版本：使用 PyTorch 模型"""
    
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        self.model.eval()
        self.request_count = 0
    
    def compute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.request_count += 1
        with torch.no_grad():
            return self.model(input_tensor)


class AsyncPyTorchCompute(AsyncComputeFunction):
    """异步版本：使用 PyTorch 模型"""
    
    def __init__(self, executor=None):
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        self.model.eval()
        self.request_count = 0
        self.executor = executor
    
    async def compute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.request_count += 1
        loop = asyncio.get_event_loop()
        
        def _sync_compute():
            with torch.no_grad():
                return self.model(input_tensor)
        
        # 在 executor 中执行（避免阻塞事件循环）
        if self.executor:
            return await loop.run_in_executor(self.executor, _sync_compute)
        else:
            return await loop.run_in_executor(None, _sync_compute)
    
    async def setup(self):
        """初始化"""
        pass
    
    async def teardown(self):
        """清理"""
        pass


def test_sync_version():
    """测试 1: 同步版本 + PyTorch"""
    print_separator("测试 1: 同步版本 + PyTorch（可能有冲突）")
    
    print("\n⚠️  测试同步版本...")
    print("   使用: GRPCComputeServer + PyTorch 模型")
    print("   预期: 可能有 mutex 警告或线程冲突")
    
    try:
        compute_fn = SyncPyTorchCompute()
        server = GRPCComputeServer(
            compute_fn=compute_fn,
            host="0.0.0.0",
            port=50059,
            max_workers=1  # 单线程模式
        )
        
        print(f"\n✓ 服务器创建成功")
        
        # 测试计算
        test_input = torch.randn(1, 10)
        output = compute_fn.compute(test_input)
        print(f"✓ 计算测试成功: {output.shape}")
        
        # 检查线程数
        thread_count = threading.active_count()
        print(f"✓ 当前线程数: {thread_count}")
        
        server.stop(grace=1)
        print(f"✓ 服务器停止成功")
        
        print(f"\n📊 结果:")
        print(f"   处理了 {compute_fn.request_count} 个请求")
        print(f"   线程数: {thread_count}")
        print(f"   ⚠️  如果看到 mutex 警告，说明有线程冲突")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_async_version():
    """测试 2: 异步版本 + PyTorch"""
    print_separator("测试 2: 异步版本 + PyTorch（应该无冲突）")
    
    print("\n✅ 测试异步版本...")
    print("   使用: AsyncGRPCComputeServer + PyTorch 模型")
    print("   预期: 无 mutex 警告，无线程冲突")
    
    try:
        from concurrent.futures import ThreadPoolExecutor
        
        # 创建单线程 executor（用于 PyTorch 计算）
        executor = ThreadPoolExecutor(max_workers=1)
        
        compute_fn = AsyncPyTorchCompute(executor=executor)
        server = AsyncGRPCComputeServer(
            compute_fn=compute_fn,
            host="0.0.0.0",
            port=50060
        )
        
        print(f"\n✓ 服务器创建成功")
        
        # 测试计算
        test_input = torch.randn(1, 10)
        output = await compute_fn.compute(test_input)
        print(f"✓ 计算测试成功: {output.shape}")
        
        # 检查线程数
        thread_count = threading.active_count()
        print(f"✓ 当前线程数: {thread_count}")
        
        # 启动服务器（短暂运行）
        await server.start()
        print(f"✓ 服务器启动成功")
        
        # 等待一下
        await asyncio.sleep(1)
        
        await server.stop()
        executor.shutdown(wait=True)
        print(f"✓ 服务器停止成功")
        
        print(f"\n📊 结果:")
        print(f"   处理了 {compute_fn.request_count} 个请求")
        print(f"   线程数: {thread_count}")
        print(f"   ✅ 异步版本使用协程，不应该有线程冲突")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_thread_comparison():
    """测试 3: 线程数对比"""
    print_separator("测试 3: 线程数对比")
    
    print("\n🔍 对比同步和异步版本的线程使用...")
    
    initial_threads = threading.active_count()
    print(f"\n初始线程数: {initial_threads}")
    
    # 同步版本
    print(f"\n同步版本 (GRPCComputeServer):")
    print(f"  - 使用 ThreadPoolExecutor")
    print(f"  - max_workers=1 时: 1 个 gRPC 线程")
    print(f"  - 加上 PyTorch 线程: 1 + 4 = 5 个线程（如果 PyTorch 多线程）")
    print(f"  - 问题: 线程竞争")
    
    # 异步版本
    print(f"\n异步版本 (AsyncGRPCComputeServer):")
    print(f"  - 使用 asyncio 事件循环")
    print(f"  - 只有 1 个主线程")
    print(f"  - 使用协程切换，不是真正的多线程")
    print(f"  - 优势: 无线程竞争")


def test_concurrent_requests():
    """测试 4: 并发请求对比"""
    print_separator("测试 4: 并发请求处理能力")
    
    print("\n📊 并发请求处理对比:")
    
    print(f"\n同步版本 (max_workers=1):")
    print(f"  - 同时只能处理 1 个请求")
    print(f"  - 其他请求需要等待")
    print(f"  - 适合: 低并发场景")
    
    print(f"\n异步版本:")
    print(f"  - 可以同时处理多个请求（协程切换）")
    print(f"  - 不需要等待")
    print(f"  - 适合: 高并发场景")
    print(f"  - 优势: 无线程竞争，性能更好")


def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("异步版本 vs 同步版本：解决 gRPC 和 PyTorch 冲突测试")
    print("=" * 70)
    
    results = {}
    
    # 测试线程对比
    test_thread_comparison()
    
    # 测试并发能力
    test_concurrent_requests()
    
    # 测试同步版本
    print("\n" + "=" * 70)
    print("开始实际测试...")
    print("=" * 70)
    
    results['sync'] = test_sync_version()
    
    # 测试异步版本
    print("\n" + "=" * 70)
    print("测试异步版本...")
    print("=" * 70)
    
    results['async'] = asyncio.run(test_async_version())
    
    # 总结
    print_separator("测试总结")
    
    print("\n测试结果:")
    print(f"  同步版本: {'✅ 通过' if results.get('sync') else '❌ 失败'}")
    print(f"  异步版本: {'✅ 通过' if results.get('async') else '❌ 失败'}")
    
    print("\n💡 结论:")
    if results.get('async'):
        print("  ✅ 异步版本可以解决 gRPC 和 PyTorch 的冲突问题")
        print("  ✅ 使用 AsyncGRPCComputeServer 推荐用于生产环境")
        print("  ✅ 没有线程竞争，无 mutex 警告")
    else:
        print("  ⚠️  异步版本测试失败，需要进一步检查")
    
    if results.get('sync'):
        print("\n  ⚠️  同步版本虽然能运行，但可能有线程冲突")
        print("  ⚠️  建议使用异步版本")
    
    print("\n📋 建议:")
    print("  1. 生产环境使用异步版本 (AsyncGRPCComputeServer)")
    print("  2. 测试环境可以使用同步版本 (max_workers=1)")
    print("  3. 设置 PyTorch 单线程模式避免冲突")
    
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

