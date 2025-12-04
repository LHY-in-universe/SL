#!/usr/bin/env python3
"""
gRPC 服务器测试脚本 - 异步版本（不使用模型）

只测试通信功能，使用简单的数学运算代替模型
- 使用异步版本（AsyncGRPCComputeServer）
- 不需要加载模型
- 不需要 PyTorch 模型
- 只测试数据传输
- 无线程竞争问题
"""

import os
import sys
import asyncio
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

from splitlearn_comm import AsyncGRPCComputeServer
from splitlearn_comm.core import AsyncComputeFunction

# 测试配置
PORT = 50056
HOST = "0.0.0.0"


class SimpleAsyncComputeFunction(AsyncComputeFunction):
    """
    简单的异步计算函数 - 不使用模型
    
    只做简单的数学运算来测试通信功能
    使用异步版本，避免线程竞争
    """
    
    def __init__(self):
        self.request_count = 0
        print("✓ 简单异步计算函数创建（不使用模型）")
    
    async def compute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """执行简单计算：输入 * 2 + 1"""
        self.request_count += 1
        req_id = self.request_count
        
        print("\n" + "=" * 70)
        print(f"📥 服务器收到请求 #{req_id}")
        print("=" * 70)
        
        # 显示输入数据信息
        print(f"\n📊 输入数据信息:")
        print(f"   形状: {input_tensor.shape}")
        print(f"   数据类型: {input_tensor.dtype}")
        print(f"   数据大小: {input_tensor.numel() * 4 / 1024:.2f} KB")
        
        # 显示输入数据的统计信息
        print(f"\n📈 输入数据统计:")
        print(f"   最小值: {input_tensor.min().item():.6f}")
        print(f"   最大值: {input_tensor.max().item():.6f}")
        print(f"   平均值: {input_tensor.mean().item():.6f}")
        
        # 执行简单计算（模拟模型推理）
        print(f"\n⚙️  执行计算: output = input * 2 + 1")
        start_time = time.time()
        
        with torch.no_grad():
            output = input_tensor * 2 + 1
        
        compute_time = (time.time() - start_time) * 1000
        
        # 显示输出数据信息
        print(f"\n📤 输出数据信息:")
        print(f"   形状: {output.shape}")
        print(f"   数据类型: {output.dtype}")
        print(f"   数据大小: {output.numel() * 4 / 1024:.2f} KB")
        print(f"   计算耗时: {compute_time:.2f} ms")
        
        # 显示输出数据的统计信息
        print(f"\n📈 输出数据统计:")
        print(f"   最小值: {output.min().item():.6f}")
        print(f"   最大值: {output.max().item():.6f}")
        print(f"   平均值: {output.mean().item():.6f}")
        
        # 数据传输信息
        input_size_kb = input_tensor.numel() * 4 / 1024
        output_size_kb = output.numel() * 4 / 1024
        total_size_kb = input_size_kb + output_size_kb
        
        print(f"\n📡 数据传输统计:")
        print(f"   接收数据: {input_size_kb:.2f} KB")
        print(f"   发送数据: {output_size_kb:.2f} KB")
        print(f"   总传输: {total_size_kb:.2f} KB")
        print(f"   总耗时: {compute_time:.2f} ms")
        if compute_time > 0:
            print(f"   吞吐量: {total_size_kb / (compute_time / 1000):.2f} KB/s")
        
        print("=" * 70)
        
        return output
    
    def get_info(self):
        return {
            "name": "SimpleAsyncComputeFunction",
            "description": "简单异步计算函数（不使用模型）",
            "operation": "output = input * 2 + 1",
            "total_requests": self.request_count
        }
    
    async def setup(self):
        """初始化"""
        pass
    
    async def teardown(self):
        """清理"""
        pass


async def async_main():
    """异步主函数"""
    print("\n" + "=" * 70)
    print("🚀 gRPC 服务器启动（异步版本 - 不使用模型）")
    print("=" * 70)
    
    # 创建计算函数（不使用模型）
    print(f"\n🔧 创建异步计算函数...")
    compute_fn = SimpleAsyncComputeFunction()
    
    # 创建异步服务器
    print(f"\n🌐 创建异步 gRPC 服务器...")
    print(f"   监听地址: {HOST}:{PORT}")
    print(f"   使用协程（不是线程池）")
    print(f"   ✅ 无线程竞争问题")
    
    server = AsyncGRPCComputeServer(
        compute_fn=compute_fn,
        host=HOST,
        port=PORT
    )
    print("   ✓ 异步服务器创建成功")
    
    # 启动服务器
    print(f"\n▶️  启动服务器...")
    await server.start()
    print("   ✓ 服务器已启动")
    
    print("\n" + "=" * 70)
    print("✅ 服务器运行中，等待客户端连接...")
    print("=" * 70)
    print(f"\n📡 服务器地址: localhost:{PORT}")
    print(f"💡 在另一个终端运行客户端: python testcode/client_comm_simple.py")
    print(f"💡 或者使用异步客户端: python testcode/client_comm_simple_async.py")
    print(f"⏹️  按 Ctrl+C 停止服务器\n")
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n\n🛑 收到停止信号，正在关闭服务器...")
        await server.stop()
        print("   ✓ 服务器已关闭")
        print(f"\n📊 总共处理了 {compute_fn.request_count} 个请求")


def main():
    """主函数"""
    try:
        asyncio.run(async_main())
        return 0
    except KeyboardInterrupt:
        print("\n\n🛑 服务器被用户中断")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

