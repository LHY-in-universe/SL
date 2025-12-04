#!/usr/bin/env python3
"""
异步 gRPC 服务器 - 使用 testcode 中的模型文件

使用 Comm 和 Core 库：
- 使用 AsyncGRPCComputeServer（异步服务器）
- 使用 AsyncModelComputeFunction（异步计算函数）
- 加载 testcode 中的 .pt 模型文件
"""

import os
import sys
import asyncio
import time
import torch
import logging
from concurrent.futures import ThreadPoolExecutor

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

# 配置详细日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

from splitlearn_comm import AsyncGRPCComputeServer
from splitlearn_comm.core import AsyncModelComputeFunction

# 测试配置
PORT = 50061
HOST = "0.0.0.0"
MODEL_PATH = os.path.join(current_dir, "gpt2_trunk_full.pt")


async def async_main():
    """异步主函数"""
    print("\n" + "=" * 70)
    print("🚀 异步 gRPC 服务器启动（使用模型文件）")
    print("=" * 70)
    
    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ 模型文件不存在: {MODEL_PATH}")
        print("   请先运行: python testcode/prepare_models.py")
        return 1
    
    # 加载模型
    print(f"\n📦 加载模型: {MODEL_PATH}")
    try:
        from splitlearn_core.models.gpt2 import GPT2TrunkModel
        print("   ✓ GPT2TrunkModel 导入成功")
    except ImportError as e:
        print(f"   ⚠️  无法导入 GPT2TrunkModel: {e}")
        print("   尝试直接加载...")
    
    print("   正在加载模型...")
    print("   （这可能需要一些时间，请耐心等待...）")
    start_time = time.time()
    
    try:
        # 在 executor 中异步加载模型，避免阻塞事件循环
        def _load_model():
            """在后台线程中加载模型"""
            print("   ⏳ 开始 torch.load()（在后台线程中）...")
            import sys
            sys.stdout.flush()
            model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
            model.eval()
            return model
        
        # 使用临时 executor 加载模型
        temp_executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(temp_executor, _load_model)
        temp_executor.shutdown(wait=False)
        
        print("   ✓ 模型文件加载成功")
        print("   ✓ 模型设置为评估模式")
        
        load_time = time.time() - start_time
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ 模型加载完成 (耗时: {load_time:.2f} 秒)")
        print(f"   ✓ 参数量: {total_params:,}")
        print(f"   ✓ 模型类型: {type(model).__name__}")
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 创建单线程 executor（用于 PyTorch 推理）
    executor = ThreadPoolExecutor(max_workers=1)
    print(f"\n🔧 创建异步计算函数...")
    print(f"   使用单线程 executor（避免线程竞争）")
    
    # 创建异步计算函数
    compute_fn = AsyncModelComputeFunction(
        model=model,
        device="cpu",
        executor=executor
    )
    print("   ✓ 异步计算函数创建成功")
    
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
    print(f"💡 在另一个终端运行客户端: python testcode/client_async_with_model.py")
    print(f"⏹️  按 Ctrl+C 停止服务器\n")
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n\n🛑 收到停止信号，正在关闭服务器...")
        await server.stop()
        executor.shutdown(wait=True)
        print("   ✓ 服务器已关闭")
    
    return 0


def main():
    """主函数"""
    try:
        return asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n\n🛑 服务器被用户中断")
        return 0
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

