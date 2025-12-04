#!/usr/bin/env python3
"""
测试模型加载部分（不启动服务器）

这个脚本专门测试模型加载过程，检查是否有 mutex 警告或其他问题。
"""

import os
import sys
import warnings
import time

# 在导入任何模块之前设置环境变量
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

# 抑制 Python 警告（但保留 mutex 警告以便观察）
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*coroutine.*')
# 注意：不抑制 mutex 警告，这样我们可以看到它们

print("=" * 70)
print("模型加载测试（仅测试加载，不启动服务器）")
print("=" * 70)
print()

print("环境变量设置:")
print(f"  OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS')}")
print(f"  MKL_NUM_THREADS = {os.environ.get('MKL_NUM_THREADS')}")
print(f"  NUMEXPR_NUM_THREADS = {os.environ.get('NUMEXPR_NUM_THREADS')}")
print()

# 导入必要的模块
print("导入模块...")
from splitlearn_manager.quickstart import ManagedServer
from splitlearn_manager.config import ModelConfig, ServerConfig
import torch

print("✓ 模块导入完成")
print()

# 配置 PyTorch 线程数
print("配置 PyTorch 线程数...")
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(f"✓ PyTorch 线程数: {torch.get_num_threads()}")
    print(f"✓ PyTorch 线程间操作线程数: {torch.get_num_interop_threads()}")
except RuntimeError as e:
    print(f"⚠️  无法设置线程数: {e}")
print()

# 创建服务器配置（但不启动 gRPC）
print("创建服务器配置...")
server_config = ServerConfig(
    host="0.0.0.0",
    port=50051,
    max_workers=1,
    max_models=1
)
print("✓ 服务器配置创建完成")
print()

# 创建 ManagedServer（但不启动）
print("创建 ManagedServer...")
server = ManagedServer(
    model_type="gpt2",
    component="trunk",
    port=50051,
    max_workers=1
)
print("✓ ManagedServer 创建完成")
print()

# 只测试模型加载部分
print("=" * 70)
print("开始测试模型加载...")
print("=" * 70)
print()

print("注意：这将只加载模型，不启动 gRPC 服务器")
print("如果看到 mutex 警告，它们应该出现在模型加载过程中")
print()

start_time = time.time()

try:
    # 手动调用模型加载（模拟 _async_start 中的加载部分）
    import asyncio
    
    async def test_model_loading():
        """测试模型加载"""
        # 创建模型配置
        model_config_dict = {
            "component": "trunk",
            "split_points": [2, 10],
            "cache_dir": "./models",
        }
        model_config = ModelConfig(
            model_id="gpt2_trunk",
            model_path="gpt2",
            model_type="gpt2",
            device="cpu",
            config=model_config_dict
        )
        
        print(f"模型配置:")
        print(f"  Model ID: {model_config.model_id}")
        print(f"  Model Type: {model_config.model_type}")
        print(f"  Component: {model_config.config.get('component')}")
        print(f"  Device: {model_config.device}")
        print()
        
        # 创建 AsyncManagedServer（但不启动 gRPC）
        from splitlearn_manager.server import AsyncManagedServer
        async_server = AsyncManagedServer(config=server_config)
        
        print("开始加载模型...")
        print("(这可能需要一些时间，请耐心等待)")
        print()
        
        # 加载模型
        success = await async_server.load_model(model_config)
        
        if success:
            print("✓ 模型加载成功！")
            print()
            
            # 检查模型信息
            models = await async_server.list_models()
            if models:
                print("已加载的模型:")
                for model_info in models:
                    print(f"  - {model_info.get('model_id')}: {model_info.get('status')}")
            print()
            
            # 测试模型推理
            print("测试模型推理...")
            import torch
            test_input = torch.randn(1, 10, 768)
            print(f"输入形状: {test_input.shape}")
            
            # 获取计算函数
            compute_fn = async_server.compute_fn
            output = await compute_fn.compute(test_input)
            print(f"输出形状: {output.shape}")
            print("✓ 模型推理测试成功！")
            print()
        else:
            print("✗ 模型加载失败")
        
        return success
    
    # 运行异步测试
    success = asyncio.run(test_model_loading())
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("=" * 70)
    print("测试结果")
    print("=" * 70)
    print(f"模型加载耗时: {elapsed:.2f} 秒")
    if success:
        print("✓ 模型加载测试通过")
    else:
        print("✗ 模型加载测试失败")
    print()
    
except Exception as e:
    print()
    print("=" * 70)
    print("测试失败")
    print("=" * 70)
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 70)
print("测试完成")
print("=" * 70)
print()
print("提示：如果看到 mutex 警告，它们应该出现在模型加载过程中。")
print("如果只有 1-2 个警告，这是正常的（PyTorch 初始化）。")
print("如果有很多警告，可能需要进一步优化。")

