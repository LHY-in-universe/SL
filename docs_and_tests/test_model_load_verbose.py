#!/usr/bin/env python3
"""
详细测试模型加载，显示所有输出和进度
"""

import os
import sys
import asyncio
import logging
import time
from pathlib import Path

# 在导入之前设置环境变量
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

# 配置详细日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("=" * 70)
print("模型加载详细测试")
print("=" * 70)
print()

from splitlearn_manager.core.async_model_manager import AsyncModelManager
from splitlearn_manager.config import ModelConfig

async def test_model_loading_verbose():
    """详细测试模型加载"""
    
    print("[1] 创建 AsyncModelManager...")
    manager = AsyncModelManager(max_models=5, max_workers=1)
    print("✓ 管理器创建成功\n")
    
    print("[2] 创建模型配置...")
    model_config = ModelConfig(
        model_id="gpt2_trunk_test",
        model_path="gpt2",
        model_type="gpt2",
        device="cpu",
        config={
            "component": "trunk",
            "split_points": [2, 10],
            "cache_dir": "./models"
        }
    )
    print(f"✓ 配置: {model_config.model_id}")
    print(f"  - 类型: {model_config.model_type}")
    print(f"  - 组件: {model_config.config.get('component')}")
    print(f"  - 设备: {model_config.device}\n")
    
    print("[3] 检查缓存目录...")
    cache_dir = Path("./models")
    if cache_dir.exists():
        items = list(cache_dir.iterdir())
        if items:
            print(f"✓ 目录存在，包含 {len(items)} 个项目:")
            for item in items:
                size_mb = item.stat().st_size / 1024 / 1024
                print(f"  - {item.name} ({size_mb:.2f} MB)")
        else:
            print("✓ 目录存在但为空")
    else:
        print("⚠ 目录不存在，将创建")
        cache_dir.mkdir(parents=True, exist_ok=True)
    print()
    
    print("[4] 开始加载模型...")
    print("    注意：首次加载可能需要几分钟（下载和处理模型）")
    print("    请耐心等待...\n")
    
    start_time = time.time()
    
    try:
        # 加载模型（带超时）
        print(">>> 调用 load_model()...")
        result = await asyncio.wait_for(
            manager.load_model(model_config),
            timeout=300.0  # 5 分钟超时
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ 模型加载成功！耗时: {elapsed:.2f} 秒\n")
        
        # 列出模型
        print("[5] 列出已加载的模型...")
        models = await manager.list_models()
        print(f"✓ 找到 {len(models)} 个模型:")
        for model in models:
            print(f"  - ID: {model.get('model_id')}")
            print(f"    状态: {model.get('status')}")
            print(f"    设备: {model.get('device')}")
            print(f"    请求数: {model.get('request_count', 0)}")
            print()
        
        # 获取模型详细信息
        if models:
            print("[6] 获取模型详细信息...")
            try:
                managed_model = await manager.get_model(models[0].get('model_id'))
                if managed_model:
                    print("✓ 模型对象:")
                    print(f"  - 类型: {type(managed_model.model)}")
                    print(f"  - 参数数量: {sum(p.numel() for p in managed_model.model.parameters()):,}")
                    print()
            except Exception as e:
                print(f"⚠ 无法获取模型对象: {e}\n")
        
        # 检查缓存目录
        print("[7] 检查模型文件...")
        if cache_dir.exists():
            items = list(cache_dir.iterdir())
            if items:
                print(f"✓ 缓存目录现在包含 {len(items)} 个项目:")
                total_size = 0
                for item in items:
                    size_mb = item.stat().st_size / 1024 / 1024
                    total_size += size_mb
                    print(f"  - {item.name} ({size_mb:.2f} MB)")
                print(f"  总大小: {total_size:.2f} MB")
            else:
                print("⚠ 缓存目录仍然为空（模型可能缓存在 HuggingFace 默认位置）")
        print()
        
        print("=" * 70)
        print("✓ 测试完成！模型加载成功！")
        print("=" * 70)
        return True
        
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        print(f"\n✗ 模型加载超时（超过 300 秒）")
        print(f"  已等待: {elapsed:.2f} 秒")
        print("  可能原因：")
        print("  - 首次下载模型需要更长时间")
        print("  - 网络连接较慢")
        print("  - 模型处理时间较长")
        return False
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ 模型加载失败（耗时 {elapsed:.2f} 秒）")
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print("\n[8] 清理资源...")
        await manager.shutdown()
        print("✓ 清理完成")

if __name__ == "__main__":
    try:
        success = asyncio.run(test_model_loading_verbose())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

