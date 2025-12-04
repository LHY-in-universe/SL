#!/usr/bin/env python3
"""
直接测试模型加载功能
使用 AsyncModelManager 来诊断问题
"""

import asyncio
import logging
import sys
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from splitlearn_manager.core.async_model_manager import AsyncModelManager
from splitlearn_manager.config import ModelConfig

async def test_model_loading():
    """测试模型加载"""
    print("=" * 70)
    print("测试模型加载功能")
    print("=" * 70)
    
    # 创建模型管理器
    print("\n[1] 创建 AsyncModelManager...")
    manager = AsyncModelManager(max_models=5)
    print("✓ 管理器创建成功")
    
    # 创建模型配置
    print("\n[2] 创建模型配置...")
    model_config = ModelConfig(
        model_id="gpt2_trunk_test",
        model_path="gpt2",  # 使用 HuggingFace 模型名称
        model_type="gpt2",  # 使用 gpt2 类型，会触发 SplitLearnCore 加载
        device="cpu",
        config={
            "component": "trunk",
            "split_points": [2, 10],
            "cache_dir": "./models"
        }
    )
    print(f"✓ 配置创建成功: {model_config.model_id}")
    print(f"  - model_type: {model_config.model_type}")
    print(f"  - component: {model_config.config.get('component')}")
    print(f"  - split_points: {model_config.config.get('split_points')}")
    print(f"  - cache_dir: {model_config.config.get('cache_dir')}")
    
    # 检查缓存目录
    cache_dir = Path(model_config.config.get("cache_dir", "./models"))
    print(f"\n[3] 检查缓存目录: {cache_dir}")
    if cache_dir.exists():
        print(f"✓ 目录存在，内容:")
        for item in cache_dir.iterdir():
            print(f"  - {item.name} ({item.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        print(f"⚠ 目录不存在，将创建")
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 尝试加载模型
    print("\n[4] 开始加载模型...")
    print("   这可能需要几分钟（首次需要下载模型）...")
    try:
        result = await manager.load_model(model_config)
        print(f"✓ 模型加载成功: {result}")
        
        # 列出模型
        print("\n[5] 列出已加载的模型...")
        models = await manager.list_models()
        print(f"✓ 找到 {len(models)} 个模型:")
        for model in models:
            print(f"  - {model.get('model_id')}: {model.get('status')}")
        
        # 获取模型信息
        print("\n[6] 获取模型详细信息...")
        info = await manager.get_model_info("gpt2_trunk_test")
        print(f"✓ 模型信息:")
        for key, value in info.items():
            print(f"  - {key}: {value}")
        
        # 检查模型文件
        print("\n[7] 检查模型文件...")
        if cache_dir.exists():
            print(f"✓ 缓存目录内容:")
            for item in cache_dir.iterdir():
                size_mb = item.stat().st_size / 1024 / 1024
                print(f"  - {item.name} ({size_mb:.2f} MB)")
        
        print("\n" + "=" * 70)
        print("✓ 测试完成！模型加载成功！")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理
        print("\n[8] 清理资源...")
        await manager.shutdown()
        print("✓ 清理完成")
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(test_model_loading())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

