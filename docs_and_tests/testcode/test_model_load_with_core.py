#!/usr/bin/env python3
"""
使用 SplitLearnCore 库加载模型文件

测试从硬盘加载模型到内存的功能，使用 core 库的模型类
"""

import os
import sys
import time
import torch
from pathlib import Path

# 设置环境变量（在导入 torch 之前）
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearnCore', 'src'))

# 导入 core 库
from splitlearn_core.models.gpt2 import GPT2TrunkModel, GPT2BottomModel, GPT2TopModel
from transformers import GPT2Config

# 测试配置
MODEL_FILES = {
    "gpt2_trunk_full.pt": {
        "type": "trunk",
        "model_class": GPT2TrunkModel,
        "split_config": {"start_layer": 2, "end_layer": 10}
    },
    "gpt2_bottom_cached.pt": {
        "type": "bottom",
        "model_class": GPT2BottomModel,
        "split_config": {"end_layer": 2}
    },
    "gpt2_top_cached.pt": {
        "type": "top",
        "model_class": GPT2TopModel,
        "split_config": {"start_layer": 10}
    },
}


def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def get_model_info(model):
    """获取模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)  # 假设 float32
    
    return {
        "type": type(model).__name__,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size_mb,
    }


def test_load_with_torch_load(model_path):
    """方法 1: 直接使用 torch.load() 加载整个模型对象"""
    print("\n" + "=" * 70)
    print(f"📦 方法 1: 使用 torch.load() 加载模型")
    print("=" * 70)
    print(f"文件: {os.path.basename(model_path)}")
    
    if not os.path.exists(model_path):
        print(f"❌ 文件不存在: {model_path}")
        return None
    
    file_size = os.path.getsize(model_path)
    print(f"\n📁 文件信息:")
    print(f"   大小: {format_size(file_size)}")
    
    print(f"\n⏳ 开始加载...")
    start_time = time.time()
    
    try:
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        load_time = time.time() - start_time
        
        print(f"   ✓ 加载成功！耗时: {load_time:.2f} 秒")
        
        model_info = get_model_info(model)
        print(f"\n📊 模型信息:")
        print(f"   类型: {model_info['type']}")
        print(f"   参数量: {model_info['total_params']:,}")
        print(f"   模型大小: {model_info['model_size_mb']:.2f} MB")
        
        return model
        
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_load_with_core_class(model_path, model_info_config):
    """方法 2: 使用 core 库的模型类加载"""
    print("\n" + "=" * 70)
    print(f"📦 方法 2: 使用 SplitLearnCore 模型类加载")
    print("=" * 70)
    print(f"文件: {os.path.basename(model_path)}")
    print(f"模型类型: {model_info_config['type']}")
    
    if not os.path.exists(model_path):
        print(f"❌ 文件不存在: {model_path}")
        return None
    
    file_size = os.path.getsize(model_path)
    print(f"\n📁 文件信息:")
    print(f"   大小: {format_size(file_size)}")
    
    print(f"\n⏳ 开始加载...")
    start_time = time.time()
    
    try:
        # 方法 2a: 如果文件保存的是整个模型对象，直接加载
        loaded_data = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 检查加载的是什么
        if isinstance(loaded_data, torch.nn.Module):
            print(f"   ✓ 文件包含完整模型对象")
            model = loaded_data
            load_time = time.time() - start_time
            print(f"   ✓ 加载成功！耗时: {load_time:.2f} 秒")
            
        elif isinstance(loaded_data, dict):
            print(f"   ✓ 文件包含 state_dict")
            print(f"   正在使用 core 库创建模型实例...")
            
            # 使用 core 库创建模型实例
            model_class = model_info_config['model_class']
            split_config = model_info_config['split_config']
            
            # 创建配置
            config = GPT2Config()
            
            # 根据模型类型创建实例
            if model_info_config['type'] == 'trunk':
                model = model_class(
                    config=config,
                    start_layer=split_config['start_layer'],
                    end_layer=split_config['end_layer']
                )
            elif model_info_config['type'] == 'bottom':
                model = model_class(
                    config=config,
                    end_layer=split_config['end_layer']
                )
            elif model_info_config['type'] == 'top':
                model = model_class(
                    config=config,
                    start_layer=split_config['start_layer']
                )
            
            # 加载 state_dict
            print(f"   正在加载 state_dict...")
            model.load_state_dict(loaded_data, strict=False)
            
            load_time = time.time() - start_time
            print(f"   ✓ 加载成功！耗时: {load_time:.2f} 秒")
        else:
            print(f"   ⚠️  未知的数据类型: {type(loaded_data)}")
            return None
        
        model.eval()
        print(f"   ✓ 模型设置为评估模式")
        
        model_info = get_model_info(model)
        print(f"\n📊 模型信息:")
        print(f"   类型: {model_info['type']}")
        print(f"   参数量: {model_info['total_params']:,}")
        print(f"   模型大小: {model_info['model_size_mb']:.2f} MB")
        
        # 测试推理
        print(f"\n🧪 测试模型推理...")
        try:
            if model_info_config['type'] == 'trunk':
                test_input = torch.randn(1, 5, 768)  # [batch, seq_len, hidden_dim]
            elif model_info_config['type'] == 'bottom':
                test_input = torch.randint(0, 50257, (1, 5))  # [batch, seq_len] token ids
            else:  # top
                test_input = torch.randn(1, 5, 768)  # [batch, seq_len, hidden_dim]
            
            with torch.no_grad():
                output = model(test_input)
            
            print(f"   ✓ 推理测试成功")
            print(f"   输入形状: {test_input.shape}")
            print(f"   输出形状: {output.shape}")
            
        except Exception as e:
            print(f"   ⚠️  推理测试失败: {e}")
            print(f"   （这可能是因为输入格式不正确）")
        
        return model
        
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("🧪 使用 SplitLearnCore 库加载模型文件测试")
    print("=" * 70)
    print("\n📋 测试目标:")
    print("   1. 使用 torch.load() 直接加载模型")
    print("   2. 使用 SplitLearnCore 模型类加载模型")
    print("   3. 显示模型信息和性能统计")
    print("   4. 测试模型推理功能")
    
    # 获取模型文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 只测试存在的文件
    existing_files = {}
    for filename, config in MODEL_FILES.items():
        filepath = os.path.join(current_dir, filename)
        if os.path.exists(filepath):
            existing_files[filepath] = config
    
    if not existing_files:
        print(f"\n❌ 没有找到任何模型文件")
        print(f"   查找路径: {current_dir}")
        return 1
    
    print(f"\n📁 找到 {len(existing_files)} 个模型文件:")
    for filepath in existing_files.keys():
        print(f"   - {os.path.basename(filepath)}")
    
    # 测试每个模型
    results = []
    for model_path, config in existing_files.items():
        print("\n" + "=" * 70)
        print(f"测试文件: {os.path.basename(model_path)}")
        print("=" * 70)
        
        # 方法 1: 直接使用 torch.load()
        model1 = test_load_with_torch_load(model_path)
        
        # 方法 2: 使用 core 库
        model2 = test_load_with_core_class(model_path, config)
        
        success = model1 is not None or model2 is not None
        results.append((os.path.basename(model_path), success))
        
        time.sleep(1)  # 短暂休息
    
    # 总结
    print("\n" + "=" * 70)
    print("📊 测试总结")
    print("=" * 70)
    
    for model_name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"   {status}: {model_name}")
    
    success_count = sum(1 for _, s in results if s)
    print(f"\n总计: {success_count}/{len(results)} 个模型加载成功")
    
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

