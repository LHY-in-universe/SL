#!/usr/bin/env python3
"""
测试模型从硬盘加载到内存的功能

只测试模型加载，不涉及 gRPC 服务器或客户端
"""

import os
import sys
import time
import torch

# 设置环境变量（在导入 torch 之前）
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

# 添加路径（如果需要导入 splitlearn_core）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearnCore', 'src'))

# 测试配置
MODEL_FILES = [
    "gpt2_trunk_full.pt",
    "gpt2_bottom_cached.pt",
    "gpt2_top_cached.pt",
]


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
    
    # 计算模型大小（参数数量 * 4 字节，假设 float32）
    model_size_mb = total_params * 4 / (1024 * 1024)
    
    return {
        "type": type(model).__name__,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size_mb,
    }


def test_load_model(model_path):
    """测试加载单个模型"""
    print("\n" + "=" * 70)
    print(f"📦 测试加载模型: {os.path.basename(model_path)}")
    print("=" * 70)
    
    # 1. 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 文件不存在: {model_path}")
        return False
    
    # 2. 显示文件信息
    file_size = os.path.getsize(model_path)
    print(f"\n📁 文件信息:")
    print(f"   路径: {model_path}")
    print(f"   大小: {format_size(file_size)}")
    print(f"   状态: 文件在磁盘上（未加载到内存）")
    
    # 3. 加载模型
    print(f"\n⏳ 开始加载模型到内存...")
    print(f"   执行: torch.load('{os.path.basename(model_path)}', map_location='cpu')")
    
    start_time = time.time()
    
    try:
        # 加载模型
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        
        load_time = time.time() - start_time
        
        print(f"   ✓ 加载成功！")
        print(f"   ✓ 耗时: {load_time:.2f} 秒")
        
        # 4. 显示模型信息
        print(f"\n📊 模型信息:")
        model_info = get_model_info(model)
        print(f"   类型: {model_info['type']}")
        print(f"   总参数量: {model_info['total_params']:,}")
        print(f"   可训练参数: {model_info['trainable_params']:,}")
        print(f"   模型大小（估算）: {model_info['model_size_mb']:.2f} MB")
        
        # 5. 设置评估模式
        print(f"\n🔧 设置模型为评估模式...")
        model.eval()
        print(f"   ✓ model.eval() 完成")
        
        # 6. 测试模型是否能正常推理（可选）
        print(f"\n🧪 测试模型推理功能...")
        try:
            # 创建一个测试输入（假设是 GPT-2 trunk 的输入格式）
            # GPT-2 trunk 通常接受 [batch, seq_len, hidden_dim] 格式
            test_input = torch.randn(1, 5, 768)  # 小输入用于测试
            
            with torch.no_grad():
                output = model(test_input)
            
            print(f"   ✓ 推理测试成功")
            print(f"   输入形状: {test_input.shape}")
            print(f"   输出形状: {output.shape}")
            
            # 显示输出统计
            print(f"   输出统计:")
            print(f"     最小值: {output.min().item():.6f}")
            print(f"     最大值: {output.max().item():.6f}")
            print(f"     平均值: {output.mean().item():.6f}")
            
        except Exception as e:
            print(f"   ⚠️  推理测试失败: {e}")
            print(f"   （这可能是因为输入格式不正确，但不影响加载测试）")
        
        # 7. 内存使用情况
        print(f"\n💾 内存使用情况:")
        print(f"   文件大小（磁盘）: {format_size(file_size)}")
        print(f"   模型大小（内存估算）: {format_size(model_info['model_size_mb'] * 1024 * 1024)}")
        print(f"   加载时间: {load_time:.2f} 秒")
        if load_time > 0:
            print(f"   加载速度: {file_size / (1024 * 1024) / load_time:.2f} MB/s")
        
        print("\n" + "=" * 70)
        print("✅ 模型加载测试完成")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("🧪 模型从硬盘加载到内存测试")
    print("=" * 70)
    print("\n📋 测试目标:")
    print("   1. 检查模型文件是否存在")
    print("   2. 从磁盘加载模型到内存")
    print("   3. 显示模型信息")
    print("   4. 测试模型是否能正常推理")
    print("   5. 显示加载时间和性能统计")
    
    # 获取模型文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_files = [os.path.join(current_dir, f) for f in MODEL_FILES]
    
    # 只测试存在的文件
    existing_files = [f for f in model_files if os.path.exists(f)]
    
    if not existing_files:
        print(f"\n❌ 没有找到任何模型文件")
        print(f"   查找路径: {current_dir}")
        print(f"   查找文件: {', '.join(MODEL_FILES)}")
        return 1
    
    print(f"\n📁 找到 {len(existing_files)} 个模型文件:")
    for f in existing_files:
        print(f"   - {os.path.basename(f)}")
    
    # 测试每个模型
    results = []
    for model_file in existing_files:
        success = test_load_model(model_file)
        results.append((os.path.basename(model_file), success))
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

