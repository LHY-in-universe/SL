"""
使用增量加载创建 Qwen2.5-3B 分割模型
只下载需要的分片，不下载整个模型！
"""
import sys
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
splitlearn_path = os.path.join(project_root, 'SplitLearning', 'src')
sys.path.insert(0, splitlearn_path)

print("=" * 70)
print("Qwen2.5-3B 增量加载 - 只下载需要的分片")
print("=" * 70)
print("\n优势:")
print("  ✓ 只下载前3层和后2层需要的权重")
print("  ✓ 不下载整个3.1GB模型")
print("  ✓ 节省带宽和时间")
print("  ✓ 降低内存峰值")
print()

try:
    from splitlearn import ModelFactory
    
    print("【1】使用增量加载创建分割模型...")
    print("   配置:")
    print("   - 模型: Qwen2.5-3B (28层)")
    print("   - Bottom: 前3层")
    print("   - Trunk: 中间23层")
    print("   - Top: 后2层")
    print("   - 模式: low_memory=True (增量加载)")
    print()
    
    # 关键：使用 low_memory=True 启用增量加载
    bottom, trunk, top = ModelFactory.create_split_models(
        model_type='qwen2',
        model_name_or_path='Qwen/Qwen2.5-3B',
        split_point_1=3,
        split_point_2=26,
        device='cpu',
        low_memory=True,    # 🔑 关键参数：只下载需要的分片
        verbose=True         # 显示详细信息
    )
    
    print("\n   ✓ 模型创建完成！\n")
    
    print("【2】保存为 .pt 文件...")
    
    # 保存路径
    bottom_path = os.path.join(current_dir, 'qwen25_3b_bottom_cached.pt')
    trunk_path = os.path.join(current_dir, 'qwen25_3b_trunk_cached.pt')
    top_path = os.path.join(current_dir, 'qwen25_3b_top_cached.pt')
    
    # 保存
    torch.save(bottom.state_dict(), bottom_path)
    size_bottom = os.path.getsize(bottom_path) / (1024*1024)
    print(f"   ✓ Bottom: {size_bottom:.1f}MB")
    
    torch.save(trunk.state_dict(), trunk_path)
    size_trunk = os.path.getsize(trunk_path) / (1024*1024)
    print(f"   ✓ Trunk: {size_trunk:.1f}MB")
    
    torch.save(top.state_dict(), top_path)
    size_top = os.path.getsize(top_path) / (1024*1024)
    print(f"   ✓ Top: {size_top:.1f}MB")
    
    total_size = size_bottom + size_trunk + size_top
    print(f"\n   总大小: {total_size:.1f}MB")
    
    print("\n【3】快速测试...")
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B')
    input_ids = tokenizer.encode("你好", return_tensors="pt")
    
    with torch.no_grad():
        h1 = bottom(input_ids)
        h2 = trunk(h1)
        output = top(h2)
    
    print(f"   ✓ 推理成功: {output.logits.shape}")
    
    print("\n" + "=" * 70)
    print("✅ 完成！")
    print("=" * 70)
    
    print("\n【生成的文件】")
    print(f"  {os.path.basename(bottom_path)}")
    print(f"  {os.path.basename(trunk_path)}")
    print(f"  {os.path.basename(top_path)}")
    
except KeyboardInterrupt:
    print("\n\n⚠️ 用户中断")
    sys.exit(0)
    
except Exception as e:
    print(f"\n✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n💡 提示:")
    print("  如果是第一次运行，模型文件需要下载")
    print("  增量加载会智能地只下载需要的分片")
    print("  但仍需要一些时间，请耐心等待...")
    sys.exit(1)

