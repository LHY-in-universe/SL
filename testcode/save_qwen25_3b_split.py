"""
创建并保存 Qwen2.5-3B 分割模型为 .pt 文件
前3层 + 后2层配置
"""
import sys
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
splitlearn_path = os.path.join(project_root, 'SplitLearning', 'src')
sys.path.insert(0, splitlearn_path)

print("=" * 70)
print("创建并保存 Qwen2.5-3B 分割模型")
print("=" * 70)
print("\n配置:")
print("  - 模型: Qwen2.5-3B (28层)")
print("  - Bottom: 前3层 (层0-2)")
print("  - Trunk: 中间23层 (层3-25)")
print("  - Top: 后2层 (层26-27)")
print("\n首次运行会下载完整模型 (~3.1GB)，请耐心等待...\n")

try:
    from splitlearn import ModelFactory
    
    print("【1】创建分割模型...")
    bottom, trunk, top = ModelFactory.create_split_models(
        model_type='qwen2',
        model_name_or_path='Qwen/Qwen2.5-3B',
        split_point_1=3,
        split_point_2=26,
        device='cpu'
    )
    print("   ✓ 模型创建完成\n")
    
    # 保存路径
    save_dir = current_dir
    bottom_path = os.path.join(save_dir, 'qwen25_3b_bottom_cached.pt')
    trunk_path = os.path.join(save_dir, 'qwen25_3b_trunk_cached.pt')
    top_path = os.path.join(save_dir, 'qwen25_3b_top_cached.pt')
    
    print("【2】保存分割模型...")
    
    # 保存 Bottom
    print(f"   保存 Bottom 到 {os.path.basename(bottom_path)}...")
    torch.save(bottom.state_dict(), bottom_path)
    size_mb = os.path.getsize(bottom_path) / (1024*1024)
    print(f"   ✓ Bottom 已保存 ({size_mb:.1f}MB)")
    
    # 保存 Trunk
    print(f"   保存 Trunk 到 {os.path.basename(trunk_path)}...")
    torch.save(trunk.state_dict(), trunk_path)
    size_mb = os.path.getsize(trunk_path) / (1024*1024)
    print(f"   ✓ Trunk 已保存 ({size_mb:.1f}MB)")
    
    # 保存 Top
    print(f"   保存 Top 到 {os.path.basename(top_path)}...")
    torch.save(top.state_dict(), top_path)
    size_mb = os.path.getsize(top_path) / (1024*1024)
    print(f"   ✓ Top 已保存 ({size_mb:.1f}MB)")
    
    # 总大小
    total_size = (os.path.getsize(bottom_path) + 
                  os.path.getsize(trunk_path) + 
                  os.path.getsize(top_path)) / (1024*1024)
    
    print(f"\n   总大小: {total_size:.1f}MB")
    
    print("\n【3】验证加载...")
    # 测试能否重新加载
    from transformers import Qwen2Config
    config = Qwen2Config.from_pretrained('Qwen/Qwen2.5-3B')
    
    from splitlearn.models.qwen2 import Qwen2BottomModel
    test_bottom = Qwen2BottomModel(config, end_layer=3)
    test_bottom.load_state_dict(torch.load(bottom_path))
    print("   ✓ 可以成功重新加载")
    
    print("\n" + "=" * 70)
    print("✅ 完成！")
    print("=" * 70)
    
    print("\n【生成的文件】")
    print(f"  • {bottom_path}")
    print(f"  • {trunk_path}")
    print(f"  • {top_path}")
    
    print("\n【使用方法】")
    print("""
from transformers import Qwen2Config
from splitlearn.models.qwen2 import Qwen2BottomModel, Qwen2TrunkModel, Qwen2TopModel
import torch

# 加载配置
config = Qwen2Config.from_pretrained('Qwen/Qwen2.5-3B')

# 加载分割模型
bottom = Qwen2BottomModel(config, end_layer=3)
bottom.load_state_dict(torch.load('qwen25_3b_bottom_cached.pt'))

trunk = Qwen2TrunkModel(config, start_layer=3, end_layer=26)
trunk.load_state_dict(torch.load('qwen25_3b_trunk_cached.pt'))

top = Qwen2TopModel(config, start_layer=26)
top.load_state_dict(torch.load('qwen25_3b_top_cached.pt'))

# 使用
# ...
""")
    
except KeyboardInterrupt:
    print("\n\n⚠️ 用户中断")
    sys.exit(0)
    
except Exception as e:
    print(f"\n✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

