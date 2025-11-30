#!/bin/bash
cd /Users/lhy/Desktop/Git/SL/testcode

export PYTORCH_ENABLE_MPS_FALLBACK=0
export OMP_NUM_THREADS=1

python3 -u << 'PYEOF'
import sys
sys.path.insert(0, '/Users/lhy/Desktop/Git/SL/SplitLearning/src')

print("=" * 70)
print("Qwen2.5-3B 分割测试")
print("=" * 70)

try:
    from splitlearn import ModelFactory
    import torch
    import os
    
    print("\n【1】创建分割模型...")
    print("配置: 前3层 + 后2层")
    print("模式: 增量加载\n")
    
    bottom, trunk, top = ModelFactory.create_split_models(
        model_type='qwen2',
        model_name_or_path='Qwen/Qwen2.5-3B',
        split_point_1=3,
        split_point_2=26,
        device='cpu',
        low_memory=True,
        verbose=True
    )
    
    print("\n【2】保存为 .pt 文件...")
    torch.save(bottom.state_dict(), 'qwen25_3b_bottom.pt')
    print("✓ Bottom saved")
    
    torch.save(trunk.state_dict(), 'qwen25_3b_trunk.pt')
    print("✓ Trunk saved")
    
    torch.save(top.state_dict(), 'qwen25_3b_top.pt')
    print("✓ Top saved")
    
    # 显示文件大小
    b_size = os.path.getsize('qwen25_3b_bottom.pt') / (1024*1024)
    t_size = os.path.getsize('qwen25_3b_trunk.pt') / (1024*1024)
    o_size = os.path.getsize('qwen25_3b_top.pt') / (1024*1024)
    
    print(f"\n文件大小:")
    print(f"  Bottom: {b_size:.1f}MB")
    print(f"  Trunk: {t_size:.1f}MB")
    print(f"  Top: {o_size:.1f}MB")
    print(f"  Total: {b_size+t_size+o_size:.1f}MB")
    
    print("\n✅ 测试完成！")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

PYEOF

