#!/usr/bin/env python
"""
强制 CPU 模式测试 Qwen2.5-3B - 避免 MPS 锁问题
"""
import os
# 在导入任何库之前设置环境变量
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
sys.path.insert(0, '/Users/lhy/Desktop/Git/SL/SplitLearning/src')

print("=" * 70)
print("Qwen2.5-3B CPU 模式测试")
print("=" * 70)

# 使用命令行方式，避免交互式导入问题
import subprocess

script = """
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

import sys
sys.path.insert(0, '/Users/lhy/Desktop/Git/SL/SplitLearning/src')

from splitlearn import ModelFactory
import torch

print("创建 Qwen2.5-3B 分割模型...")
bottom, trunk, top = ModelFactory.create_split_models(
    model_type='qwen2',
    model_name_or_path='Qwen/Qwen2.5-3B',
    split_point_1=3,
    split_point_2=26,
    device='cpu',
    low_memory=True,
    verbose=True
)

print("\\n保存模型...")
torch.save(bottom.state_dict(), 'qwen25_3b_bottom_cached.pt')
torch.save(trunk.state_dict(), 'qwen25_3b_trunk_cached.pt')
torch.save(top.state_dict(), 'qwen25_3b_top_cached.pt')
print("✓ 完成")
"""

result = subprocess.run(
    ['python', '-c', script],
    cwd='/Users/lhy/Desktop/Git/SL/testcode',
    capture_output=True,
    text=True,
    timeout=300
)

print(result.stdout)
if result.stderr:
    print("错误:", result.stderr)

