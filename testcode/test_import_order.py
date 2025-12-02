"""
测试导入顺序 - 找出 mutex blocking 的原因
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearning', 'src'))
sys.path.insert(0, os.path.join(project_root, 'splitlearn-comm', 'src'))

print("测试 1: 只导入 torch")
import torch
print("  ✅ torch OK")

print("\n测试 2: 导入 transformers")
from transformers import GPT2Config
print("  ✅ transformers OK")

print("\n测试 3: 导入 splitlearn.core")
from splitlearn.core import BaseSplitModel
print("  ✅ splitlearn.core OK")

print("\n测试 4: 导入 splitlearn.registry")
from splitlearn.registry import ModelRegistry
print("  ✅ splitlearn.registry OK")

print("\n测试 5: 导入 splitlearn.models.gpt2.bottom (关键!)")
from splitlearn.models.gpt2.bottom import GPT2BottomModel
print("  ✅ GPT2BottomModel OK")

print("\n测试 6: 导入 splitlearn.models.gpt2.top")
from splitlearn.models.gpt2.top import GPT2TopModel
print("  ✅ GPT2TopModel OK")

print("\n测试 7: torch.load 加载模型")
bottom_path = os.path.join(current_dir, "gpt2_bottom_cached.pt")
bottom = torch.load(bottom_path, map_location='cpu', weights_only=False)
print("  ✅ Bottom 模型加载 OK")

print("\n测试 8: 导入 grpc")
import grpc
print("  ✅ grpc OK")

print("\n测试 9: 导入 splitlearn_comm")
from splitlearn_comm import GRPCComputeClient
print("  ✅ splitlearn_comm OK")

print("\n" + "=" * 50)
print("✅ 所有导入测试通过！")

