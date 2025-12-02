"""
检查 grpc 是否被间接导入
"""
import sys

print("1. 检查初始状态")
print(f"   grpc 在 sys.modules 中: {'grpc' in sys.modules}")

print("\n2. 导入 torch")
import torch
print(f"   grpc 在 sys.modules 中: {'grpc' in sys.modules}")

print("\n3. 导入 transformers")
from transformers import GPT2Config
print(f"   grpc 在 sys.modules 中: {'grpc' in sys.modules}")

print("\n4. 检查所有已加载的模块中是否有 grpc 相关")
grpc_modules = [m for m in sys.modules.keys() if 'grpc' in m.lower()]
print(f"   grpc 相关模块: {grpc_modules}")

print("\n5. 现在尝试导入 splitlearn")
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearning', 'src'))

# 显式禁用任何 grpc 相关的东西
os.environ['GRPC_VERBOSITY'] = 'NONE'
os.environ['GRPC_TRACE'] = ''

print("   开始导入 splitlearn...")
import splitlearn
print("   ✅ splitlearn 导入成功!")

print(f"\n6. 导入后 grpc 在 sys.modules 中: {'grpc' in sys.modules}")
grpc_modules = [m for m in sys.modules.keys() if 'grpc' in m.lower()]
print(f"   grpc 相关模块: {grpc_modules}")

