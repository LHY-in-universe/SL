"""
逐步测试 - 精确定位问题在哪个 __init__.py
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearning', 'src'))

import torch
print("torch OK")

# 测试1: 直接导入 splitlearn 包的 __init__.py
print("\n测试: import splitlearn")
print("  这会触发 splitlearn/__init__.py ...")
import splitlearn
print("  ✅ OK")

