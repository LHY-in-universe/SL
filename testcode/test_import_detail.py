"""
更细粒度的导入测试
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearning', 'src'))

print("1. 导入 torch")
import torch
print("   OK")

print("2. 导入 torch.nn")
import torch.nn as nn
print("   OK")

print("3. 导入 splitlearn.core.base")
from splitlearn.core.base import BaseSplitModel
print("   OK")

print("4. 导入 splitlearn.core.bottom")
from splitlearn.core.bottom import BaseBottomModel
print("   OK")

print("5. 导入 splitlearn.core.trunk")
from splitlearn.core.trunk import BaseTrunkModel
print("   OK")

print("6. 导入 splitlearn.core.top")
from splitlearn.core.top import BaseTopModel
print("   OK")

print("\n✅ splitlearn.core 全部导入成功!")

