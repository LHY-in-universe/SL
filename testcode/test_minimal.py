"""
最小测试 - 找出问题根源
"""
print("1. 导入 os, sys")
import os
import sys
print("   OK")

print("2. 设置路径")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearning', 'src'))
print("   OK")

print("3. 导入 abc")
from abc import ABC, abstractmethod
print("   OK")

print("4. 导入 typing")
from typing import Dict, Any, Optional
print("   OK")

print("5. 导入 pathlib")
from pathlib import Path
print("   OK")

print("6. 导入 json")
import json
print("   OK")

print("7. 导入 torch (这一步可能触发问题)")
import torch
print("   OK")

print("8. 导入 torch.nn")
import torch.nn as nn
print("   OK")

print("9. 定义一个简单类")
class TestClass(nn.Module, ABC):
    def __init__(self):
        super().__init__()
print("   OK")

print("\n✅ 基础导入全部成功!")
print("\n现在测试直接读取 base.py 文件内容...")

# 直接执行 base.py 的内容
base_path = os.path.join(project_root, 'SplitLearning', 'src', 'splitlearn', 'core', 'base.py')
print(f"读取: {base_path}")
with open(base_path) as f:
    content = f.read()
print(f"文件大小: {len(content)} 字符")
print("\n尝试 exec 执行...")
exec(content)
print("   OK")

