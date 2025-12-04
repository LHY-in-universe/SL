#!/usr/bin/env python3
"""
测试 PyTorch 初始化时的 mutex 警告

这个脚本演示了为什么即使设置了单线程，PyTorch 在初始化时仍可能出现 mutex 警告。
"""

import os
import sys

print("=" * 70)
print("PyTorch 初始化测试")
print("=" * 70)
print()

# 测试 1: 在导入 torch 之前设置环境变量
print("测试 1: 在导入 torch 之前设置环境变量")
print("-" * 70)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

print("环境变量已设置:")
print(f"  OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS')}")
print(f"  MKL_NUM_THREADS = {os.environ.get('MKL_NUM_THREADS')}")
print()

print("现在导入 torch...")
print("(注意：导入 torch 时会进行初始化，可能会触发 mutex 警告)")
print()

# 导入 torch - 这里可能会触发 mutex 警告
import torch

print("✓ torch 导入完成")
print()

# 测试 2: 检查线程数设置
print("测试 2: 检查 PyTorch 线程数")
print("-" * 70)
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print("✓ 线程数已设置为 1")
    print(f"  torch.get_num_threads() = {torch.get_num_threads()}")
    print(f"  torch.get_num_interop_threads() = {torch.get_num_interop_threads()}")
except RuntimeError as e:
    print(f"✗ 无法设置线程数: {e}")
    print("  (这可能是因为在导入时已经启动了并行工作)")
print()

# 测试 3: 解释为什么会有 mutex 警告
print("测试 3: 为什么会有 mutex 警告？")
print("-" * 70)
print("""
PyTorch 在导入时会进行以下初始化操作：

1. **C++ 后端初始化**
   - PyTorch 的底层是用 C++ 写的
   - 在 Python 导入 torch 时，C++ 代码会立即执行
   - 这些 C++ 代码会创建一些内部结构（如线程池、内存管理器等）

2. **线程池创建**
   - 即使设置了单线程，PyTorch 在初始化时可能会：
     - 创建内部线程池结构
     - 初始化互斥锁（mutex）来保护这些结构
     - 这些操作发生在设置线程数之前

3. **库依赖初始化**
   - PyTorch 依赖多个底层库（如 MKL、OpenMP）
   - 这些库在导入时也会进行初始化
   - 它们的初始化可能早于环境变量的生效

4. **为什么是"正常现象"**
   - 这个警告出现在初始化阶段，不是运行时
   - 初始化完成后，所有操作都会使用单线程
   - 不影响实际功能，只是初始化时的内部操作
""")

print()
print("=" * 70)
print("结论:")
print("=" * 70)
print("""
1. Mutex 警告出现在 PyTorch 导入/初始化时
2. 这是库内部初始化的正常行为
3. 初始化完成后，所有操作都会遵循单线程设置
4. 不影响功能，只是初始化时的内部操作

如果看到 1 个 mutex 警告，这通常是正常的。
只有在运行时出现大量 mutex 警告才需要关注。
""")

