#!/usr/bin/env python3
"""
检查 Python 和 PyTorch 环境
判断是否会出现 mutex 警告
"""

import sys
import os

print("=" * 70)
print("Python 环境检测")
print("=" * 70)

# 基本信息
print(f"\nPython 信息:")
print(f"  路径: {sys.executable}")
print(f"  版本: {sys.version.split()[0]}")

# 检查是否是 Anaconda
is_anaconda = 'anaconda' in sys.executable.lower() or 'conda' in sys.executable.lower()
if is_anaconda:
    print(f"  类型: ⚠️  Anaconda/Conda")
    if 'CONDA_DEFAULT_ENV' in os.environ:
        print(f"  Conda 环境: {os.environ['CONDA_DEFAULT_ENV']}")
else:
    print(f"  类型: ✓ 系统 Python 或 Framework Python")

# 检查 PyTorch
print(f"\nPyTorch 信息:")
try:
    import torch
    print(f"  版本: {torch.__version__}")
    print(f"  位置: {torch.__file__}")

    # 检查是否使用 MKL
    parallel_info = torch.__config__.parallel_info()
    uses_mkl = 'mkl' in parallel_info.lower()

    if uses_mkl:
        print(f"  数学库: ⚠️  Intel MKL")
    else:
        print(f"  数学库: ✓ 非 MKL (OpenBLAS 或其他)")

except ImportError:
    print(f"  状态: ❌ 未安装")
    uses_mkl = None
except Exception as e:
    print(f"  错误: {e}")
    uses_mkl = None

# 预测
print(f"\n" + "=" * 70)
print("mutex 警告预测")
print("=" * 70)

if is_anaconda and uses_mkl:
    print("""
⚠️  可能会出现 mutex 警告

原因:
  - 使用 Anaconda Python
  - PyTorch 链接到 Intel MKL
  - MKL 在初始化时会创建 mutex 锁

是否正常:
  ✓ 完全正常，这是 Anaconda PyTorch 的已知特性
  ✓ 只在初始化时出现一次
  ✓ 不影响任何功能
  ✓ 不影响性能

如何处理:
  1. 忽略警告（推荐）- 功能完全正常
  2. 切换到 Framework Python - 不会出现警告
  3. 阅读 'Anaconda环境mutex问题说明.md' 了解详情
""")
elif is_anaconda and not uses_mkl:
    print("""
✓ 可能不会出现 mutex 警告

原因:
  - 使用 Anaconda Python
  - 但 PyTorch 不使用 MKL
""")
else:
    print("""
✓ 不太可能出现 mutex 警告

原因:
  - 不是 Anaconda Python
  - 或者 PyTorch 不使用 MKL
""")

print("=" * 70)
