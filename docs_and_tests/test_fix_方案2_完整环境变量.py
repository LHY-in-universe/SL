#!/usr/bin/env python3
"""
方案 2: 使用完整的 MKL 环境变量设置
"""
import os
import sys
import signal

# 超时处理
def timeout_handler(signum, frame):
    print("\n❌ 超时！仍然卡住")
    sys.exit(1)

print("=" * 70)
print("测试方案 2: 完整的 MKL 环境变量设置")
print("=" * 70)

# ✅ 完整的环境变量设置
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'  # ← 关键！强制串行
os.environ['MKL_DYNAMIC'] = 'FALSE'               # ← 禁用动态调整
os.environ['NUMEXPR_NUM_THREADS'] = '1'

print("\n环境变量:")
print(f"  OMP_NUM_THREADS = {os.environ['OMP_NUM_THREADS']}")
print(f"  MKL_NUM_THREADS = {os.environ['MKL_NUM_THREADS']}")
print(f"  MKL_THREADING_LAYER = {os.environ['MKL_THREADING_LAYER']}")
print(f"  MKL_DYNAMIC = {os.environ['MKL_DYNAMIC']}")

# 添加路径
sys.path.insert(0, '/Users/lhy/Desktop/Git/SL/SplitLearnCore/src')

# 设置 30 秒超时
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)

print("\n导入 splitlearn_core（30秒超时）...")
try:
    import splitlearn_core
    signal.alarm(0)  # 取消超时
    print("✓ 导入成功！")

    # 测试功能
    print("\n测试 ModelFactory...")
    from splitlearn_core import ModelFactory
    print("✓ ModelFactory 可用")

    print("\n" + "=" * 70)
    print("✅ 方案 2 有效！")
    print("=" * 70)

except Exception as e:
    signal.alarm(0)
    print(f"\n✗ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
