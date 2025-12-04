#!/usr/bin/env python3
"""
方案 3: 直接导入 factory，避免触发 models 的批量导入
"""
import os
import sys
import signal

# 超时处理
def timeout_handler(signum, frame):
    print("\n❌ 超时！仍然卡住")
    sys.exit(1)

print("=" * 70)
print("测试方案 3: 直接导入 factory 模块")
print("=" * 70)

# 基本环境变量
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

print("\n环境变量:")
print(f"  OMP_NUM_THREADS = {os.environ['OMP_NUM_THREADS']}")
print(f"  MKL_NUM_THREADS = {os.environ['MKL_NUM_THREADS']}")

# 添加路径
sys.path.insert(0, '/Users/lhy/Desktop/Git/SL/SplitLearnCore/src')

# 设置 30 秒超时
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)

print("\n方法 A: 导入整个包（会触发 models）...")
print("  import splitlearn_core  ← 这个会卡住")
print("  ⏭️  跳过此步骤\n")

print("方法 B: 直接导入 factory（不触发 models）...")
try:
    from splitlearn_core.factory import ModelFactory
    signal.alarm(0)  # 取消超时
    print("✓ ModelFactory 导入成功！")

    # 测试实际使用
    print("\n测试创建模型...")
    signal.alarm(60)  # 加载模型可能需要更长时间

    bottom, trunk, top = ModelFactory.create_split_models(
        model_type='gpt2',
        model_name_or_path='gpt2',
        split_point_1=2,
        split_point_2=10,
        device='cpu'
    )
    signal.alarm(0)

    print("✓ 模型创建成功！")
    print(f"  Bottom: {sum(p.numel() for p in bottom.parameters())/1e6:.2f}M 参数")
    print(f"  Trunk:  {sum(p.numel() for p in trunk.parameters())/1e6:.2f}M 参数")
    print(f"  Top:    {sum(p.numel() for p in top.parameters())/1e6:.2f}M 参数")

    print("\n" + "=" * 70)
    print("✅ 方案 3 有效！")
    print("=" * 70)
    print("\n关键：使用 'from splitlearn_core.factory import ModelFactory'")
    print("而不是 'from splitlearn_core import ModelFactory'")

except Exception as e:
    signal.alarm(0)
    print(f"\n✗ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
