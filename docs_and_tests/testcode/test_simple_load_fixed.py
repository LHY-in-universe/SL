#!/usr/bin/env python3
"""
简单的模型加载测试 - 使用 core 库（修复版）

✅ 修复要点：在导入任何模块之前先设置环境变量
"""

# ============================================================================
# ✅ 第一步：先设置环境变量（在导入任何库之前！）
# ============================================================================
import os
import sys

print("[0/10] 设置环境变量（必须在导入任何模块之前！）")
os.environ.setdefault('OMP_NUM_THREADS', '1')
print("   ✓ OMP_NUM_THREADS = 1")
os.environ.setdefault('MKL_NUM_THREADS', '1')
print("   ✓ MKL_NUM_THREADS = 1")
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
print("   ✓ NUMEXPR_NUM_THREADS = 1")

# ============================================================================
# 第二步：添加路径
# ============================================================================
print("\n[1/10] 添加 SplitLearnCore 路径...")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
core_src_path = os.path.join(project_root, 'SplitLearnCore', 'src')
sys.path.insert(0, core_src_path)
print(f"   ✓ 路径已添加: {core_src_path}")

# ============================================================================
# 第三步：现在可以安全地导入 splitlearn_core
# ============================================================================
print("\n[2/10] 导入 splitlearn_core（环境变量已设置，不会有 mutex 警告）...")

print("   [2.1] 导入 splitlearn_core...")
import splitlearn_core
print("   ✓ splitlearn_core 导入成功")

print("   [2.2] 导入 splitlearn_core.models...")
import splitlearn_core.models
print("   ✓ splitlearn_core.models 导入成功")

print("   [2.3] 导入 splitlearn_core.models.gpt2...")
import splitlearn_core.models.gpt2
print("   ✓ splitlearn_core.models.gpt2 导入成功")

print("   [2.4] 导入 GPT2TrunkModel...")
from splitlearn_core.models.gpt2 import GPT2TrunkModel
print("   ✓ GPT2TrunkModel 导入成功！")

# ============================================================================
# 第四步：导入其他库
# ============================================================================
print("\n[3/10] 导入其他库...")
import time
print("   ✓ time")

import torch
print("   ✓ torch (版本: {})".format(torch.__version__))

from transformers import GPT2Config
print("   ✓ transformers")

def format_size(size_bytes):
    """格式化文件大小"""
    return f"{size_bytes / (1024*1024):.2f} MB"

def main():
    print("\n" + "=" * 70)
    print("🧪 简单模型加载测试（修复版）")
    print("=" * 70)

    # 测试文件
    model_path = os.path.join(current_dir, "gpt2_trunk_full.pt")

    # 1. 检查文件
    print(f"\n📁 检查文件: {os.path.basename(model_path)}")
    if not os.path.exists(model_path):
        print(f"❌ 文件不存在: {model_path}")
        print(f"\n提示：请先运行相应的脚本来创建模型文件")
        return 1

    file_size = os.path.getsize(model_path)
    print(f"   ✓ 文件存在")
    print(f"   ✓ 文件大小: {format_size(file_size)}")

    # 2. 方法 1: 直接使用 torch.load()
    print(f"\n" + "-" * 70)
    print("方法 1: 直接使用 torch.load()")
    print("-" * 70)

    print(f"\n⏳ 开始加载（torch.load）...")
    print(f"   这可能需要一些时间，请耐心等待...")

    start_time = time.time()
    try:
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        load_time = time.time() - start_time

        print(f"\n   ✓ 加载成功！")
        print(f"   ✓ 耗时: {load_time:.2f} 秒")
        print(f"   ✓ 模型类型: {type(model).__name__}")

        # 模型信息
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ 参数量: {total_params:,}")
        print(f"   ✓ 模型大小: {format_size(total_params * 4)}")

        # 测试推理
        print(f"\n🧪 测试推理...")
        model.eval()
        test_input = torch.randn(1, 5, 768)
        with torch.no_grad():
            output = model(test_input)
        print(f"   ✓ 推理成功")
        print(f"   输入: {test_input.shape} -> 输出: {output.shape}")

        method1_success = True

    except Exception as e:
        print(f"\n   ❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        method1_success = False

    # 3. 方法 2: 使用 core 库创建模型实例
    print(f"\n" + "-" * 70)
    print("方法 2: 使用 SplitLearnCore 模型类")
    print("-" * 70)

    print(f"\n⏳ 创建模型实例...")
    try:
        config = GPT2Config()
        model_instance = GPT2TrunkModel(
            config=config,
            start_layer=2,
            end_layer=10
        )
        print(f"   ✓ 模型实例创建成功")
        print(f"   ✓ 类型: {type(model_instance).__name__}")

        # 尝试加载 state_dict（如果文件包含 state_dict）
        print(f"\n⏳ 尝试加载 state_dict...")
        loaded_data = torch.load(model_path, map_location='cpu', weights_only=False)

        if isinstance(loaded_data, dict):
            print(f"   ✓ 文件包含 state_dict")
            model_instance.load_state_dict(loaded_data, strict=False)
            print(f"   ✓ state_dict 加载成功")
        else:
            print(f"   ℹ️  文件包含完整模型对象，不是 state_dict")
            print(f"   （这是正常的）")

        method2_success = True

    except Exception as e:
        print(f"\n   ❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        method2_success = False

    # 总结
    print(f"\n" + "=" * 70)
    print("📊 测试总结")
    print("=" * 70)
    print(f"   方法 1 (torch.load): {'✅ 成功' if method1_success else '❌ 失败'}")
    print(f"   方法 2 (core 库): {'✅ 成功' if method2_success else '❌ 失败'}")

    if method1_success or method2_success:
        print(f"\n✅ 至少有一种方法成功，模型加载功能正常！")
    else:
        print(f"\n❌ 两种方法都失败了，请检查模型文件")

    return 0 if (method1_success or method2_success) else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
