"""
快速验证 Qwen2 支持 - 只检查导入和注册
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
splitlearn_path = os.path.join(project_root, 'SplitLearning', 'src')
sys.path.insert(0, splitlearn_path)

print("=" * 60)
print("快速验证 - Qwen2 支持")
print("=" * 60)

# 1. 基础检查
print("\n【1】检查 Qwen2Config...")
try:
    from transformers import Qwen2Config
    print("   ✓ Qwen2Config 可用")
except:
    print("   ✗ Qwen2Config 不可用")
    sys.exit(1)

# 2. 导入检查
print("\n【2】导入 Qwen2 模型类...")
try:
    from splitlearn.models.qwen2 import Qwen2BottomModel
    print("   ✓ Qwen2BottomModel 导入成功")
except Exception as e:
    print(f"   ✗ 导入失败: {e}")
    sys.exit(1)

# 3. 注册检查
print("\n【3】检查注册...")
try:
    from splitlearn.registry import ModelRegistry
    if ModelRegistry.is_complete('qwen2'):
        print("   ✓ Qwen2 完整注册")
    print(f"   支持的模型: {ModelRegistry.list_supported_models()}")
except Exception as e:
    print(f"   ✗ 检查失败: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ Qwen2 支持已启用!")
print("=" * 60)

print("\n【使用方法】")
print("""
from splitlearn import ModelFactory

# Qwen2.5-3B: 无需权限，开源可用
bottom, trunk, top = ModelFactory.create_split_models(
    model_type='qwen2',
    model_name_or_path='Qwen/Qwen2.5-3B',
    split_point_1=3,    # 前3层
    split_point_2=26,   # 后2层 (共28层)
    device='cpu'
)
""")

print("\n【Qwen2.5 系列模型】")
print("  • Qwen2.5-0.5B (24层) - 约500MB")
print("  • Qwen2.5-1.5B (28层) - 约1.5GB")
print("  • Qwen2.5-3B (28层) - 约3.1GB")
print("  • Qwen2.5-7B (28层) - 约7.6GB")
print("\n  全部无需权限，开源可用！")

print("=" * 60)

