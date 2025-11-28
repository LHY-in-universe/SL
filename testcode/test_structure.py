"""
简化的测试脚本 - 不使用实际的模型加载，只测试代码结构
"""
import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
splitlearn_path = os.path.join(project_root, 'SplitLearning', 'src')
sys.path.insert(0, splitlearn_path)

print("=== SplitLearning 代码结构测试 ===\n")

# 测试 1: 检查文件结构
print("1. 检查文件结构...")
expected_files = [
    'splitlearn/__init__.py',
    'splitlearn/factory.py',
    'splitlearn/registry.py',
    'splitlearn/models/gpt2/__init__.py',
]

for file in expected_files:
    full_path = os.path.join(splitlearn_path, file)
    exists = "✓" if os.path.exists(full_path) else "✗"
    print(f"   {exists} {file}")

# 测试 2: 尝试导入（不触发 torch）
print("\n2. 测试导入...")
try:
    # 先检查是否能找到模块
    import importlib.util
    spec = importlib.util.find_spec("splitlearn")
    if spec is None:
        print("   ✗ 无法找到 splitlearn 模块")
    else:
        print(f"   ✓ 找到 splitlearn 模块: {spec.origin}")
        
    print("\n3. 查看 ModelRegistry...")
    from splitlearn.registry import ModelRegistry
    print("   ✓ ModelRegistry 导入成功")
    
    # 列出已注册的模型
    info = ModelRegistry.get_model_info()
    if info:
        print(f"   已注册模型: {list(info.keys())}")
    else:
        print("   警告: 没有已注册的模型")
        
except Exception as e:
    print(f"   ✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 测试完成 ===")
print("\n注意: 由于 PyTorch 导入存在问题，建议:")
print("1. 重新安装 PyTorch: pip uninstall torch && pip install torch")
print("2. 或使用 conda: conda install pytorch -c pytorch")
