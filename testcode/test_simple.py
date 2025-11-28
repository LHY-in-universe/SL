"""
简化测试 - 使用本地创建的小模型而不是下载 GPT-2
"""
import sys
import os
import torch
import torch.nn as nn

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
splitlearn_path = os.path.join(project_root, 'SplitLearning', 'src')
sys.path.insert(0, splitlearn_path)

print("=== 简化的 SplitLearning 测试 ===\n")

# 测试 1: 导入检查
print("1. 测试导入...")
try:
    from splitlearn.core import BaseBottomModel, BaseTrunkModel, BaseTopModel
    print("   ✓ 核心类导入成功")
    
    from splitlearn.registry import ModelRegistry
    print("   ✓ ModelRegistry 导入成功")
    
    from splitlearn.models.gpt2 import GPT2BottomModel, GPT2TrunkModel, GPT2TopModel
    print("   ✓ GPT2 模型类导入成功")
    
except Exception as e:
    print(f"   ✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 2: 检查注册状态
print("\n2. 检查模型注册状态...")
try:
    info = ModelRegistry.get_model_info()
    print(f"   已注册模型: {list(info.keys())}")
    
    for model_type, status in info.items():
        complete = "✓" if status['complete'] else "✗"
        print(f"   {complete} {model_type}: {status}")
        
except Exception as e:
    print(f"   ✗ 检查失败: {e}")

# 测试 3: 创建简单的配置对象（不加载实际模型）
print("\n3. 测试配置对象...")
try:
    from transformers import GPT2Config
    
    # 创建一个小的配置用于测试
    config = GPT2Config(
        vocab_size=1000,
        n_positions=128,
        n_embd=256,
        n_layer=4,  # 只有 4 层
        n_head=4
    )
    print(f"   ✓ 创建测试配置: {config.n_layer} 层")
    
    # 测试能否实例化模型类（使用随机初始化，不加载预训练权重）
    print("\n4. 测试模型实例化（随机初始化）...")
    
    # 创建一个简单的 state_dict
    dummy_state_dict = {}
    
    # 尝试创建 Bottom 模型
    print("   创建 Bottom 模型...")
    bottom = GPT2BottomModel(config, end_layer=2)
    print(f"   ✓ Bottom 模型创建成功，参数量: {bottom.num_parameters():,}")
    
    # 尝试创建 Trunk 模型
    print("   创建 Trunk 模型...")
    trunk = GPT2TrunkModel(config, start_layer=2, end_layer=3)
    print(f"   ✓ Trunk 模型创建成功，参数量: {trunk.num_parameters():,}")
    
    # 尝试创建 Top 模型
    print("   创建 Top 模型...")
    top = GPT2TopModel(config, start_layer=3)
    print(f"   ✓ Top 模型创建成功，参数量: {top.num_parameters():,}")
    
    # 测试前向传播
    print("\n5. 测试前向传播...")
    batch_size, seq_len = 2, 10
    
    # 创建随机输入
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    print(f"   输入形状: {input_ids.shape}")
    
    # Bottom
    hidden_1 = bottom(input_ids)
    print(f"   ✓ Bottom 输出形状: {hidden_1.shape}")
    
    # Trunk
    hidden_2 = trunk(hidden_1)
    print(f"   ✓ Trunk 输出形状: {hidden_2.shape}")
    
    # Top
    output = top(hidden_2)
    print(f"   ✓ Top 输出 logits 形状: {output.logits.shape}")
    
    print("\n=== 所有测试通过! ===")
    print("\n注意: 这个测试使用随机初始化的模型，不是预训练模型。")
    print("如果要测试预训练模型，需要确保网络连接正常以下载模型文件。")
    
except Exception as e:
    print(f"   ✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
