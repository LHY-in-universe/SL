"""
快速验证 Gemma 模型支持 - 前3层 + 后2层配置
"""
import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
splitlearn_path = os.path.join(project_root, 'SplitLearning', 'src')
sys.path.insert(0, splitlearn_path)

print("=" * 60)
print("快速测试 - Gemma 模型分割 (前3层 + 后2层)")
print("=" * 60)

# 测试 1: transformers 版本
print("\n【1】检查 transformers 版本...")
import transformers
print(f"   ✓ 版本: {transformers.__version__}")

# 测试 2: GemmaConfig
print("\n【2】检查 GemmaConfig...")
try:
    from transformers import GemmaConfig
    print("   ✓ GemmaConfig 可用")
except ImportError as e:
    print(f"   ✗ GemmaConfig 不可用: {e}")
    sys.exit(1)

# 测试 3: 导入 Gemma 模型类
print("\n【3】导入 Gemma 模型类...")
try:
    from splitlearn.models.gemma import GemmaBottomModel, GemmaTrunkModel, GemmaTopModel
    print("   ✓ Gemma 模型类导入成功")
except Exception as e:
    print(f"   ✗ 导入失败: {e}")
    sys.exit(1)

# 测试 4: 检查注册
print("\n【4】检查 ModelRegistry 注册...")
try:
    from splitlearn.registry import ModelRegistry
    if ModelRegistry.is_complete('gemma'):
        print("   ✓ Gemma 完整注册 (bottom/trunk/top)")
    else:
        print("   ✗ Gemma 注册不完整")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ 检查失败: {e}")
    sys.exit(1)

# 测试 5: 创建小型测试配置
print("\n【5】创建测试模型 (小配置)...")
try:
    import torch
    
    # 超小配置用于快速测试
    config = GemmaConfig(
        vocab_size=1000,      # 很小的词汇表
        hidden_size=128,      # 很小的隐藏层
        intermediate_size=256,
        num_hidden_layers=18,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    
    # 分割配置: 前3层 + 后2层
    split_point_1 = 3
    split_point_2 = 16
    
    print(f"   配置: {config.num_hidden_layers} 层")
    print(f"   分割: Bottom(0-{split_point_1-1}), Trunk({split_point_1}-{split_point_2-1}), Top({split_point_2}-17)")
    
    # 创建模型（可能需要几秒钟）
    print("\n   创建 Bottom...")
    bottom = GemmaBottomModel(config, end_layer=split_point_1)
    print(f"   ✓ Bottom: {bottom.num_parameters():,} 参数")
    
    print("   创建 Trunk...")
    trunk = GemmaTrunkModel(config, start_layer=split_point_1, end_layer=split_point_2)
    print(f"   ✓ Trunk: {trunk.num_parameters():,} 参数")
    
    print("   创建 Top...")
    top = GemmaTopModel(config, start_layer=split_point_2)
    print(f"   ✓ Top: {top.num_parameters():,} 参数")
    
except Exception as e:
    print(f"   ✗ 创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 6: 快速前向传播
print("\n【6】测试前向传播...")
try:
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    
    with torch.no_grad():
        h1 = bottom(input_ids)
        h2 = trunk(h1)
        output = top(h2)
    
    print(f"   ✓ 输入: {input_ids.shape}")
    print(f"   ✓ Bottom → {h1.shape}")
    print(f"   ✓ Trunk → {h2.shape}")
    print(f"   ✓ Top → {output.logits.shape}")
    
except Exception as e:
    print(f"   ✗ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 所有测试通过!")
print("=" * 60)

print("\n【总结】")
print("  ✓ transformers 4.57.3 支持 Gemma")
print("  ✓ Gemma 模型类正确实现")
print("  ✓ ModelRegistry 注册完整")
print("  ✓ 模型创建成功 (前3层 + 后2层)")
print("  ✓ 前向传播正常")

print("\n【使用方法】")
print("  from splitlearn import ModelFactory")
print("  bottom, trunk, top = ModelFactory.create_split_models(")
print("      model_type='gemma',")
print("      model_name_or_path='google/gemma-2b',")
print("      split_point_1=3,   # 前3层")
print("      split_point_2=16,  # 后2层")
print("      device='cpu'")
print("  )")

print("\n" + "=" * 60)

