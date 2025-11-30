"""
Qwen2.5-3B 模型拆分测试 - 前3层 + 后2层

测试 Qwen2.5-3B 模型的拆分功能：
- 模型: Qwen/Qwen2.5-3B (28层)
- Bottom: 前 3 层 (层 0-2)
- Trunk: 中间 23 层 (层 3-25)
- Top: 后 2 层 (层 26-27)

优势: Qwen2.5 无需 Hugging Face 权限，开源可用
"""
import sys
import os
import torch

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
splitlearn_path = os.path.join(project_root, 'SplitLearning', 'src')
sys.path.insert(0, splitlearn_path)

print("=" * 70)
print("Qwen2.5-3B 模型拆分测试 - 前3层 + 后2层")
print("=" * 70)

# ============================================================================
# 测试 1: 检查环境
# ============================================================================
print("\n【测试 1】检查环境...")
try:
    import transformers
    print(f"   ✓ transformers 版本: {transformers.__version__}")
    
    from transformers import Qwen2Config
    print("   ✓ Qwen2Config 可用")
    
except Exception as e:
    print(f"   ✗ 环境检查失败: {e}")
    sys.exit(1)

# ============================================================================
# 测试 2: 导入模型类
# ============================================================================
print("\n【测试 2】导入 Qwen2 模型类...")
try:
    from splitlearn.models.qwen2 import Qwen2BottomModel, Qwen2TrunkModel, Qwen2TopModel
    print("   ✓ Qwen2 模型类导入成功")
    
    from splitlearn.registry import ModelRegistry
    from splitlearn import ModelFactory
    print("   ✓ ModelRegistry 和 ModelFactory 导入成功")
    
except Exception as e:
    print(f"   ✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 测试 3: 检查 Qwen2 注册
# ============================================================================
print("\n【测试 3】检查 Qwen2 注册状态...")
try:
    if ModelRegistry.is_model_registered('qwen2'):
        print("   ✓ Qwen2 已注册")
        
        if ModelRegistry.is_complete('qwen2'):
            print("   ✓ Qwen2 完整注册 (bottom/trunk/top)")
        else:
            print("   ✗ Qwen2 注册不完整")
            sys.exit(1)
    else:
        print("   ✗ Qwen2 未注册")
        sys.exit(1)
    
    # 显示所有已注册模型
    supported = ModelRegistry.list_supported_models()
    print(f"   已注册模型: {supported}")
    
except Exception as e:
    print(f"   ✗ 检查失败: {e}")
    sys.exit(1)

# ============================================================================
# 测试 4: 使用小配置测试（不下载模型）
# ============================================================================
print("\n【测试 4】创建小型测试配置...")
print("   (使用随机初始化，不下载预训练模型)")

try:
    # 创建一个小的测试配置（模拟 Qwen2.5-3B 结构）
    test_config = Qwen2Config(
        vocab_size=5000,          # 较小的词汇表用于测试
        hidden_size=512,          # 较小的隐藏层
        intermediate_size=1024,   
        num_hidden_layers=28,     # Qwen2.5-3B 标准层数
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=512,
    )
    print(f"   ✓ 创建测试配置: {test_config.num_hidden_layers} 层")
    
except Exception as e:
    print(f"   ✗ 配置创建失败: {e}")
    sys.exit(1)

# ============================================================================
# 测试 5: 创建分割模型
# ============================================================================
print("\n【测试 5】创建分割模型...")

# 分割点配置
split_point_1 = 3   # Bottom: 前3层 (0-2)
split_point_2 = 26  # Top: 后2层 (26-27)

print(f"\n   Qwen2.5-3B (28层) 分割配置:")
print(f"   ├─ Bottom: 层 0-{split_point_1-1}   (前 {split_point_1} 层)")
print(f"   ├─ Trunk:  层 {split_point_1}-{split_point_2-1}  (中间 {split_point_2-split_point_1} 层)")
print(f"   └─ Top:    层 {split_point_2}-27  (后 {test_config.num_hidden_layers-split_point_2} 层)")

try:
    # 创建 Bottom
    print(f"\n   创建 Bottom 模型...")
    bottom = Qwen2BottomModel(test_config, end_layer=split_point_1)
    print(f"   ✓ Bottom: {bottom.num_parameters():,} 参数, {bottom.memory_footprint_mb():.2f} MB")
    
    # 创建 Trunk
    print(f"   创建 Trunk 模型...")
    trunk = Qwen2TrunkModel(test_config, start_layer=split_point_1, end_layer=split_point_2)
    print(f"   ✓ Trunk:  {trunk.num_parameters():,} 参数, {trunk.memory_footprint_mb():.2f} MB")
    
    # 创建 Top
    print(f"   创建 Top 模型...")
    top = Qwen2TopModel(test_config, start_layer=split_point_2)
    print(f"   ✓ Top:    {top.num_parameters():,} 参数, {top.memory_footprint_mb():.2f} MB")
    
    # 统计
    total_params = bottom.num_parameters() + trunk.num_parameters() + top.num_parameters()
    total_memory = bottom.memory_footprint_mb() + trunk.memory_footprint_mb() + top.memory_footprint_mb()
    print(f"\n   总计: {total_params:,} 参数, {total_memory:.2f} MB")
    
except Exception as e:
    print(f"   ✗ 模型创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 测试 6: 前向传播
# ============================================================================
print("\n【测试 6】测试前向传播...")

try:
    batch_size = 2
    seq_len = 16
    
    # 创建测试输入
    input_ids = torch.randint(0, test_config.vocab_size, (batch_size, seq_len))
    print(f"   输入: {input_ids.shape}")
    
    # 前向传播
    with torch.no_grad():
        # Bottom
        print("   → Bottom...")
        hidden_1 = bottom(input_ids)
        print(f"   ✓ Bottom 输出: {hidden_1.shape}")
        assert hidden_1.shape == (batch_size, seq_len, test_config.hidden_size)
        
        # Trunk
        print("   → Trunk...")
        hidden_2 = trunk(hidden_1)
        print(f"   ✓ Trunk 输出: {hidden_2.shape}")
        assert hidden_2.shape == (batch_size, seq_len, test_config.hidden_size)
        
        # Top
        print("   → Top...")
        output = top(hidden_2)
        print(f"   ✓ Top logits: {output.logits.shape}")
        assert output.logits.shape == (batch_size, seq_len, test_config.vocab_size)
    
    # 输出统计
    print(f"\n   输出统计:")
    print(f"   - Logits 范围: [{output.logits.min().item():.3f}, {output.logits.max().item():.3f}]")
    print(f"   - Logits 均值: {output.logits.mean().item():.3f}")
    print(f"   - Logits 标准差: {output.logits.std().item():.3f}")
    
    # 预测示例
    predicted_ids = output.logits.argmax(dim=-1)
    print(f"\n   预测示例 (第1个样本前10个token):")
    print(f"   {predicted_ids[0, :10].tolist()}")
    
except Exception as e:
    print(f"   ✗ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 测试 7: ModelFactory 接口（可选）
# ============================================================================
print("\n【测试 7】测试 ModelFactory 创建接口...")

test_with_pretrained = False  # 设置为 True 以测试真实的 Qwen2.5-3B

if test_with_pretrained:
    print("\n   尝试下载并加载 Qwen2.5-3B 预训练模型...")
    print("   警告: 首次下载约 3.1GB，需要时间")
    
    try:
        from transformers import AutoTokenizer
        
        # 加载分词器
        print("   加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B')
        
        # 使用 ModelFactory 创建分割模型
        print("   创建分割模型...")
        bottom_pt, trunk_pt, top_pt = ModelFactory.create_split_models(
            model_type='qwen2',
            model_name_or_path='Qwen/Qwen2.5-3B',
            split_point_1=3,
            split_point_2=26,
            device='cpu'
        )
        
        print(f"   ✓ 预训练模型加载成功")
        print(f"   Bottom: {bottom_pt.num_parameters():,} 参数")
        print(f"   Trunk: {trunk_pt.num_parameters():,} 参数")
        print(f"   Top: {top_pt.num_parameters():,} 参数")
        
        # 简单测试
        test_text = "人工智能的未来是"
        input_ids = tokenizer.encode(test_text, return_tensors="pt")
        
        with torch.no_grad():
            h1 = bottom_pt(input_ids)
            h2 = trunk_pt(h1)
            output = top_pt(h2)
        
        next_token_id = output.logits[0, -1].argmax().item()
        next_token = tokenizer.decode([next_token_id])
        print(f"\n   测试生成: '{test_text}' → '{next_token}'")
        
    except Exception as e:
        print(f"   ⚠ 预训练模型测试失败: {e}")
        print("   提示: 确保网络连接正常，首次下载需要时间")
else:
    print("   ⊗ 跳过预训练模型测试 (test_with_pretrained=False)")
    print("\n   如需测试真实的 Qwen2.5-3B:")
    print("   1. 设置 test_with_pretrained = True")
    print("   2. 确保网络连接良好（首次下载 ~3.1GB）")
    print("   3. Qwen2.5 无需许可，开源可用！")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 70)
print("✅ 所有测试通过!")
print("=" * 70)

print("\n【测试总结】")
print("  ✓ Qwen2 模型类成功导入")
print("  ✓ Qwen2 在 ModelRegistry 中完整注册")
print("  ✓ 成功创建分割模型: Bottom(前3层) + Trunk(中间23层) + Top(后2层)")
print("  ✓ 前向传播正常工作")
print("  ✓ 输出形状正确")

print("\n【配置详情】")
print(f"  模型: Qwen2.5-3B")
print(f"  总层数: 28 层")
print(f"  分割点: split_point_1={split_point_1}, split_point_2={split_point_2}")
print(f"  - Bottom: 层 0-{split_point_1-1} (前 {split_point_1} 层)")
print(f"  - Trunk: 层 {split_point_1}-{split_point_2-1} (中间 {split_point_2-split_point_1} 层)")
print(f"  - Top: 层 {split_point_2}-27 (后 {28-split_point_2} 层)")

print("\n【优势】")
print("  ✓ 无需 Hugging Face 权限 - Qwen2.5 开源可用")
print("  ✓ 性能优异 - Qwen2.5 在多个基准上表现出色")
print("  ✓ 支持中英文 - 适合多语言应用")
print("  ✓ 代码完整 - 基于现有 Qwen2 实现")

print("\n【实际使用】")
print("""
from splitlearn import ModelFactory
from transformers import AutoTokenizer

# 无需权限，直接使用！
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B')
bottom, trunk, top = ModelFactory.create_split_models(
    model_type='qwen2',
    model_name_or_path='Qwen/Qwen2.5-3B',
    split_point_1=3,    # 前3层
    split_point_2=26,   # 后2层
    device='cpu'
)

# 使用
input_ids = tokenizer.encode("你好", return_tensors="pt")
with torch.no_grad():
    h1 = bottom(input_ids)
    h2 = trunk(h1)
    output = top(h2)
""")

print("\n" + "=" * 70)

