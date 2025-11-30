"""
测试 Gemma 模型分割 - 前3层 Bottom + 后2层 Top

本脚本测试 Gemma 模型的分割功能：
- Bottom: 前 3 层 (层 0-2)
- Trunk: 中间层 (层 3-16，对于 Gemma 2B)
- Top: 后 2 层 (层 16-18，对于 Gemma 2B)
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
print("测试 Gemma 模型分割 - 前3层 Bottom + 后2层 Top")
print("=" * 70)

# ============================================================================
# 测试 1: 导入检查
# ============================================================================
print("\n【测试 1】导入 Gemma 模型类...")
try:
    from splitlearn.models.gemma import GemmaBottomModel, GemmaTrunkModel, GemmaTopModel
    print("   ✓ Gemma 模型类导入成功")
    
    from splitlearn.registry import ModelRegistry
    print("   ✓ ModelRegistry 导入成功")
    
    from splitlearn import ModelFactory
    print("   ✓ ModelFactory 导入成功")
    
except Exception as e:
    print(f"   ✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 测试 2: 检查 Gemma 注册状态
# ============================================================================
print("\n【测试 2】检查 Gemma 注册状态...")
try:
    if ModelRegistry.is_model_registered('gemma'):
        print("   ✓ Gemma 已注册")
        
        if ModelRegistry.is_complete('gemma'):
            print("   ✓ Gemma 所有组件（bottom/trunk/top）已完整注册")
        else:
            print("   ✗ Gemma 注册不完整")
            sys.exit(1)
    else:
        print("   ✗ Gemma 未注册")
        sys.exit(1)
        
    # 显示注册信息
    info = ModelRegistry.get_model_info()
    gemma_info = info.get('gemma', {})
    print(f"   Gemma 注册详情: {gemma_info}")
    
except Exception as e:
    print(f"   ✗ 检查失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 测试 3: 使用随机初始化创建模型（不需要下载）
# ============================================================================
print("\n【测试 3】创建随机初始化的 Gemma 模型...")
print("   (使用小型配置进行测试，不下载预训练模型)")

try:
    from transformers import GemmaConfig
    
    # 创建一个小的测试配置（模拟 Gemma 2B 的结构但更小）
    test_config = GemmaConfig(
        vocab_size=5000,          # 较小的词汇表
        hidden_size=512,          # 较小的隐藏层
        intermediate_size=1024,   # 较小的中间层
        num_hidden_layers=18,     # Gemma 2B 标准层数
        num_attention_heads=8,    # 较少的注意力头
        num_key_value_heads=4,    # GQA
        max_position_embeddings=512,
    )
    print(f"   ✓ 创建测试配置: {test_config.num_hidden_layers} 层")
    
except ImportError as e:
    print(f"   ✗ transformers 版本过低，不支持 GemmaConfig")
    print(f"   需要 transformers >= 4.38.0")
    print(f"   错误: {e}")
    print("\n   跳过实际模型测试，但代码实现已完成。")
    print("   升级命令: pip install --upgrade 'transformers>=4.38.0'")
    sys.exit(0)

# ============================================================================
# 测试 4: 创建分割模型
# ============================================================================
print("\n【测试 4】创建分割模型实例...")
print("   配置: Bottom(0-2) + Trunk(3-16) + Top(16-18)")

try:
    # 分割点
    split_point_1 = 3   # Bottom: 层 0-2 (前3层)
    split_point_2 = 16  # Top: 层 16-17 (后2层)
    
    print(f"\n   分割配置:")
    print(f"   - Bottom: 层 0 到 {split_point_1-1} (共 {split_point_1} 层)")
    print(f"   - Trunk:  层 {split_point_1} 到 {split_point_2-1} (共 {split_point_2-split_point_1} 层)")
    print(f"   - Top:    层 {split_point_2} 到 {test_config.num_hidden_layers-1} (共 {test_config.num_hidden_layers-split_point_2} 层)")
    
    # 创建 Bottom 模型
    print(f"\n   创建 Bottom 模型 (前 {split_point_1} 层)...")
    bottom = GemmaBottomModel(test_config, end_layer=split_point_1)
    print(f"   ✓ Bottom 创建成功")
    print(f"      参数量: {bottom.num_parameters():,}")
    print(f"      内存占用: {bottom.memory_footprint_mb():.2f} MB")
    
    # 创建 Trunk 模型
    print(f"\n   创建 Trunk 模型 (层 {split_point_1}-{split_point_2-1})...")
    trunk = GemmaTrunkModel(test_config, start_layer=split_point_1, end_layer=split_point_2)
    print(f"   ✓ Trunk 创建成功")
    print(f"      参数量: {trunk.num_parameters():,}")
    print(f"      内存占用: {trunk.memory_footprint_mb():.2f} MB")
    
    # 创建 Top 模型
    print(f"\n   创建 Top 模型 (后 {test_config.num_hidden_layers-split_point_2} 层)...")
    top = GemmaTopModel(test_config, start_layer=split_point_2)
    print(f"   ✓ Top 创建成功")
    print(f"      参数量: {top.num_parameters():,}")
    print(f"      内存占用: {top.memory_footprint_mb():.2f} MB")
    
    # 总参数量
    total_params = bottom.num_parameters() + trunk.num_parameters() + top.num_parameters()
    print(f"\n   总参数量: {total_params:,}")
    print(f"   总内存占用: {bottom.memory_footprint_mb() + trunk.memory_footprint_mb() + top.memory_footprint_mb():.2f} MB")
    
except Exception as e:
    print(f"   ✗ 模型创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 测试 5: 前向传播测试
# ============================================================================
print("\n【测试 5】测试前向传播...")

try:
    batch_size = 2
    seq_len = 16
    
    # 创建随机输入
    input_ids = torch.randint(0, test_config.vocab_size, (batch_size, seq_len))
    print(f"   输入形状: {input_ids.shape}")
    print(f"   输入范围: [{input_ids.min().item()}, {input_ids.max().item()}]")
    
    # 通过 Bottom
    print("\n   → 通过 Bottom 模型...")
    with torch.no_grad():
        hidden_1 = bottom(input_ids)
    print(f"   ✓ Bottom 输出形状: {hidden_1.shape}")
    print(f"   ✓ 预期形状: [{batch_size}, {seq_len}, {test_config.hidden_size}]")
    assert hidden_1.shape == (batch_size, seq_len, test_config.hidden_size), "Bottom 输出形状不正确"
    
    # 通过 Trunk
    print("\n   → 通过 Trunk 模型...")
    with torch.no_grad():
        hidden_2 = trunk(hidden_1)
    print(f"   ✓ Trunk 输出形状: {hidden_2.shape}")
    print(f"   ✓ 预期形状: [{batch_size}, {seq_len}, {test_config.hidden_size}]")
    assert hidden_2.shape == (batch_size, seq_len, test_config.hidden_size), "Trunk 输出形状不正确"
    
    # 通过 Top
    print("\n   → 通过 Top 模型...")
    with torch.no_grad():
        output = top(hidden_2)
    print(f"   ✓ Top 输出 logits 形状: {output.logits.shape}")
    print(f"   ✓ 预期形状: [{batch_size}, {seq_len}, {test_config.vocab_size}]")
    assert output.logits.shape == (batch_size, seq_len, test_config.vocab_size), "Top 输出形状不正确"
    
    # 检查输出范围
    print(f"\n   输出统计:")
    print(f"   - Logits 最小值: {output.logits.min().item():.4f}")
    print(f"   - Logits 最大值: {output.logits.max().item():.4f}")
    print(f"   - Logits 平均值: {output.logits.mean().item():.4f}")
    print(f"   - Logits 标准差: {output.logits.std().item():.4f}")
    
    # 测试预测
    predicted_ids = output.logits.argmax(dim=-1)
    print(f"\n   预测的 token IDs 形状: {predicted_ids.shape}")
    print(f"   第一个样本的前10个预测: {predicted_ids[0, :10].tolist()}")
    
except Exception as e:
    print(f"   ✗ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 测试 6: 测试 ModelFactory（可选，如果需要预训练模型）
# ============================================================================
print("\n【测试 6】测试 ModelFactory 接口...")
print("   (此测试需要下载模型，可能需要时间和网络连接)")

test_with_pretrained = False  # 设置为 True 以测试预训练模型

if test_with_pretrained:
    try:
        print("\n   尝试使用 ModelFactory 创建 Gemma 2B 分割模型...")
        print("   警告: 需要 Hugging Face 访问权限和良好的网络连接")
        
        bottom_pt, trunk_pt, top_pt = ModelFactory.create_split_models(
            model_type='gemma',
            model_name_or_path='google/gemma-2b',
            split_point_1=3,   # 前3层
            split_point_2=16,  # 后2层
            device='cpu'
        )
        
        print("   ✓ 预训练模型加载成功")
        print(f"   Bottom 参数量: {bottom_pt.num_parameters():,}")
        print(f"   Trunk 参数量: {trunk_pt.num_parameters():,}")
        print(f"   Top 参数量: {top_pt.num_parameters():,}")
        
    except Exception as e:
        print(f"   ⚠ 预训练模型测试跳过: {e}")
        print("   提示: 需要接受 Gemma 许可协议并使用 'huggingface-cli login'")
else:
    print("   ⊗ 跳过预训练模型测试 (test_with_pretrained=False)")
    print("   如需测试预训练模型，请:")
    print("   1. 设置 test_with_pretrained = True")
    print("   2. 确保已接受 Gemma 许可协议: https://huggingface.co/google/gemma-2b")
    print("   3. 登录 Hugging Face: huggingface-cli login")

# ============================================================================
# 测试总结
# ============================================================================
print("\n" + "=" * 70)
print("✅ 所有测试通过!")
print("=" * 70)

print("\n【测试总结】")
print(f"✓ Gemma 模型类成功导入")
print(f"✓ Gemma 在 ModelRegistry 中完整注册")
print(f"✓ 成功创建分割模型: Bottom(前3层) + Trunk(中间) + Top(后2层)")
print(f"✓ 前向传播正常工作")
print(f"✓ 输出形状正确")

print("\n【配置详情】")
print(f"- 模型类型: Gemma (google/gemma-2b 架构)")
print(f"- 总层数: {test_config.num_hidden_layers} 层")
print(f"- Bottom: 层 0-{split_point_1-1} (前 {split_point_1} 层)")
print(f"- Trunk: 层 {split_point_1}-{split_point_2-1} (中间 {split_point_2-split_point_1} 层)")
print(f"- Top: 层 {split_point_2}-{test_config.num_hidden_layers-1} (后 {test_config.num_hidden_layers-split_point_2} 层)")

print("\n【注意事项】")
print("1. 本测试使用随机初始化的小型模型进行验证")
print("2. 实际使用时可以加载预训练的 google/gemma-2b 或 google/gemma-7b")
print("3. 需要 transformers >= 4.38.0 版本")
print("4. 需要在 Hugging Face 上接受 Gemma 许可协议")

print("\n【使用示例】")
print("""
from splitlearn import ModelFactory

# 分割 Gemma 2B: 前3层 + 后2层
bottom, trunk, top = ModelFactory.create_split_models(
    model_type='gemma',
    model_name_or_path='google/gemma-2b',
    split_point_1=3,    # Bottom: 前3层 (0-2)
    split_point_2=16,   # Top: 后2层 (16-17)
    device='cpu'
)

# 使用分割模型
input_ids = tokenizer.encode("Hello", return_tensors="pt")
with torch.no_grad():
    h1 = bottom(input_ids)
    h2 = trunk(h1)
    output = top(h2)
""")

print("=" * 70)

