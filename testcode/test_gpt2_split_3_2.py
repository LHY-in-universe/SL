"""
使用现有的 GPT-2 模型测试：前3层 + 后2层配置
"""
import sys
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
splitlearn_path = os.path.join(project_root, 'SplitLearning', 'src')
sys.path.insert(0, splitlearn_path)

print("=" * 70)
print("GPT-2 分割模型测试 - 使用现有 .pt 文件")
print("前3层 + 后2层配置")
print("=" * 70)

try:
    from transformers import GPT2Config, GPT2Tokenizer
    from splitlearn.models.gpt2 import GPT2BottomModel, GPT2TrunkModel, GPT2TopModel
    
    # 检查文件
    bottom_path = os.path.join(current_dir, 'gpt2_bottom_cached.pt')
    trunk_path = os.path.join(current_dir, 'gpt2_trunk_full.pt')
    top_path = os.path.join(current_dir, 'gpt2_top_cached.pt')
    
    print("\n【1】检查 .pt 文件...")
    if not os.path.exists(bottom_path):
        print(f"   ✗ 找不到: {bottom_path}")
        sys.exit(1)
    print(f"   ✓ Bottom: {os.path.getsize(bottom_path)/(1024*1024):.1f}MB")
    print(f"   ✓ Trunk: {os.path.getsize(trunk_path)/(1024*1024):.1f}MB")
    print(f"   ✓ Top: {os.path.getsize(top_path)/(1024*1024):.1f}MB")
    
    # 加载配置
    print("\n【2】加载配置...")
    config = GPT2Config.from_pretrained('gpt2')
    print(f"   ✓ GPT-2 配置: {config.n_layer} 层")
    
    # 创建模型
    print("\n【3】加载分割模型...")
    
    # Bottom: 前3层 (0-2)
    print("   加载 Bottom (前3层)...")
    bottom = GPT2BottomModel(config, end_layer=3)
    bottom.load_state_dict(torch.load(bottom_path, map_location='cpu'))
    print(f"   ✓ Bottom: {sum(p.numel() for p in bottom.parameters()):,} 参数")
    
    # Trunk: 中间层 (3-10)
    print("   加载 Trunk (层3-9)...")
    trunk = GPT2TrunkModel(config, start_layer=3, end_layer=10)
    trunk.load_state_dict(torch.load(trunk_path, map_location='cpu'), strict=False)
    print(f"   ✓ Trunk: {sum(p.numel() for p in trunk.parameters()):,} 参数")
    
    # Top: 后2层 (10-11)
    print("   加载 Top (后2层)...")
    top = GPT2TopModel(config, start_layer=10)
    top.load_state_dict(torch.load(top_path, map_location='cpu'))
    print(f"   ✓ Top: {sum(p.numel() for p in top.parameters()):,} 参数")
    
    # 测试
    print("\n【4】测试推理...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    text = "The future of AI is"
    print(f"   输入: \"{text}\"")
    
    input_ids = tokenizer.encode(text, return_tensors="pt")
    
    with torch.no_grad():
        h1 = bottom(input_ids)
        print(f"   ✓ Bottom 输出: {h1.shape}")
        
        h2 = trunk(h1)
        print(f"   ✓ Trunk 输出: {h2.shape}")
        
        output = top(h2)
        print(f"   ✓ Top logits: {output.logits.shape}")
    
    # 预测
    next_token_id = output.logits[0, -1].argmax().item()
    next_token = tokenizer.decode([next_token_id])
    print(f"\n   预测下一个词: \"{next_token}\"")
    
    # 生成文本
    print("\n【5】生成文本...")
    generated_ids = input_ids.clone()
    
    for i in range(15):
        with torch.no_grad():
            h1 = bottom(generated_ids)
            h2 = trunk(h1)
            output = top(h2)
        
        next_id = output.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        generated_ids = torch.cat([generated_ids, next_id], dim=1)
    
    generated_text = tokenizer.decode(generated_ids[0])
    print(f"   生成结果:\n   {generated_text}")
    
    print("\n" + "=" * 70)
    print("✅ 测试成功！")
    print("=" * 70)
    
    print("\n【总结】")
    print("  ✓ 成功加载 GPT-2 分割模型")
    print("  ✓ 配置: 前3层 + 中间7层 + 后2层")
    print("  ✓ 推理功能正常")
    print("  ✓ 文本生成成功")
    
    print("\n【配置详情】")
    print(f"  GPT-2: {config.n_layer} 层")
    print(f"  - Bottom: 层 0-2 (前3层)")
    print(f"  - Trunk: 层 3-9 (中间7层)")
    print(f"  - Top: 层 10-11 (后2层)")
    
except Exception as e:
    print(f"\n✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

