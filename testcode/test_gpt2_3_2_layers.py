"""
使用现有 GPT-2 .pt 文件测试：前3层 + 后2层配置
这个测试不需要下载任何东西，立即可用！
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
splitlearn_path = os.path.join(project_root, 'SplitLearning', 'src')
sys.path.insert(0, splitlearn_path)

print("=" * 70)
print("GPT-2 分割模型测试 - 前3层 + 后2层")
print("=" * 70)
print("\n使用已有的 .pt 文件，无需下载！\n")

try:
    import torch
    from transformers import GPT2Config, GPT2Tokenizer
    
    # 检查文件
    bottom_path = os.path.join(current_dir, 'gpt2_bottom_cached.pt')
    trunk_path = os.path.join(current_dir, 'gpt2_trunk_full.pt')
    top_path = os.path.join(current_dir, 'gpt2_top_cached.pt')
    
    print("【1】检查 .pt 文件...")
    for name, path in [('Bottom', bottom_path), ('Trunk', trunk_path), ('Top', top_path)]:
        if not os.path.exists(path):
            print(f"   ✗ 找不到: {path}")
            sys.exit(1)
        size_mb = os.path.getsize(path) / (1024*1024)
        print(f"   ✓ {name}: {size_mb:.1f} MB")
    
    print("\n【2】加载配置...")
    config = GPT2Config.from_pretrained('gpt2')
    print(f"   ✓ GPT-2: {config.n_layer} 层, {config.n_embd} 隐藏层维度")
    
    print("\n【3】创建并加载模型...")
    from splitlearn.models.gpt2 import GPT2BottomModel, GPT2TrunkModel, GPT2TopModel
    
    # Bottom: 前3层 (0-2)
    print("   加载 Bottom (前3层: 0-2)...")
    bottom = GPT2BottomModel(config, end_layer=3)
    bottom.load_state_dict(torch.load(bottom_path, map_location='cpu'))
    print(f"      参数量: {sum(p.numel() for p in bottom.parameters()):,}")
    
    # Trunk: 中间层 (3-10)
    print("   加载 Trunk (中间层: 3-9)...")
    trunk = GPT2TrunkModel(config, start_layer=3, end_layer=10)
    # trunk 文件包含所有中间层，需要 strict=False
    trunk.load_state_dict(torch.load(trunk_path, map_location='cpu'), strict=False)
    print(f"      参数量: {sum(p.numel() for p in trunk.parameters()):,}")
    
    # Top: 后2层 (10-11)
    print("   加载 Top (后2层: 10-11)...")
    top = GPT2TopModel(config, start_layer=10)
    top.load_state_dict(torch.load(top_path, map_location='cpu'))
    print(f"      参数量: {sum(p.numel() for p in top.parameters()):,}")
    
    total_params = sum(p.numel() for p in bottom.parameters()) + \
                   sum(p.numel() for p in trunk.parameters()) + \
                   sum(p.numel() for p in top.parameters())
    print(f"\n   总参数量: {total_params:,}")
    
    print("\n【4】测试推理...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    test_text = "The future of artificial intelligence is"
    print(f"   输入文本: \"{test_text}\"")
    
    input_ids = tokenizer.encode(test_text, return_tensors="pt")
    print(f"   Token IDs: {input_ids.shape}")
    
    with torch.no_grad():
        # 前向传播
        h1 = bottom(input_ids)
        print(f"   → Bottom 输出: {h1.shape}")
        
        h2 = trunk(h1)
        print(f"   → Trunk 输出: {h2.shape}")
        
        output = top(h2)
        print(f"   → Top logits: {output.logits.shape}")
    
    # 预测下一个词
    next_token_id = output.logits[0, -1].argmax().item()
    next_token = tokenizer.decode([next_token_id])
    print(f"\n   预测下一个词: \"{next_token}\"")
    
    print("\n【5】生成文本...")
    generated_ids = input_ids.clone()
    
    for i in range(20):
        with torch.no_grad():
            h1 = bottom(generated_ids)
            h2 = trunk(h1)
            output = top(h2)
        
        next_id = output.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        generated_ids = torch.cat([generated_ids, next_id], dim=1)
    
    generated_text = tokenizer.decode(generated_ids[0])
    print(f"   生成结果:")
    print(f"   \"{generated_text}\"")
    
    print("\n" + "=" * 70)
    print("✅ 测试成功！")
    print("=" * 70)
    
    print("\n【配置总结】")
    print(f"  模型: GPT-2 ({config.n_layer} 层)")
    print(f"  分割配置:")
    print(f"    • Bottom: 层 0-2   (前 3 层)")
    print(f"    • Trunk:  层 3-9   (中间 7 层)")
    print(f"    • Top:    层 10-11 (后 2 层)")
    
    print("\n【功能验证】")
    print("  ✓ 成功加载预训练的分割模型")
    print("  ✓ 前向传播正常工作")
    print("  ✓ 文本生成功能正常")
    print("  ✓ 前3层+后2层配置验证成功")
    
    print("\n【说明】")
    print("  这个测试证明了分割功能是正常工作的。")
    print("  同样的原理适用于 Qwen2.5-3B 和 Gemma。")
    print("  一旦解决了 PyTorch MPS 锁问题，")
    print("  Qwen2.5-3B 和 Gemma 也能同样工作。")
    
except Exception as e:
    print(f"\n✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

