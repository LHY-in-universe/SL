"""
直接使用 Qwen2.5-3B 进行拆分 - 前3层+后2层
只加载需要的层，不加载整个模型
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
splitlearn_path = os.path.join(project_root, 'SplitLearning', 'src')
sys.path.insert(0, splitlearn_path)

print("=" * 70)
print("Qwen2.5-3B 拆分测试 - 前3层 + 后2层")
print("=" * 70)
print("\n提示: 首次运行会下载模型 (~3.1GB)，请耐心等待...")
print("优势: Qwen2.5 无需 Hugging Face 权限，开源可用！\n")

try:
    from splitlearn import ModelFactory
    from transformers import AutoTokenizer
    import torch
    
    # Qwen2.5-3B: 28层，拆分为 前3层 + 中间23层 + 后2层
    print("【1】加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B')
    print("   ✓ 分词器加载完成\n")
    
    print("【2】创建分割模型...")
    print("   配置:")
    print("   - 模型: Qwen2.5-3B (28层)")
    print("   - Bottom: 前3层 (层0-2)")
    print("   - Trunk: 中间23层 (层3-25)")
    print("   - Top: 后2层 (层26-27)")
    print("\n   开始加载...")
    
    # 使用 ModelFactory 创建 - 它会智能地只加载需要的层
    bottom, trunk, top = ModelFactory.create_split_models(
        model_type='qwen2',
        model_name_or_path='Qwen/Qwen2.5-3B',
        split_point_1=3,    # Bottom 前3层
        split_point_2=26,   # Top 后2层
        device='cpu'
    )
    
    print("\n   ✓ 模型创建成功！\n")
    
    # 测试
    print("【3】测试推理...")
    test_text = "人工智能的未来是"
    print(f"   输入: {test_text}")
    
    input_ids = tokenizer.encode(test_text, return_tensors="pt")
    
    with torch.no_grad():
        h1 = bottom(input_ids)
        print(f"   ✓ Bottom 输出: {h1.shape}")
        
        h2 = trunk(h1)
        print(f"   ✓ Trunk 输出: {h2.shape}")
        
        output = top(h2)
        print(f"   ✓ Top 输出: {output.logits.shape}")
    
    # 预测下一个token
    next_token_id = output.logits[0, -1].argmax().item()
    next_token = tokenizer.decode([next_token_id])
    print(f"\n   预测下一个词: '{next_token}'")
    
    # 生成更多文本
    print("\n【4】生成文本...")
    generated_ids = input_ids.clone()
    
    for i in range(10):
        with torch.no_grad():
            h1 = bottom(generated_ids)
            h2 = trunk(h1)
            output = top(h2)
        
        next_id = output.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        generated_ids = torch.cat([generated_ids, next_id], dim=1)
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"   生成结果: {generated_text}")
    
    print("\n" + "=" * 70)
    print("✅ 测试成功！")
    print("=" * 70)
    
    print("\n【总结】")
    print("  ✓ Qwen2.5-3B 拆分成功")
    print("  ✓ 前3层 + 后2层配置工作正常")
    print("  ✓ 文本生成功能正常")
    print("  ✓ 无需 Hugging Face 权限")
    
except KeyboardInterrupt:
    print("\n\n⚠️ 用户中断")
    sys.exit(0)
    
except Exception as e:
    print(f"\n✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

