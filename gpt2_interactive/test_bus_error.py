#!/usr/bin/env python3
import sys
import os
import torch
from pathlib import Path

# 添加 SplitLearnCore 到路径
splitcore_path = Path(__file__).parent.parent / "SplitLearnCore" / "src"
if splitcore_path.exists():
    sys.path.insert(0, str(splitcore_path))

from splitlearn_core.quickstart import load_full_model

# 强制使用 CPU
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)

print("测试 Bus error 原因...")
print("=" * 60)

# 加载模型
model, tokenizer = load_full_model('gpt2', device='cpu', dtype=torch.float32)
print("✓ 模型加载成功")

# 测试1: 简单前向传播
print("\n测试1: 简单前向传播")
try:
    inputs = tokenizer("hello", return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
    print(f"✓ 成功: {outputs.logits.shape}")
except Exception as e:
    print(f"✗ 失败: {e}")

# 测试2: 循环生成（模拟实际使用）
print("\n测试2: 循环生成（1步）")
try:
    input_ids = tokenizer.encode("hello", return_tensors="pt")
    generated_ids = input_ids.clone()
    
    with torch.inference_mode():
        outputs = model(generated_ids)
        logits = outputs.logits
        next_token_logits = logits[0, -1, :].cpu()
        next_token_id = next_token_logits.argmax().item()
        next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
        generated_ids = torch.cat([generated_ids, next_token_tensor], dim=1)
    
    print(f"✓ 成功: {generated_ids.shape}")
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试3: 多次循环
print("\n测试3: 多次循环生成（5步）")
try:
    input_ids = tokenizer.encode("hello", return_tensors="pt")
    generated_ids = input_ids.clone()
    
    with torch.inference_mode():
        for step in range(5):
            outputs = model(generated_ids)
            logits = outputs.logits
            next_token_logits = logits[0, -1, :].cpu()
            next_token_id = next_token_logits.argmax().item()
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
            generated_ids = torch.cat([generated_ids, next_token_tensor], dim=1)
            print(f"  步骤 {step+1}: {generated_ids.shape}")
    
    print(f"✓ 成功: 最终形状 {generated_ids.shape}")
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
