#!/usr/bin/env python3
"""
详细测试完整 gpt2 模型
"""

import sys
import os
import time
import torch
import numpy as np
from pathlib import Path

# 添加路径
splitcore_path = Path(__file__).parent.parent / "SplitLearnCore" / "src"
if splitcore_path.exists():
    sys.path.insert(0, str(splitcore_path))

monitor_path = Path(__file__).parent.parent / "SplitLearnMonitor" / "src"
if monitor_path.exists():
    sys.path.insert(0, str(monitor_path))

from splitlearn_core.quickstart import load_full_model

# 强制使用 CPU
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
torch.set_num_threads(1)

print('=' * 60)
print('完整 gpt2 模型详细测试')
print('=' * 60)
print()

# 加载模型
print('1. 加载完整 gpt2 模型...')
try:
    start = time.time()
    model, tokenizer = load_full_model('gpt2', device='cpu', dtype=torch.float32)
    load_time = time.time() - start
    print(f'   ✓ 加载成功，耗时: {load_time:.2f} 秒')
    print(f'   参数数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
except Exception as e:
    print(f'   ✗ 加载失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# 测试生成
print('2. 测试生成（逐步，最多10个token）...')
test_input = 'hello'
input_ids = tokenizer.encode(test_input, return_tensors='pt')
generated_ids = input_ids.clone()

try:
    with torch.inference_mode():
        for step in range(10):
            print(f'   生成 token {step+1}...', end=' ', flush=True)
            
            token_start = time.time()
            
            # 前向传播
            outputs = model(generated_ids)
            logits = outputs.logits
            
            # 移到 CPU
            next_token_logits = logits[0, -1, :].cpu()
            
            # Top-k 采样
            top_k = 50
            top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, len(next_token_logits)))
            top_k_probs = torch.softmax(top_k_logits, dim=-1)
            
            # Numpy 采样
            sampled_idx = np.random.choice(len(top_k_probs), p=top_k_probs.numpy())
            next_token_id = top_k_indices[sampled_idx].item()
            
            if next_token_id == tokenizer.eos_token_id:
                print('EOS')
                break
            
            # Concat
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
            generated_ids = torch.cat([generated_ids.clone(), next_token_tensor], dim=1)
            
            token_time = (time.time() - token_start) * 1000
            token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
            print(f'✓ \"{token_text}\" ({token_time:.1f}ms)')
    
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f'\n   ✓ 生成成功: \"{full_text}\"')
    print(f'   总 token 数: {generated_ids.shape[1]}')
    
except Exception as e:
    print(f'\n   ✗ 生成失败（步骤 {step+1}）: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print('=' * 60)
print('✓ 完整 gpt2 模型测试通过！')
print('=' * 60)
