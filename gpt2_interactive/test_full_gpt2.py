#!/usr/bin/env python3
"""
测试完整 gpt2 模型，逐步验证每个操作
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
print('完整 gpt2 模型测试（逐步验证）')
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
    sys.exit(1)
print()

# 测试1: 简单前向传播
print('2. 测试简单前向传播...')
try:
    test_input = 'hello'
    input_ids = tokenizer.encode(test_input, return_tensors='pt')
    with torch.inference_mode():
        outputs = model(input_ids)
    print(f'   ✓ 前向传播成功: {outputs.logits.shape}')
except Exception as e:
    print(f'   ✗ 前向传播失败: {e}')
    sys.exit(1)
print()

# 测试2: 获取 logits 并移到 CPU
print('3. 测试 logits 处理...')
try:
    with torch.inference_mode():
        outputs = model(input_ids)
        logits = outputs.logits
        next_token_logits = logits[0, -1, :].cpu()
    print(f'   ✓ Logits 处理成功: {next_token_logits.shape}')
except Exception as e:
    print(f'   ✗ Logits 处理失败: {e}')
    sys.exit(1)
print()

# 测试3: Top-k 操作
print('4. 测试 Top-k 操作...')
try:
    top_k = 50
    top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, len(next_token_logits)))
    top_k_probs = torch.softmax(top_k_logits, dim=-1)
    print(f'   ✓ Top-k 操作成功: {top_k_probs.shape}')
except Exception as e:
    print(f'   ✗ Top-k 操作失败: {e}')
    sys.exit(1)
print()

# 测试4: Numpy 采样
print('5. 测试 Numpy 采样...')
try:
    sampled_idx = np.random.choice(len(top_k_probs), p=top_k_probs.numpy())
    next_token_id = top_k_indices[sampled_idx].item()
    print(f'   ✓ Numpy 采样成功: token_id={next_token_id}')
except Exception as e:
    print(f'   ✗ Numpy 采样失败: {e}')
    sys.exit(1)
print()

# 测试5: 创建新 tensor 并 concat
print('6. 测试 Tensor 操作...')
try:
    generated_ids = input_ids.clone()
    next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
    generated_ids = torch.cat([generated_ids.clone(), next_token_tensor], dim=1)
    print(f'   ✓ Tensor 操作成功: {generated_ids.shape}')
except Exception as e:
    print(f'   ✗ Tensor 操作失败: {e}')
    sys.exit(1)
print()

# 测试6: 循环生成（逐步）
print('7. 测试循环生成（逐步验证）...')
try:
    generated_ids = input_ids.clone()
    with torch.inference_mode():
        for step in range(5):
            print(f'   步骤 {step+1}...', end=' ', flush=True)
            
            # 前向传播
            outputs = model(generated_ids)
            logits = outputs.logits
            
            # 移到 CPU
            next_token_logits = logits[0, -1, :].cpu()
            
            # Top-k
            top_k_logits, top_k_indices = torch.topk(next_token_logits, min(50, len(next_token_logits)))
            top_k_probs = torch.softmax(top_k_logits, dim=-1)
            
            # 采样
            sampled_idx = np.random.choice(len(top_k_probs), p=top_k_probs.numpy())
            next_token_id = top_k_indices[sampled_idx].item()
            
            # Concat
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
            generated_ids = torch.cat([generated_ids.clone(), next_token_tensor], dim=1)
            
            token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
            print(f'✓ \"{token_text}\"')
    
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f'   ✓ 循环生成成功: \"{full_text}\"')
except Exception as e:
    print(f'\n   ✗ 循环生成失败（步骤 {step+1}）: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print('=' * 60)
print('✓ 完整 gpt2 模型所有测试通过！')
print('=' * 60)
