#!/usr/bin/env python3
"""
逐步测试完整 gpt2 模型，精确定位 Bus error
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
print('完整 gpt2 模型逐步测试')
print('=' * 60)
print()

# 加载模型
print('1. 加载模型...')
model, tokenizer = load_full_model('gpt2', device='cpu', dtype=torch.float32)
print(f'   ✓ 加载成功\n')

# 准备输入
test_input = 'hello'
input_ids = tokenizer.encode(test_input, return_tensors='pt')
generated_ids = input_ids.clone()
print(f'2. 输入: "{test_input}"')
print(f'   输入 shape: {input_ids.shape}\n')

# 逐步测试每个操作
print('3. 逐步测试每个操作...\n')

try:
    with torch.inference_mode():
        print('   3.1 前向传播...', end=' ', flush=True)
        outputs = model(generated_ids)
        logits = outputs.logits
        print(f'✓ {logits.shape}')
        
        print('   3.2 获取最后一个 token logits...', end=' ', flush=True)
        next_token_logits = logits[0, -1, :]
        print(f'✓ {next_token_logits.shape}')
        
        print('   3.3 移到 CPU...', end=' ', flush=True)
        next_token_logits_cpu = next_token_logits.cpu()
        print(f'✓ {next_token_logits_cpu.shape}')
        
        print('   3.4 Top-k 操作...', end=' ', flush=True)
        top_k = 50
        top_k_logits, top_k_indices = torch.topk(next_token_logits_cpu, min(top_k, len(next_token_logits_cpu)))
        print(f'✓ {top_k_logits.shape}')
        
        print('   3.5 Softmax...', end=' ', flush=True)
        top_k_probs = torch.softmax(top_k_logits, dim=-1)
        print(f'✓ {top_k_probs.shape}')
        
        print('   3.6 转换为 numpy...', end=' ', flush=True)
        top_k_probs_np = top_k_probs.numpy()
        print(f'✓ {top_k_probs_np.shape}')
        
        print('   3.7 Numpy 采样...', end=' ', flush=True)
        sampled_idx = np.random.choice(len(top_k_probs_np), p=top_k_probs_np)
        next_token_id = top_k_indices[sampled_idx].item()
        print(f'✓ token_id={next_token_id}')
        
        print('   3.8 创建新 tensor...', end=' ', flush=True)
        next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
        print(f'✓ {next_token_tensor.shape}')
        
        print('   3.9 Clone generated_ids...', end=' ', flush=True)
        generated_ids_clone = generated_ids.clone()
        print(f'✓ {generated_ids_clone.shape}')
        
        print('   3.10 Concat...', end=' ', flush=True)
        generated_ids = torch.cat([generated_ids_clone, next_token_tensor], dim=1)
        print(f'✓ {generated_ids.shape}')
        
        print('   3.11 解码...', end=' ', flush=True)
        token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
        print(f'✓ "{token_text}"')
        
        print('\n   ✓ 第一个 token 生成成功！')
        print(f'   当前序列: {tokenizer.decode(generated_ids[0], skip_special_tokens=True)}')
        
        # 尝试生成第二个 token
        print('\n4. 尝试生成第二个 token...\n')
        
        print('   4.1 前向传播（第二次）...', end=' ', flush=True)
        outputs2 = model(generated_ids)
        logits2 = outputs2.logits
        print(f'✓ {logits2.shape}')
        
        print('   4.2 获取 logits 并移到 CPU...', end=' ', flush=True)
        next_token_logits2 = logits2[0, -1, :].cpu()
        print(f'✓ {next_token_logits2.shape}')
        
        print('   4.3 Top-k...', end=' ', flush=True)
        top_k_logits2, top_k_indices2 = torch.topk(next_token_logits2, min(50, len(next_token_logits2)))
        top_k_probs2 = torch.softmax(top_k_logits2, dim=-1)
        print(f'✓')
        
        print('   4.4 Numpy 采样...', end=' ', flush=True)
        sampled_idx2 = np.random.choice(len(top_k_probs2.numpy()), p=top_k_probs2.numpy())
        next_token_id2 = top_k_indices2[sampled_idx2].item()
        print(f'✓ token_id={next_token_id2}')
        
        print('   4.5 Concat（第二次）...', end=' ', flush=True)
        next_token_tensor2 = torch.tensor([[next_token_id2]], dtype=torch.long)
        generated_ids = torch.cat([generated_ids.clone(), next_token_tensor2], dim=1)
        print(f'✓ {generated_ids.shape}')
        
        token_text2 = tokenizer.decode([next_token_id2], skip_special_tokens=True)
        print(f'\n   ✓ 第二个 token 生成成功: "{token_text2}"')
        print(f'   当前序列: {tokenizer.decode(generated_ids[0], skip_special_tokens=True)}')
        
        print('\n' + '=' * 60)
        print('✓ 完整 gpt2 模型测试通过（2个token）！')
        print('=' * 60)
        
except Exception as e:
    print(f'\n   ✗ 失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
