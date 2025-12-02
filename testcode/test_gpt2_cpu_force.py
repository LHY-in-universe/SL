"""
强制纯 CPU 模式测试 - 解决锁死问题
"""
import os
# 必须在导入 torch 之前设置！
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用 CUDA
# 关键：禁用 MPS
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import sys
import torch

# 确保使用 CPU
device = torch.device('cpu')
print(f"Using device: {device}")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
splitlearn_path = os.path.join(project_root, 'SplitLearning', 'src')
sys.path.insert(0, splitlearn_path)

print("=" * 70)
print("GPT-2 测试 (强制 CPU 模式)")
print("=" * 70)

try:
    from transformers import GPT2Config, GPT2Tokenizer
    from splitlearn.models.gpt2 import GPT2BottomModel, GPT2TrunkModel, GPT2TopModel
    
    bottom_path = os.path.join(current_dir, 'gpt2_bottom_cached.pt')
    trunk_path = os.path.join(current_dir, 'gpt2_trunk_full.pt')
    top_path = os.path.join(current_dir, 'gpt2_top_cached.pt')
    
    print("1. 加载配置...")
    config = GPT2Config.from_pretrained('gpt2')
    
    print("2. 加载模型 (CPU)...")
    # 强制 map_location='cpu'
    bottom = GPT2BottomModel(config, end_layer=3).to(device)
    bottom.load_state_dict(torch.load(bottom_path, map_location=device))
    
    trunk = GPT2TrunkModel(config, start_layer=3, end_layer=10).to(device)
    trunk.load_state_dict(torch.load(trunk_path, map_location=device), strict=False)
    
    top = GPT2TopModel(config, start_layer=10).to(device)
    top.load_state_dict(torch.load(top_path, map_location=device))
    
    print("✓ 模型加载成功！")
    
    print("3. 测试推理...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode("AI is", return_tensors="pt").to(device)
    
    with torch.no_grad():
        h1 = bottom(input_ids)
        h2 = trunk(h1)
        output = top(h2)
        
    print(f"✓ 推理成功: {output.logits.shape}")
    print("✅ 测试通过！")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

强制纯 CPU 模式测试 - 解决锁死问题
"""
import os
# 必须在导入 torch 之前设置！
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用 CUDA
# 关键：禁用 MPS
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import sys
import torch

# 确保使用 CPU
device = torch.device('cpu')
print(f"Using device: {device}")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
splitlearn_path = os.path.join(project_root, 'SplitLearning', 'src')
sys.path.insert(0, splitlearn_path)

print("=" * 70)
print("GPT-2 测试 (强制 CPU 模式)")
print("=" * 70)

try:
    from transformers import GPT2Config, GPT2Tokenizer
    from splitlearn.models.gpt2 import GPT2BottomModel, GPT2TrunkModel, GPT2TopModel
    
    bottom_path = os.path.join(current_dir, 'gpt2_bottom_cached.pt')
    trunk_path = os.path.join(current_dir, 'gpt2_trunk_full.pt')
    top_path = os.path.join(current_dir, 'gpt2_top_cached.pt')
    
    print("1. 加载配置...")
    config = GPT2Config.from_pretrained('gpt2')
    
    print("2. 加载模型 (CPU)...")
    # 强制 map_location='cpu'
    bottom = GPT2BottomModel(config, end_layer=3).to(device)
    bottom.load_state_dict(torch.load(bottom_path, map_location=device))
    
    trunk = GPT2TrunkModel(config, start_layer=3, end_layer=10).to(device)
    trunk.load_state_dict(torch.load(trunk_path, map_location=device), strict=False)
    
    top = GPT2TopModel(config, start_layer=10).to(device)
    top.load_state_dict(torch.load(top_path, map_location=device))
    
    print("✓ 模型加载成功！")
    
    print("3. 测试推理...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode("AI is", return_tensors="pt").to(device)
    
    with torch.no_grad():
        h1 = bottom(input_ids)
        h2 = trunk(h1)
        output = top(h2)
        
    print(f"✓ 推理成功: {output.logits.shape}")
    print("✅ 测试通过！")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()


