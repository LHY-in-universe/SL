#!/usr/bin/env python3
"""
在 Anaconda 环境中使用预加载的模型
不使用 ModelFactory，避免死锁

前提：已经运行过 save_split_models.py 创建了模型文件
"""

import os
import sys
import json
from pathlib import Path

# 设置环境变量
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

print("=" * 70)
print("在 Anaconda 中使用预加载的模型")
print("=" * 70)

# 导入基础库（这些不会导致死锁）
import torch
from transformers import AutoTokenizer, GPT2Config

print("\n✓ 基础库导入成功")

# ✅ 只导入需要的模型类（不会死锁）
# 注意：不要 import splitlearn_core 或 from splitlearn_core import ModelFactory
sys.path.insert(0, '/Users/lhy/Desktop/Git/SL/SplitLearnCore/src')

print("\n导入模型类...")
try:
    from splitlearn_core.models.gpt2 import GPT2BottomModel, GPT2TrunkModel, GPT2TopModel
    print("✓ 模型类导入成功（没有死锁！）")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 检查模型文件
models_dir = Path("./models")
model_files = {
    'bottom': models_dir / "bottom" / "gpt2_2-10_bottom.pt",
    'trunk': models_dir / "trunk" / "gpt2_2-10_trunk.pt",
    'top': models_dir / "top" / "gpt2_2-10_top.pt",
}

print("\n[1] 检查模型文件...")
all_exist = True
for name, path in model_files.items():
    if path.exists():
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"  ✓ {name}: {size_mb:.2f} MB")
    else:
        print(f"  ✗ {name}: 文件不存在")
        all_exist = False

if not all_exist:
    print("\n❌ 模型文件缺失！")
    print("\n请先运行以下命令创建模型文件:")
    print("  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 save_split_models.py")
    sys.exit(1)

# 读取元数据
print("\n[2] 读取模型配置...")
metadata_files = {
    'bottom': models_dir / "bottom" / "gpt2_2-10_bottom_metadata.json",
    'trunk': models_dir / "trunk" / "gpt2_2-10_trunk_metadata.json",
    'top': models_dir / "top" / "gpt2_2-10_top_metadata.json",
}

with open(metadata_files['bottom']) as f:
    bottom_meta = json.load(f)
with open(metadata_files['trunk']) as f:
    trunk_meta = json.load(f)
with open(metadata_files['top']) as f:
    top_meta = json.load(f)

print(f"  Bottom: 层 0-{bottom_meta['end_layer']}")
print(f"  Trunk:  层 {trunk_meta['start_layer']}-{trunk_meta['end_layer']}")
print(f"  Top:    层 {top_meta['start_layer']}+")

# 创建模型结构
print("\n[3] 创建模型结构...")
config = GPT2Config.from_pretrained("gpt2")

bottom = GPT2BottomModel(config, end_layer=bottom_meta['end_layer'])
trunk = GPT2TrunkModel(
    config,
    start_layer=trunk_meta['start_layer'],
    end_layer=trunk_meta['end_layer']
)
top = GPT2TopModel(config, start_layer=top_meta['start_layer'])

print("  ✓ 模型结构创建完成")

# 加载权重
print("\n[4] 加载预训练权重...")
bottom.load_state_dict(torch.load(
    model_files['bottom'],
    map_location='cpu',
    weights_only=True
))
trunk.load_state_dict(torch.load(
    model_files['trunk'],
    map_location='cpu',
    weights_only=True
))
top.load_state_dict(torch.load(
    model_files['top'],
    map_location='cpu',
    weights_only=True
))

# 设置为评估模式
bottom.eval()
trunk.eval()
top.eval()

print("  ✓ 权重加载完成")

# 模型信息
bottom_params = sum(p.numel() for p in bottom.parameters()) / 1e6
trunk_params = sum(p.numel() for p in trunk.parameters()) / 1e6
top_params = sum(p.numel() for p in top.parameters()) / 1e6

print(f"\n模型参数:")
print(f"  Bottom: {bottom_params:.2f}M 参数")
print(f"  Trunk:  {trunk_params:.2f}M 参数")
print(f"  Top:    {top_params:.2f}M 参数")

# 测试推理
print("\n[5] 测试推理...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "The future of AI is"
input_ids = tokenizer.encode(text, return_tensors="pt")

print(f"  输入: '{text}'")

with torch.no_grad():
    h1 = bottom(input_ids)
    h2 = trunk(h1)
    output = top(h2)

next_token_id = output.logits[0, -1].argmax().item()
next_token = tokenizer.decode([next_token_id])

print(f"  预测的下一个词: '{next_token}'")
print("  ✓ 推理成功")

# 文本生成
print("\n[6] 生成文本（10 个 tokens）...")
generated_ids = input_ids.clone()
for i in range(10):
    with torch.no_grad():
        h1 = bottom(generated_ids)
        h2 = trunk(h1)
        output = top(h2)

    next_id = output.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    generated_ids = torch.cat([generated_ids, next_id], dim=1)

generated_text = tokenizer.decode(generated_ids[0])
print(f"\n  原始: '{text}'")
print(f"  生成: '{generated_text}'")

# 成功
print("\n" + "=" * 70)
print("✅ 成功！在 Anaconda 中使用预加载的模型")
print("=" * 70)
print("\n说明:")
print("  ✓ 不使用 ModelFactory（避免死锁）")
print("  ✓ 只导入单个模型类（安全）")
print("  ✓ 加载预先保存的权重")
print("  ✓ 功能完全正常！")
print("\n限制:")
print("  ✗ 不能动态改变拆分点（需要在 Framework Python 中重新创建）")
print("  ✗ 不能加载新模型（需要在 Framework Python 中创建）")
print()
