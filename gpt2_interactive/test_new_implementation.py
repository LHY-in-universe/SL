#!/usr/bin/env python3
"""
测试新的 load_full_model 实现
"""

import sys
from pathlib import Path

# 添加 SplitLearnCore 到路径
splitcore_path = Path(__file__).parent.parent / "SplitLearnCore" / "src"
if splitcore_path.exists():
    sys.path.insert(0, str(splitcore_path))

from splitlearn_core.quickstart import load_full_model
import torch

print("测试 load_full_model 函数...")
print("=" * 60)

# 使用小模型进行快速测试
model, tokenizer = load_full_model(
    model_name_or_path="sshleifer/tiny-gpt2",
    device="cpu",
    dtype=torch.float32,
    low_cpu_mem_usage=True
)

print("\n✓ 模型加载成功！")
print(f"模型类型: {type(model).__name__}")
print(f"分词器类型: {type(tokenizer).__name__}")

# 测试简单推理
test_text = "Hello"
inputs = tokenizer(test_text, return_tensors="pt")
with torch.inference_mode():
    outputs = model(**inputs)

print(f"\n✓ 推理测试成功！")
print(f"输入: '{test_text}'")
print(f"输出 logits 形状: {outputs.logits.shape}")

print("\n" + "=" * 60)
print("所有测试通过！新的实现工作正常。")
print("=" * 60)

