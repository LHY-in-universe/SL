#!/usr/bin/env python3
"""
Qwen2-VL 文字输入处理演示

说明文字输入的处理流程，以及如何与视觉特征结合。
"""
import torch
from transformers import Qwen2VLProcessor
from PIL import Image
import numpy as np

print('=' * 70)
print('Qwen2-VL 文字输入处理流程')
print('=' * 70)

# 1. 加载 processor
processor = Qwen2VLProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct', cache_dir='./models')

# 2. 准备输入
test_image = Image.fromarray((np.random.rand(448, 448, 3) * 255).astype('uint8'))
test_text = "描述这张图片"

print(f'\n【输入】')
print(f'  文字: "{test_text}"')
print(f'  图像: {test_image.size}')

# 3. 使用 processor 处理
inputs = processor(images=[test_image], text=[test_text], return_tensors="pt")

print(f'\n【Processor 输出】')
for key, value in inputs.items():
    if isinstance(value, torch.Tensor):
        print(f'  {key}: shape={tuple(value.shape)}, dtype={value.dtype}')
    else:
        print(f'  {key}: {value}')

# 4. 文字处理详解
print(f'\n【文字处理步骤】')
print('1. 文字 → Tokenization')
print(f'   输入: "{test_text}"')
print(f'   输出: input_ids = {inputs["input_ids"].tolist()[0]}')

# 解码查看
decoded = processor.tokenizer.decode(inputs["input_ids"][0])
print(f'   解码验证: {decoded}')

print('\n2. input_ids → Embeddings (需要 embed_tokens 层)')
print('   需要从完整模型中获取 model.embed_tokens')
print('   text_embeds = embed_tokens(input_ids)')
print(f'   输出 shape: [batch_size, text_len, hidden_size]')
print(f'   例如: [1, {inputs["input_ids"].shape[1]}, 1536]')

print('\n3. 视觉特征处理')
print('   vision_feats = bottom(pixel_values, grid_thw)')
print('   输出 shape: [num_vision_tokens, hidden_size]')
print('   例如: [256, 1536]')

print('\n4. 拼接视觉和文字特征')
print('   combined = torch.cat([vision_feats, text_embeds], dim=1)')
print('   输出 shape: [batch_size, vision_len + text_len, hidden_size]')
print('   例如: [1, 256 + 3, 1536] = [1, 259, 1536]')

print('\n5. 通过 trunk 和 top 层')
print('   trunk_out = trunk(combined_embeds, attention_mask)')
print('   top_out = top(trunk_out, attention_mask)')
print('   最终输出: logits shape [batch_size, seq_len, vocab_size]')

print('\n' + '=' * 70)
print('【关键点】')
print('=' * 70)
print('1. 文字需要先 tokenize 成 input_ids')
print('2. input_ids 需要通过 embed_tokens 转换为 embeddings')
print('3. 视觉特征和文字 embeddings 需要拼接（维度要匹配）')
print('4. 拼接后的序列通过 trunk → top → logits')
print('5. 注意：Qwen2-VL 中视觉特征的位置可能需要根据具体格式调整')
print('=' * 70)
