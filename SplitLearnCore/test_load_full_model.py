#!/usr/bin/env python3
"""
测试 load_full_model 函数的详细测试脚本

测试逻辑说明：
1. 测试模型加载功能
2. 测试分词器功能
3. 测试模型推理功能
4. 测试时间统计
5. 测试错误处理
"""

import sys
import os
import time
import torch
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from splitlearn_core.quickstart import load_full_model


def test_basic_loading():
    """测试1: 基本加载功能"""
    print("=" * 60)
    print("测试1: 基本模型加载")
    print("=" * 60)
    
    start_time = time.time()
    
    # 调用 load_full_model 函数
    model, tokenizer = load_full_model(
        model_name_or_path="sshleifer/tiny-gpt2",  # 使用小模型避免下载时间过长
        device="cpu",
        low_cpu_mem_usage=True
    )
    
    load_time = time.time() - start_time
    
    print(f"✓ 模型加载成功")
    print(f"  模型类型: {type(model).__name__}")
    print(f"  分词器类型: {type(tokenizer).__name__}")
    print(f"  加载耗时: {load_time:.2f} 秒")
    print(f"  模型设备: {next(model.parameters()).device}")
    print(f"  模型参数数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print()
    
    return model, tokenizer


def test_tokenizer_functionality(model, tokenizer):
    """测试2: 分词器功能"""
    print("=" * 60)
    print("测试2: 分词器功能")
    print("=" * 60)
    
    test_text = "Hello, how are you?"
    
    # 测试编码
    encoded = tokenizer(test_text, return_tensors="pt")
    print(f"✓ 编码测试")
    print(f"  输入文本: '{test_text}'")
    print(f"  编码后形状: {encoded['input_ids'].shape}")
    print(f"  Token IDs: {encoded['input_ids'].tolist()[0][:10]}...")  # 只显示前10个
    
    # 测试解码
    decoded = tokenizer.decode(encoded['input_ids'][0])
    print(f"  解码后文本: '{decoded}'")
    print(f"  编码-解码一致性: {decoded.strip() == test_text}")
    print()
    
    return encoded


def test_model_inference(model, tokenizer, encoded):
    """测试3: 模型推理功能"""
    print("=" * 60)
    print("测试3: 模型推理")
    print("=" * 60)
    
    # 将输入移到模型设备
    input_ids = encoded['input_ids'].to(next(model.parameters()).device)
    
    # 推理
    inference_start = time.time()
    with torch.inference_mode():
        outputs = model(input_ids)
    inference_time = time.time() - inference_start
    
    print(f"✓ 推理成功")
    print(f"  输入形状: {input_ids.shape}")
    print(f"  输出 logits 形状: {outputs.logits.shape}")
    print(f"  推理耗时: {inference_time*1000:.2f} ms")
    
    # 检查输出合理性
    vocab_size = outputs.logits.shape[-1]
    print(f"  词汇表大小: {vocab_size}")
    print(f"  输出范围: [{outputs.logits.min().item():.2f}, {outputs.logits.max().item():.2f}]")
    print()
    
    return outputs


def test_generation(model, tokenizer):
    """测试4: 文本生成功能"""
    print("=" * 60)
    print("测试4: 简单文本生成")
    print("=" * 60)
    
    prompt = "The quick brown"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
        next(model.parameters()).device
    )
    
    # 简单生成（只生成一个token）
    with torch.inference_mode():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[0, -1, :]
        next_token_id = next_token_logits.argmax().item()
    
    generated_token = tokenizer.decode([next_token_id])
    full_text = prompt + generated_token
    
    print(f"✓ 生成测试")
    print(f"  提示词: '{prompt}'")
    print(f"  生成的下一个token: '{generated_token}'")
    print(f"  完整文本: '{full_text}'")
    print()
    
    return full_text


def test_error_handling():
    """测试5: 错误处理"""
    print("=" * 60)
    print("测试5: 错误处理")
    print("=" * 60)
    
    try:
        # 尝试加载不存在的模型
        model, tokenizer = load_full_model(
            model_name_or_path="non-existent-model-12345",
            device="cpu"
        )
        print("✗ 错误: 应该抛出异常但没有")
    except Exception as e:
        print(f"✓ 错误处理正常")
        print(f"  捕获的异常类型: {type(e).__name__}")
        print(f"  错误信息: {str(e)[:100]}...")
    print()


def test_cache_functionality():
    """测试6: 缓存功能"""
    print("=" * 60)
    print("测试6: 缓存功能")
    print("=" * 60)
    
    # 第一次加载（会下载）
    print("第一次加载（可能下载）...")
    start1 = time.time()
    model1, tokenizer1 = load_full_model(
        model_name_or_path="sshleifer/tiny-gpt2",
        device="cpu"
    )
    time1 = time.time() - start1
    
    # 第二次加载（应该从缓存）
    print("第二次加载（应该从缓存）...")
    start2 = time.time()
    model2, tokenizer2 = load_full_model(
        model_name_or_path="sshleifer/tiny-gpt2",
        device="cpu"
    )
    time2 = time.time() - start2
    
    print(f"✓ 缓存测试")
    print(f"  第一次加载耗时: {time1:.2f} 秒")
    print(f"  第二次加载耗时: {time2:.2f} 秒")
    print(f"  缓存加速: {time1/time2:.2f}x" if time2 > 0 else "  缓存加速: N/A")
    print()


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("SplitLearnCore - load_full_model 功能测试")
    print("=" * 60 + "\n")
    
    try:
        # 测试1: 基本加载
        model, tokenizer = test_basic_loading()
        
        # 测试2: 分词器
        encoded = test_tokenizer_functionality(model, tokenizer)
        
        # 测试3: 模型推理
        outputs = test_model_inference(model, tokenizer, encoded)
        
        # 测试4: 文本生成
        test_generation(model, tokenizer)
        
        # 测试5: 错误处理
        test_error_handling()
        
        # 测试6: 缓存功能
        test_cache_functionality()
        
        print("=" * 60)
        print("✓ 所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

