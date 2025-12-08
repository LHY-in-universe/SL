#!/usr/bin/python3
"""
简单的 GPT-2 交互式对话脚本
使用 Hugging Face transformers 加载模型，记录交互时间
"""

import time
import torch
import os
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 强制使用 CPU，避免 MPS 相关的 Bus error
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
device = torch.device('cpu')  # 强制使用 CPU 以避免 Bus error

def load_model():
    """加载 GPT-2 模型和分词器"""
    print("正在加载 GPT-2 模型...")
    start_time = time.time()
    
    try:
        # 加载预训练的 GPT-2 模型和分词器
        model_name = "gpt2"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # 设置 pad_token（GPT-2 默认没有 pad_token）
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 设置模型为评估模式并移动到 CPU
        model.eval()
        model = model.to(device)
        
        load_time = time.time() - start_time
        print(f"模型加载完成，耗时: {load_time:.2f} 秒")
        print(f"使用设备: {device}\n")
        
        return model, tokenizer
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise

def generate_response(model, tokenizer, user_input, max_length=100):
    """生成回复，返回回复和时间统计"""
    # 编码输入，并创建 attention_mask
    encode_start = time.time()
    encoded = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    encode_time = time.time() - encode_start
    
    # 模型推理（只计算模型运行时间）
    inference_start = time.time()
    try:
        with torch.no_grad():
            # 传递 attention_mask 以避免警告和错误
            generate_kwargs = {
                "max_length": max_length,
                "num_return_sequences": 1,
                "temperature": 0.7,
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id,
            }
            
            # 如果有 attention_mask，则传递它
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask
            
            output = model.generate(input_ids, **generate_kwargs)
    except Exception as e:
        print(f"生成错误: {e}")
        raise
    finally:
        inference_time = time.time() - inference_start
    
    # 解码输出
    decode_start = time.time()
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    decode_time = time.time() - decode_start
    
    # 移除原始输入部分，只返回生成的部分
    if response.startswith(user_input):
        response = response[len(user_input):].strip()
    
    # 返回回复和时间统计
    time_stats = {
        'encode_time': encode_time,
        'inference_time': inference_time,  # 这是模型实际运行时间
        'decode_time': decode_time,
        'total_time': encode_time + inference_time + decode_time
    }
    
    return response, time_stats

def main():
    """主函数"""
    print("=" * 60)
    print("GPT-2 交互式对话")
    print("=" * 60)
    print("输入 'quit' 或 'exit' 退出程序\n")
    
    # 加载模型
    model, tokenizer = load_model()
    
    # 交互循环
    session_start = time.time()
    interaction_count = 0
    total_inference_time = 0.0  # 累计模型推理时间
    
    while True:
        try:
            # 获取用户输入
            user_input = input("你: ").strip()
            
            if not user_input:
                continue
            
            # 检查退出命令
            if user_input.lower() in ['quit', 'exit', '退出']:
                break
            
            # 记录交互开始时间
            interaction_start = time.time()
            interaction_count += 1
            
            # 生成回复
            response, time_stats = generate_response(model, tokenizer, user_input)
            
            # 累计模型推理时间
            total_inference_time += time_stats['inference_time']
            
            # 计算总交互时间（包括输入等待时间）
            interaction_time = time.time() - interaction_start
            
            # 显示回复和时间
            print(f"GPT-2: {response}")
            print(f"[交互 #{interaction_count} 时间统计:]")
            print(f"  编码时间: {time_stats['encode_time']*1000:.2f} ms")
            print(f"  模型推理时间: {time_stats['inference_time']*1000:.2f} ms ⭐")
            print(f"  解码时间: {time_stats['decode_time']*1000:.2f} ms")
            print(f"  总处理时间: {time_stats['total_time']*1000:.2f} ms")
            print()
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"错误: {e}")
            continue
    
    # 显示会话统计
    session_time = time.time() - session_start
    print("\n" + "=" * 60)
    print("会话统计:")
    print(f"  总交互次数: {interaction_count}")
    print(f"  总会话时间: {session_time:.2f} 秒")
    if interaction_count > 0:
        print(f"  平均交互时间: {session_time / interaction_count:.3f} 秒")
        print(f"  累计模型推理时间: {total_inference_time:.3f} 秒")
        print(f"  平均模型推理时间: {total_inference_time / interaction_count:.3f} 秒")
        print(f"  模型推理时间占比: {total_inference_time / session_time * 100:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()

