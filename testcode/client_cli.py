"""
简单的终端交互 GPT-2 客户端
客户端运行 Bottom + Top，服务端运行 Trunk
"""
import torch
import sys
import os
import time

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearning', 'src'))
sys.path.insert(0, os.path.join(project_root, 'splitlearn-comm', 'src'))

from transformers import AutoTokenizer
from splitlearn_comm import GRPCComputeClient

# ================= 配置 =================
SERVER_IP = '192.168.0.144'
SERVER_PORT = 50053
MODEL_ID = 'gpt2-trunk'
# ========================================

def main():
    print("=" * 60)
    print("GPT-2 Split Learning 终端客户端")
    print("=" * 60)
    
    # 1. 加载本地模型
    print("\n[1/3] 加载本地模型...")
    bottom_path = os.path.join(current_dir, "gpt2_bottom_cached.pt")
    top_path = os.path.join(current_dir, "gpt2_top_cached.pt")
    
    if not os.path.exists(bottom_path) or not os.path.exists(top_path):
        print("❌ 模型文件不存在！请先运行: python prepare_models.py")
        return
    
    bottom_model = torch.load(bottom_path, map_location='cpu', weights_only=False)
    top_model = torch.load(top_path, map_location='cpu', weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    print("✅ 本地模型加载完成 (Bottom + Top)")
    
    # 2. 连接服务器
    print(f"\n[2/3] 连接服务器 {SERVER_IP}:{SERVER_PORT}...")
    client = GRPCComputeClient(f"{SERVER_IP}:{SERVER_PORT}", timeout=20.0)
    
    if not client.connect():
        print("❌ 无法连接服务器！")
        return
    print("✅ 服务器连接成功 (Trunk)")
    
    # 3. 交互循环
    print("\n[3/3] 准备就绪！")
    print("=" * 60)
    print("输入 prompt 开始生成，输入 'quit' 退出")
    print("可选参数: <prompt> --length=30 --temp=1.0")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n>>> ").strip()
            
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                print("再见！")
                break
            
            # 解析参数
            max_length = 30
            temperature = 1.0
            prompt = user_input
            
            if '--length=' in user_input:
                parts = user_input.split('--length=')
                prompt = parts[0].strip()
                try:
                    length_part = parts[1].split()[0]
                    max_length = int(length_part)
                except:
                    pass
                    
            if '--temp=' in user_input:
                parts = user_input.split('--temp=')
                if '--length=' not in parts[0]:
                    prompt = parts[0].strip()
                try:
                    temp_part = parts[1].split()[0]
                    temperature = float(temp_part)
                except:
                    pass
            
            # 生成文本
            print(f"\n生成中 (length={max_length}, temp={temperature})...")
            print("-" * 40)
            
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            generated_text = prompt
            start_time = time.time()
            
            # 逐 token 生成
            for step in range(max_length):
                with torch.no_grad():
                    # Bottom (本地)
                    hidden_bottom = bottom_model(input_ids)
                    
                    # Trunk (远程)
                    hidden_trunk = client.compute(hidden_bottom, model_id=MODEL_ID)
                    
                    # Top (本地)
                    output = top_model(hidden_trunk)
                    logits = output.logits[:, -1, :] / temperature
                
                # Top-K 采样
                top_k = 50
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                probs = torch.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1)
                next_token_id = top_k_indices.gather(-1, next_token_idx)
                
                input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                
                # 解码并打印
                new_word = tokenizer.decode(next_token_id[0])
                print(new_word, end='', flush=True)
                generated_text += new_word
                
                # 遇到结束符停止
                if tokenizer.eos_token_id and next_token_id.item() == tokenizer.eos_token_id:
                    break
            
            elapsed = time.time() - start_time
            tokens = len(input_ids[0]) - len(tokenizer.encode(prompt))
            
            print(f"\n-" * 40)
            print(f"生成了 {tokens} tokens, 耗时 {elapsed:.2f}s, 速度 {tokens/elapsed:.2f} tokens/s")
            
        except KeyboardInterrupt:
            print("\n\n中断，输入 'quit' 退出")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
    
    client.close()

if __name__ == "__main__":
    main()

