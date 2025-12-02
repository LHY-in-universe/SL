"""
GPT-2 Split Learning 客户端
使用缓存的 Bottom + Top 模型
"""
import os
import sys
import time

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearning', 'src'))
sys.path.insert(0, os.path.join(project_root, 'splitlearn-comm', 'src'))

# ============== 配置 ==============
SERVER_IP = '192.168.0.144'
SERVER_PORT = 50053
# ==================================

print("=" * 50)
print("GPT-2 Split Learning 客户端")
print("=" * 50)

# 1. 先导入 torch 和 transformers
print("\n[1] 导入模块...")
import torch
from transformers import AutoTokenizer

# 2. 导入模型类（需要用于 torch.load 反序列化）
print("[2] 导入模型定义...")
from splitlearn.models.gpt2 import GPT2BottomModel, GPT2TopModel

# 3. 加载模型
print("[3] 加载 Bottom 模型...")
bottom_path = os.path.join(current_dir, "gpt2_bottom_cached.pt")
bottom_model = torch.load(bottom_path, map_location='cpu', weights_only=False)
bottom_model.eval()
print("    ✅ Bottom 加载完成")

print("[4] 加载 Top 模型...")
top_path = os.path.join(current_dir, "gpt2_top_cached.pt")
top_model = torch.load(top_path, map_location='cpu', weights_only=False)
top_model.eval()
print("    ✅ Top 加载完成")

print("[5] 加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('gpt2')
print("    ✅ Tokenizer 加载完成")

# 4. 最后才导入和连接 gRPC
print(f"[6] 连接服务器 {SERVER_IP}:{SERVER_PORT}...")
from splitlearn_comm import GRPCComputeClient
client = GRPCComputeClient(f"{SERVER_IP}:{SERVER_PORT}", timeout=20.0)
if not client.connect():
    print("    ❌ 连接失败!")
    sys.exit(1)
print("    ✅ 连接成功!")

print("\n" + "=" * 50)
print("准备就绪！输入 prompt 生成，输入 quit 退出")
print("=" * 50)

# 交互循环
while True:
    try:
        prompt = input("\n>>> ").strip()
        if prompt.lower() == 'quit':
            break
        if not prompt:
            continue
        
        print("生成中: ", end='')
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        for _ in range(30):
            with torch.no_grad():
                # Bottom (本地)
                h = bottom_model(input_ids)
                # Trunk (远程服务器)
                h = client.compute(h, model_id='gpt2-trunk')
                # Top (本地)
                out = top_model(h)
                logits = out.logits[:, -1, :]
            
            next_id = logits.argmax(dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_id], dim=-1)
            print(tokenizer.decode(next_id[0]), end='', flush=True)
            
            if next_id.item() == tokenizer.eos_token_id:
                break
        
        print()
        
    except KeyboardInterrupt:
        print("\n")
        break
    except Exception as e:
        print(f"\n错误: {e}")

client.close()
print("已退出")
