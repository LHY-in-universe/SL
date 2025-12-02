"""
调试版本 - 直接导入，避免触发 gRPC 初始化问题
"""
import os
import sys

# 设置环境变量减少噪音
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'

print("Step 0: 设置路径...")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearning', 'src'))
sys.path.insert(0, os.path.join(project_root, 'splitlearn-comm', 'src'))

print("Step 1: 导入 torch...")
import time
t1 = time.time()
import torch
print(f"  完成 ({time.time()-t1:.2f}s)")

print("Step 2: 导入 transformers...")
t1 = time.time()
from transformers import AutoTokenizer, GPT2Config
print(f"  完成 ({time.time()-t1:.2f}s)")

print("Step 3: 直接导入 GPT2 模型类...")
t1 = time.time()
# 直接导入具体文件，避免导入整个 splitlearn 包
from splitlearn.models.gpt2 import GPT2BottomModel, GPT2TopModel
print(f"  完成 ({time.time()-t1:.2f}s)")

print("Step 4: 检查模型文件...")
bottom_path = os.path.join(current_dir, "gpt2_bottom_cached.pt")
top_path = os.path.join(current_dir, "gpt2_top_cached.pt")
print(f"  Bottom: {os.path.exists(bottom_path)} ({os.path.getsize(bottom_path)/1024/1024:.1f} MB)")
print(f"  Top: {os.path.exists(top_path)} ({os.path.getsize(top_path)/1024/1024:.1f} MB)")

print("Step 5: 加载 Bottom 模型...")
t1 = time.time()
bottom_model = torch.load(bottom_path, map_location='cpu', weights_only=False)
bottom_model.eval()
print(f"  完成 ({time.time()-t1:.2f}s)")

print("Step 6: 加载 Top 模型...")
t1 = time.time()
top_model = torch.load(top_path, map_location='cpu', weights_only=False)
top_model.eval()
print(f"  完成 ({time.time()-t1:.2f}s)")

print("Step 7: 加载 Tokenizer...")
t1 = time.time()
tokenizer = AutoTokenizer.from_pretrained('gpt2')
print(f"  完成 ({time.time()-t1:.2f}s)")

print("Step 8: 导入 gRPC 客户端...")
t1 = time.time()
from splitlearn_comm import GRPCComputeClient
print(f"  完成 ({time.time()-t1:.2f}s)")

print("Step 9: 连接服务器...")
SERVER_IP = '192.168.0.144'
SERVER_PORT = 50053
t1 = time.time()
client = GRPCComputeClient(f"{SERVER_IP}:{SERVER_PORT}", timeout=20.0)
if client.connect():
    print(f"  服务器连接成功 ({time.time()-t1:.2f}s)")
else:
    print(f"  服务器连接失败!")
    sys.exit(1)

print(f"\n✅ 全部加载完成！")
print("=" * 50)
print("输入 prompt 生成文本，输入 quit 退出")
print("=" * 50)

# 简单交互
while True:
    try:
        prompt = input("\n>>> ").strip()
        if prompt.lower() == 'quit':
            break
        if not prompt:
            continue
            
        print("生成中...")
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        for i in range(20):
            with torch.no_grad():
                h1 = bottom_model(input_ids)
                h2 = client.compute(h1, model_id='gpt2-trunk')
                out = top_model(h2)
                logits = out.logits[:, -1, :]
            
            next_id = logits.argmax(dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_id], dim=-1)
            print(tokenizer.decode(next_id[0]), end='', flush=True)
        
        print("\n")
        
    except KeyboardInterrupt:
        print("\n中断")
        break
    except Exception as e:
        print(f"错误: {e}")

client.close()
print("已退出")
