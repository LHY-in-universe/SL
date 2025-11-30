"""
命令行版本的 Split Learning 演示
不需要浏览器，直接在终端中运行
"""
import sys
import os
import torch
from transformers import AutoTokenizer

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'SplitLearning', 'src'))
sys.path.append(os.path.join(project_root, 'splitlearn-comm', 'src'))

from splitlearn_comm import GRPCComputeClient

print("=" * 70)
print("Split Learning 命令行演示")
print("=" * 70)

# 1. 加载本地模型
print("\n[1/4] 加载本地模型...")
bottom_path = os.path.join(current_dir, "gpt2_bottom_cached.pt")
top_path = os.path.join(current_dir, "gpt2_top_cached.pt")

if not os.path.exists(bottom_path) or not os.path.exists(top_path):
    print("❌ 模型文件不存在！请先运行: python testcode/prepare_models.py")
    sys.exit(1)

bottom_model = torch.load(bottom_path, map_location='cpu')
top_model = torch.load(top_path, map_location='cpu')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
print("✅ Bottom 和 Top 模型已加载")

# 2. 连接服务器
print("\n[2/4] 连接远程服务器...")
client = GRPCComputeClient("127.0.0.1:50053", timeout=20.0)
if not client.connect():
    print("❌ 无法连接到服务器！")
    print("请确保服务器正在运行: python testcode/start_server.py")
    sys.exit(1)
print("✅ 已连接到服务器")

# 3. 生成文本
print("\n[3/4] 开始生成文本...")
prompt = "The future of artificial intelligence is"
print(f"Prompt: {prompt}")
print("生成中: ", end="", flush=True)

input_ids = tokenizer.encode(prompt, return_tensors="pt")
max_length = 20

for i in range(max_length):
    # Bottom (本地)
    with torch.no_grad():
        hidden_bottom = bottom_model(input_ids)
    
    # Trunk (远程)
    try:
        hidden_trunk = client.compute(hidden_bottom, model_id="gpt2-trunk")
    except Exception as e:
        print(f"\n❌ 服务器错误: {e}")
        break
    
    # Top (本地)
    with torch.no_grad():
        output = top_model(hidden_trunk)
        logits = output.logits
    
    # 采样
    next_token_id = logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
    input_ids = torch.cat([input_ids, next_token_id], dim=-1)
    
    # 打印新生成的词
    new_word = tokenizer.decode(next_token_id[0])
    print(new_word, end="", flush=True)

# 4. 完成
print("\n\n[4/4] 生成完成！")
print("=" * 70)

# 显示统计信息
stats = client.get_statistics()
print(f"\n统计信息:")
print(f"  总请求数: {stats['total_requests']}")
print(f"  平均网络延迟: {stats['avg_network_time_ms']:.2f} ms")
print(f"  平均计算时间: {stats['avg_compute_time_ms']:.2f} ms")
print(f"  平均总时间: {stats['avg_total_time_ms']:.2f} ms")

client.close()
print("\n✅ 演示完成！")
