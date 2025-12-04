"""
简单测试客户端-服务器连接
"""
import os
import sys
import torch

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearnComm', 'src'))
sys.path.insert(0, os.path.join(project_root, 'SplitLearnCore', 'src'))

from splitlearn_comm import GRPCComputeClient
from splitlearn_core.models.gpt2 import GPT2BottomModel, GPT2TopModel

print("=" * 70)
print("测试 Split Learning 连接")
print("=" * 70)

# 1. 连接服务器
print("\n[1] 连接服务器...")
client = GRPCComputeClient("localhost:50051", timeout=10.0)
if not client.connect():
    print("❌ 无法连接到服务器")
    sys.exit(1)
print("✓ 服务器连接成功")

# 2. 加载本地模型
print("\n[2] 加载本地模型...")
bottom_path = os.path.join(current_dir, "gpt2_bottom_cached.pt")
top_path = os.path.join(current_dir, "gpt2_top_cached.pt")

if not os.path.exists(bottom_path) or not os.path.exists(top_path):
    print("❌ 模型文件不存在，请先运行: python prepare_models.py")
    sys.exit(1)

bottom_model = torch.load(bottom_path, map_location='cpu', weights_only=False)
top_model = torch.load(top_path, map_location='cpu', weights_only=False)
bottom_model.eval()
top_model.eval()
print("✓ 模型加载成功")

# 3. 测试推理
print("\n[3] 测试推理...")
test_input = torch.randn(1, 1, 768)  # GPT-2 hidden size

# Bottom
with torch.no_grad():
    hidden_bottom = bottom_model(test_input)
print(f"✓ Bottom 输出形状: {hidden_bottom.shape}")

# Trunk (远程)
try:
    hidden_trunk = client.compute(hidden_bottom, model_id="gpt2-trunk")
    print(f"✓ Trunk 输出形状: {hidden_trunk.shape}")
except Exception as e:
    print(f"❌ 服务器计算失败: {e}")
    sys.exit(1)

# Top
with torch.no_grad():
    output = top_model(hidden_trunk)
print(f"✓ Top 输出形状: {output.logits.shape}")

print("\n" + "=" * 70)
print("✅ 所有测试通过！Split Learning 工作正常")
print("=" * 70)
print("\n现在可以运行 Gradio 客户端:")
print("  python client_with_gradio.py")
