"""
公网连接测试 - 183.14.28.87
"""
import sys
import os
import torch
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
comm_path = os.path.join(project_root, 'splitlearn-comm', 'src')
sys.path.insert(0, comm_path)

from splitlearn_comm import GRPCComputeClient

address = '183.14.28.87:50053'

print("=" * 60)
print(f"公网连接测试: {address}")
print("=" * 60)

client = GRPCComputeClient(address, timeout=15.0)

try:
    print("正在连接...")
    if client.connect():
        print("✅ 连接成功！")
        
        input_tensor = torch.randn(1, 10, 768)
        output = client.compute(input_tensor, model_id='gpt2-trunk')
        
        print(f"✅ 计算成功！输出: {tuple(output.shape)}")
    else:
        print("❌ 连接失败 - 请确认服务端已启动并配置端口转发")
        
except Exception as e:
    print(f"❌ 错误: {e}")

client.close()

公网连接测试 - 183.14.28.87
"""
import sys
import os
import torch
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
comm_path = os.path.join(project_root, 'splitlearn-comm', 'src')
sys.path.insert(0, comm_path)

from splitlearn_comm import GRPCComputeClient

address = '183.14.28.87:50053'

print("=" * 60)
print(f"公网连接测试: {address}")
print("=" * 60)

client = GRPCComputeClient(address, timeout=15.0)

try:
    print("正在连接...")
    if client.connect():
        print("✅ 连接成功！")
        
        input_tensor = torch.randn(1, 10, 768)
        output = client.compute(input_tensor, model_id='gpt2-trunk')
        
        print(f"✅ 计算成功！输出: {tuple(output.shape)}")
    else:
        print("❌ 连接失败 - 请确认服务端已启动并配置端口转发")
        
except Exception as e:
    print(f"❌ 错误: {e}")

client.close()


