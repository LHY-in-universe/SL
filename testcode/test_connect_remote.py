"""
客户端连接测试脚本 - 连接到远程 Trunk 服务器
"""
import torch
import sys
import os
import time

# 添加 splitlearn-comm 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
comm_path = os.path.join(project_root, 'splitlearn-comm', 'src')
sys.path.insert(0, comm_path)

from splitlearn_comm import GRPCComputeClient

# ================= 配置区域 =================
SERVER_IP = '192.168.0.144'
SERVER_PORT = 50053
MODEL_ID = 'gpt2-trunk'  # 确保这与服务端加载的模型 ID 一致
# ===========================================

def test_connection():
    address = f"{SERVER_IP}:{SERVER_PORT}"
    print(f"正在连接到服务器: {address} ...")
    
    # 创建客户端
    client = GRPCComputeClient(address, timeout=10.0)
    
    try:
        # 1. 测试连接
        if not client.connect():
            print("❌ 连接失败！请检查:")
            print("   1. IP 地址是否正确")
            print("   2. 服务端防火墙是否允许端口 50053")
            print("   3. 客户端和服务端是否在同一网络 (如果用内网IP)")
            return False
            
        print("✅ 连接成功！")
        
        # 2. 发送测试数据
        print("\n正在发送测试数据...")
        # 创建一个模拟的隐藏层张量 [Batch=1, Seq=10, Hidden=768]
        # 注意：必须匹配服务端模型的 hidden_size (GPT-2 default is 768)
        input_tensor = torch.randn(1, 10, 768)
        
        start_time = time.time()
        output = client.compute(input_tensor, model_id=MODEL_ID)
        latency = (time.time() - start_time) * 1000
        
        print(f"✅ 计算成功！")
        print(f"   输入形状: {tuple(input_tensor.shape)}")
        print(f"   输出形状: {tuple(output.shape)}")
        print(f"   往返延迟: {latency:.2f} ms")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        return False
    finally:
        client.close()

if __name__ == "__main__":
    test_connection()
