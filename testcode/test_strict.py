import sys
import os
import torch
import time

# 强制把 splitlearn_comm 加入路径
proj_root = "/Users/lhy/Desktop/Git/SL"
sys.path.insert(0, os.path.join(proj_root, "splitlearn-comm", "src"))

from splitlearn_comm import GRPCComputeClient

def run_strict_test():
    print("🚀 启动严格模式测试...")
    
    # 1. 创建客户端
    address = "192.168.216.129:50053"
    print(f"   目标: {address}")
    client = GRPCComputeClient(address, timeout=10.0)
    
    # 2. 连接
    print("   正在连接...")
    if not client.connect():
        print("❌ 连接失败")
        return
    print("✅ 连接成功")

    # 3. 准备数据 (严格按照提示)
    # 提示说: input_tensor = torch.randn(1, 10, 768)
    input_tensor = torch.randn(1, 10, 768)
    print(f"   输入形状: {tuple(input_tensor.shape)}")

    # 4. 发送请求
    print("   发送 compute 请求...")
    try:
        output_tensor = client.compute(input_tensor, model_id="gpt2-trunk")
        print("🎉 计算成功！")
        print(f"   输出形状: {tuple(output_tensor.shape)}")
    except Exception as e:
        print(f"❌ 计算失败: {e}")
        # 尝试打印更多错误细节
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_strict_test()

import os
import torch
import time

# 强制把 splitlearn_comm 加入路径
proj_root = "/Users/lhy/Desktop/Git/SL"
sys.path.insert(0, os.path.join(proj_root, "splitlearn-comm", "src"))

from splitlearn_comm import GRPCComputeClient

def run_strict_test():
    print("🚀 启动严格模式测试...")
    
    # 1. 创建客户端
    address = "192.168.216.129:50053"
    print(f"   目标: {address}")
    client = GRPCComputeClient(address, timeout=10.0)
    
    # 2. 连接
    print("   正在连接...")
    if not client.connect():
        print("❌ 连接失败")
        return
    print("✅ 连接成功")

    # 3. 准备数据 (严格按照提示)
    # 提示说: input_tensor = torch.randn(1, 10, 768)
    input_tensor = torch.randn(1, 10, 768)
    print(f"   输入形状: {tuple(input_tensor.shape)}")

    # 4. 发送请求
    print("   发送 compute 请求...")
    try:
        output_tensor = client.compute(input_tensor, model_id="gpt2-trunk")
        print("🎉 计算成功！")
        print(f"   输出形状: {tuple(output_tensor.shape)}")
    except Exception as e:
        print(f"❌ 计算失败: {e}")
        # 尝试打印更多错误细节
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_strict_test()


