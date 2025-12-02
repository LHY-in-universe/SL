"""
双模连接测试脚本 - 自动尝试内网和公网
"""
import sys
import os
import torch
import time

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
comm_path = os.path.join(project_root, 'splitlearn-comm', 'src')
sys.path.insert(0, comm_path)

from splitlearn_comm import GRPCComputeClient

def try_connect(address, name):
    print(f"\n🔵 尝试 {name} 连接: {address} ...")
    client = GRPCComputeClient(address, timeout=5.0)  # 5秒超时
    
    try:
        if client.connect():
            print(f"✅ {name} 连接成功！")
            
            # 测试计算
            print("   发送测试数据 [1, 10, 768]...")
            input_tensor = torch.randn(1, 10, 768)
            start = time.time()
            output = client.compute(input_tensor, model_id='gpt2-trunk')
            latency = (time.time() - start) * 1000
            
            print(f"✅ 计算成功！")
            print(f"   输出形状: {tuple(output.shape)}")
            print(f"   往返延迟: {latency:.2f} ms")
            client.close()
            return True
        else:
            print(f"❌ {name} 连接失败")
            return False
    except Exception as e:
        print(f"❌ {name} 错误: {e}")
        return False

def main():
    print("=" * 60)
    print("SplitLearn 服务器连接测试")
    print("=" * 60)
    
    # 1. 尝试内网
    internal_success = try_connect('192.168.0.16:50053', '内网')
    
    # 2. 如果内网失败，尝试新 IP
    if not internal_success:
        print("\n🔄 切换到公网地址 (新 IP)...")
        external_success = try_connect('183.14.28.87:50053', '公网')
        
        if not external_success:
            print("\n" + "=" * 60)
            print("❌ 所有连接尝试都失败了")
            print("建议检查：")
            print("1. 确认服务器程序已启动")
            print("2. 确认防火墙已开放端口 50053")
            print("3. 确认您的电脑已联网")
    
    print("\n测试结束")

if __name__ == "__main__":
    main()

"""
import sys
import os
import torch
import time

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
comm_path = os.path.join(project_root, 'splitlearn-comm', 'src')
sys.path.insert(0, comm_path)

from splitlearn_comm import GRPCComputeClient

def try_connect(address, name):
    print(f"\n🔵 尝试 {name} 连接: {address} ...")
    client = GRPCComputeClient(address, timeout=5.0)  # 5秒超时
    
    try:
        if client.connect():
            print(f"✅ {name} 连接成功！")
            
            # 测试计算
            print("   发送测试数据 [1, 10, 768]...")
            input_tensor = torch.randn(1, 10, 768)
            start = time.time()
            output = client.compute(input_tensor, model_id='gpt2-trunk')
            latency = (time.time() - start) * 1000
            
            print(f"✅ 计算成功！")
            print(f"   输出形状: {tuple(output.shape)}")
            print(f"   往返延迟: {latency:.2f} ms")
            client.close()
            return True
        else:
            print(f"❌ {name} 连接失败")
            return False
    except Exception as e:
        print(f"❌ {name} 错误: {e}")
        return False

def main():
    print("=" * 60)
    print("SplitLearn 服务器连接测试")
    print("=" * 60)
    
    # 1. 尝试内网
    internal_success = try_connect('192.168.0.16:50053', '内网')
    
    # 2. 如果内网失败，尝试新 IP
    if not internal_success:
        print("\n🔄 切换到公网地址 (新 IP)...")
        external_success = try_connect('183.14.28.87:50053', '公网')
        
        if not external_success:
            print("\n" + "=" * 60)
            print("❌ 所有连接尝试都失败了")
            print("建议检查：")
            print("1. 确认服务器程序已启动")
            print("2. 确认防火墙已开放端口 50053")
            print("3. 确认您的电脑已联网")
    
    print("\n测试结束")

if __name__ == "__main__":
    main()
