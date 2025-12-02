"""
内网连接测试脚本 - 专注测试 192.168.0.16
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

def main():
    address = '192.168.0.16:50053'
    
    print("=" * 60)
    print(f"内网连接测试: {address}")
    print("=" * 60)
    
    print("1. 创建客户端...")
    client = GRPCComputeClient(address, timeout=10.0)  # 增加到10秒超时
    
    try:
        print("2. 尝试连接 (请确保服务端已启动)...")
        if client.connect():
            print("✅ 连接成功！")
            
            print("3. 发送测试数据...")
            input_tensor = torch.randn(1, 10, 768)
            output = client.compute(input_tensor, model_id='gpt2-trunk')
            
            print(f"✅ 计算完成！输出形状: {tuple(output.shape)}")
        else:
            print("❌ 连接失败")
            print("建议检查：")
            print("  - 服务端 IP 是否变了？(在服务端运行 ipconfig)")
            print("  - Windows 防火墙是否允许 Python/端口 50053？")
            print("  - 是否开启了 AP 隔离？")
            
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()

内网连接测试脚本 - 专注测试 192.168.0.16
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

def main():
    address = '192.168.0.16:50053'
    
    print("=" * 60)
    print(f"内网连接测试: {address}")
    print("=" * 60)
    
    print("1. 创建客户端...")
    client = GRPCComputeClient(address, timeout=10.0)  # 增加到10秒超时
    
    try:
        print("2. 尝试连接 (请确保服务端已启动)...")
        if client.connect():
            print("✅ 连接成功！")
            
            print("3. 发送测试数据...")
            input_tensor = torch.randn(1, 10, 768)
            output = client.compute(input_tensor, model_id='gpt2-trunk')
            
            print(f"✅ 计算完成！输出形状: {tuple(output.shape)}")
        else:
            print("❌ 连接失败")
            print("建议检查：")
            print("  - 服务端 IP 是否变了？(在服务端运行 ipconfig)")
            print("  - Windows 防火墙是否允许 Python/端口 50053？")
            print("  - 是否开启了 AP 隔离？")
            
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()


