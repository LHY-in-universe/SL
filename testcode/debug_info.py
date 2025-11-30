"""
调试脚本：检查服务器返回的 ServiceInfo
"""
import sys
import os
import time

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'splitlearn-comm', 'src'))

from splitlearn_comm import GRPCComputeClient

def debug_info():
    print("连接服务器...")
    client = GRPCComputeClient("127.0.0.1:50053", timeout=5.0)
    if not client.connect():
        print("❌ 无法连接")
        return

    print("获取 Service Info...")
    info = client.get_service_info()
    print(f"Info: {info}")
    
    if info and "custom_info" in info:
        print(f"Custom Info: {info['custom_info']}")
    else:
        print("❌ Custom Info 为空！")

if __name__ == "__main__":
    debug_info()
