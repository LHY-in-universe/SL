import sys
import os
import socket
import torch
import time

# 引入 splitlearn_comm
proj_root = "/Users/lhy/Desktop/Git/SL"
sys.path.insert(0, os.path.join(proj_root, "splitlearn-comm", "src"))

from splitlearn_comm import GRPCComputeClient

TARGET_IP = "192.168.216.129"
TARGET_PORT = 50051

def test_tcp_raw():
    print(f"🔍 [Step 1] TCP Socket 原始测试 ({TARGET_IP}:{TARGET_PORT})...")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect((TARGET_IP, TARGET_PORT))
        print("   ✅ TCP 三次握手成功！")
        
        # 尝试发送一点垃圾数据，看能不能发出去
        # gRPC 服务收到非 HTTP2 数据可能会断开，但服务端应该会有日志（"Http2 handshake failed" 之类）
        msg = b"GET / HTTP/1.1\r\n\r\n" 
        s.sendall(msg)
        print("   ✅ 数据发送成功！")
        
        try:
            data = s.recv(1024)
            print(f"   ✅ 收到回包 ({len(data)} bytes): {data[:50]}...")
        except socket.timeout:
            print("   ⚠️ 没收到回包 (预期内，如果服务端是 gRPC)")
            
        s.close()
        return True
    except Exception as e:
        print(f"   ❌ TCP 连接失败: {e}")
        return False

def test_grpc():
    print(f"\n🚀 [Step 2] gRPC 业务测试...")
    client = GRPCComputeClient(f"{TARGET_IP}:{TARGET_PORT}", timeout=10.0)
    
    print("   正在建立 gRPC 连接...")
    if client.connect():
        print("   ✅ gRPC 连接成功！")
        
        print("   正在发送 Tensor...")
        x = torch.randn(1, 10, 768)
        try:
            y = client.compute(x, model_id="gpt2-trunk")
            print("   🎉🎉🎉 计算成功！全链路打通！")
            print(f"   Result: {tuple(y.shape)}")
        except Exception as e:
            print(f"   ❌ 计算步骤出错: {e}")
    else:
        print("   ❌ gRPC 连接超时 (握手失败)")

if __name__ == "__main__":
    if test_tcp_raw():
        time.sleep(1) # 给服务端一点喘息时间
        test_grpc()
    else:
        print("\n⛔️ TCP 层都不通，无需尝试 gRPC。请检查防火墙/网络路由。")

import os
import socket
import torch
import time

# 引入 splitlearn_comm
proj_root = "/Users/lhy/Desktop/Git/SL"
sys.path.insert(0, os.path.join(proj_root, "splitlearn-comm", "src"))

from splitlearn_comm import GRPCComputeClient

TARGET_IP = "192.168.216.129"
TARGET_PORT = 50051

def test_tcp_raw():
    print(f"🔍 [Step 1] TCP Socket 原始测试 ({TARGET_IP}:{TARGET_PORT})...")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect((TARGET_IP, TARGET_PORT))
        print("   ✅ TCP 三次握手成功！")
        
        # 尝试发送一点垃圾数据，看能不能发出去
        # gRPC 服务收到非 HTTP2 数据可能会断开，但服务端应该会有日志（"Http2 handshake failed" 之类）
        msg = b"GET / HTTP/1.1\r\n\r\n" 
        s.sendall(msg)
        print("   ✅ 数据发送成功！")
        
        try:
            data = s.recv(1024)
            print(f"   ✅ 收到回包 ({len(data)} bytes): {data[:50]}...")
        except socket.timeout:
            print("   ⚠️ 没收到回包 (预期内，如果服务端是 gRPC)")
            
        s.close()
        return True
    except Exception as e:
        print(f"   ❌ TCP 连接失败: {e}")
        return False

def test_grpc():
    print(f"\n🚀 [Step 2] gRPC 业务测试...")
    client = GRPCComputeClient(f"{TARGET_IP}:{TARGET_PORT}", timeout=10.0)
    
    print("   正在建立 gRPC 连接...")
    if client.connect():
        print("   ✅ gRPC 连接成功！")
        
        print("   正在发送 Tensor...")
        x = torch.randn(1, 10, 768)
        try:
            y = client.compute(x, model_id="gpt2-trunk")
            print("   🎉🎉🎉 计算成功！全链路打通！")
            print(f"   Result: {tuple(y.shape)}")
        except Exception as e:
            print(f"   ❌ 计算步骤出错: {e}")
    else:
        print("   ❌ gRPC 连接超时 (握手失败)")

if __name__ == "__main__":
    if test_tcp_raw():
        time.sleep(1) # 给服务端一点喘息时间
        test_grpc()
    else:
        print("\n⛔️ TCP 层都不通，无需尝试 gRPC。请检查防火墙/网络路由。")


