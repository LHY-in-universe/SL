"""
测试原始 Socket 服务器
"""
import socket
import time

SERVER_IP = "192.168.216.129"
SERVER_PORT = 50051

print(f"🔗 连接原始 Socket 服务器: {SERVER_IP}:{SERVER_PORT}")

try:
    # 创建 Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    
    # 连接
    print("   正在连接...")
    sock.connect((SERVER_IP, SERVER_PORT))
    print("   ✅ TCP 连接成功！")
    
    # 发送数据
    print("   发送测试数据...")
    sock.sendall(b"Hello from Mac Client!")
    print("   ✅ 数据发送成功！")
    
    # 接收响应
    print("   等待响应...")
    response = sock.recv(1024)
    print(f"   🎉 收到响应: {response.decode('utf-8', errors='ignore')}")
    print("   ✅ 原始 Socket 测试完全成功！")
    
    sock.close()
    
except socket.timeout:
    print("   ❌ 连接超时")
except ConnectionRefused:
    print("   ❌ 连接被拒绝")
except Exception as e:
    print(f"   ❌ 错误: {e}")

print("\n" + "="*60)
print("如果上面显示 ✅，说明网络完全没问题！")
print("问题出在 gRPC 库或服务端 gRPC 代码上。")
print("="*60)

测试原始 Socket 服务器
"""
import socket
import time

SERVER_IP = "192.168.216.129"
SERVER_PORT = 50051

print(f"🔗 连接原始 Socket 服务器: {SERVER_IP}:{SERVER_PORT}")

try:
    # 创建 Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    
    # 连接
    print("   正在连接...")
    sock.connect((SERVER_IP, SERVER_PORT))
    print("   ✅ TCP 连接成功！")
    
    # 发送数据
    print("   发送测试数据...")
    sock.sendall(b"Hello from Mac Client!")
    print("   ✅ 数据发送成功！")
    
    # 接收响应
    print("   等待响应...")
    response = sock.recv(1024)
    print(f"   🎉 收到响应: {response.decode('utf-8', errors='ignore')}")
    print("   ✅ 原始 Socket 测试完全成功！")
    
    sock.close()
    
except socket.timeout:
    print("   ❌ 连接超时")
except ConnectionRefused:
    print("   ❌ 连接被拒绝")
except Exception as e:
    print(f"   ❌ 错误: {e}")

print("\n" + "="*60)
print("如果上面显示 ✅，说明网络完全没问题！")
print("问题出在 gRPC 库或服务端 gRPC 代码上。")
print("="*60)


