"""
带详细调试信息的 gRPC 连接测试
"""
import os
import grpc
import sys

# 强制设置 gRPC 调试环境变量
os.environ["GRPC_VERBOSITY"] = "DEBUG"
os.environ["GRPC_TRACE"] = "all"

SERVER_ADDRESS = "192.168.216.129:50051"

print("="*60)
print("🔍 详细调试模式测试")
print("="*60)
print(f"目标: {SERVER_ADDRESS}")
print(f"代理环境变量: HTTP_PROXY={os.environ.get('HTTP_PROXY', 'None')}")
print(f"代理环境变量: HTTPS_PROXY={os.environ.get('HTTPS_PROXY', 'None')}")
print("="*60)

# 创建 Channel
print("\n[1] 创建 insecure channel...")
channel = grpc.insecure_channel(SERVER_ADDRESS)

# 等待就绪
print("[2] 等待 channel ready (timeout=10s)...")
try:
    grpc.channel_ready_future(channel).result(timeout=10.0)
    print("✅ Channel 就绪！")
except grpc.FutureTimeoutError as e:
    print(f"❌ Channel 就绪超时: {e}")
    print("\n可能的原因:")
    print("1. 服务端 gRPC 服务未正确启动")
    print("2. 服务端防火墙/安全组拦截")
    print("3. 网络中间设备拦截 HTTP/2 流量")
    print("4. 服务端代码卡死")
    sys.exit(1)
except Exception as e:
    print(f"❌ 其他错误: {e}")
    sys.exit(1)

# 尝试调用
print("\n[3] 尝试发起 RPC 调用...")
try:
    unary_unary = channel.unary_unary('/TestService/SayHello')
    response = unary_unary(b'Hello', timeout=5.0)
    print(f"🎉 收到响应: {response}")
except grpc.RpcError as e:
    print(f"⚠️ RPC 调用报错: {e.code()} - {e.details()}")
    if e.code() != grpc.StatusCode.DEADLINE_EXCEEDED:
        print("✅ 连接是通的！(只是方法名或逻辑不匹配)")
except Exception as e:
    print(f"❌ 其他错误: {e}")

channel.close()
print("\n✅ 测试完成")

带详细调试信息的 gRPC 连接测试
"""
import os
import grpc
import sys

# 强制设置 gRPC 调试环境变量
os.environ["GRPC_VERBOSITY"] = "DEBUG"
os.environ["GRPC_TRACE"] = "all"

SERVER_ADDRESS = "192.168.216.129:50051"

print("="*60)
print("🔍 详细调试模式测试")
print("="*60)
print(f"目标: {SERVER_ADDRESS}")
print(f"代理环境变量: HTTP_PROXY={os.environ.get('HTTP_PROXY', 'None')}")
print(f"代理环境变量: HTTPS_PROXY={os.environ.get('HTTPS_PROXY', 'None')}")
print("="*60)

# 创建 Channel
print("\n[1] 创建 insecure channel...")
channel = grpc.insecure_channel(SERVER_ADDRESS)

# 等待就绪
print("[2] 等待 channel ready (timeout=10s)...")
try:
    grpc.channel_ready_future(channel).result(timeout=10.0)
    print("✅ Channel 就绪！")
except grpc.FutureTimeoutError as e:
    print(f"❌ Channel 就绪超时: {e}")
    print("\n可能的原因:")
    print("1. 服务端 gRPC 服务未正确启动")
    print("2. 服务端防火墙/安全组拦截")
    print("3. 网络中间设备拦截 HTTP/2 流量")
    print("4. 服务端代码卡死")
    sys.exit(1)
except Exception as e:
    print(f"❌ 其他错误: {e}")
    sys.exit(1)

# 尝试调用
print("\n[3] 尝试发起 RPC 调用...")
try:
    unary_unary = channel.unary_unary('/TestService/SayHello')
    response = unary_unary(b'Hello', timeout=5.0)
    print(f"🎉 收到响应: {response}")
except grpc.RpcError as e:
    print(f"⚠️ RPC 调用报错: {e.code()} - {e.details()}")
    if e.code() != grpc.StatusCode.DEADLINE_EXCEEDED:
        print("✅ 连接是通的！(只是方法名或逻辑不匹配)")
except Exception as e:
    print(f"❌ 其他错误: {e}")

channel.close()
print("\n✅ 测试完成")


