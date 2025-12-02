"""
连接服务端最简 RawServer 的测试客户端
"""
import grpc
import sys

SERVER_ADDRESS = "192.168.216.129:50051"

def run():
    print(f"🚀 连接最简测试服务器: {SERVER_ADDRESS}")
    
    # 1. 建立 Channel
    channel = grpc.insecure_channel(SERVER_ADDRESS)
    
    # 2. 等待连接就绪
    try:
        print("   正在握手 (timeout=5s)...")
        grpc.channel_ready_future(channel).result(timeout=5.0)
        print("   ✅ TCP/gRPC 握手成功！通道已就绪。")
    except grpc.FutureTimeoutError:
        print("   ❌ 握手超时！服务端依然没响应。")
        return

    # 3. 发起 Generic 调用
    # 因为服务端是 GenericRpcHandler，我们可以随便调个方法名，比如 "SayHello"
    print("   发送 Generic 请求...")
    try:
        unary_unary = channel.unary_unary('/TestService/SayHello')
        response = unary_unary(b'Are you there?', timeout=5.0)
        print(f"   🎉 收到响应: {response}")
        print("   ✅ 全链路测试通过！环境没问题！")
    except grpc.RpcError as e:
        print(f"   ⚠️ 调用报错: {e}")
        # 只要有报错（比如 Unimplemented），说明连接也是通的，只是方法名对不上
        # 但如果是 DeadlineExceeded，那就是还没通
        if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
            print("   ❌ 依然是超时。")
        else:
            print("   ✅ 连接是通的！(报错是因为服务端逻辑，但这说明网络OK)")

if __name__ == "__main__":
    run()

连接服务端最简 RawServer 的测试客户端
"""
import grpc
import sys

SERVER_ADDRESS = "192.168.216.129:50051"

def run():
    print(f"🚀 连接最简测试服务器: {SERVER_ADDRESS}")
    
    # 1. 建立 Channel
    channel = grpc.insecure_channel(SERVER_ADDRESS)
    
    # 2. 等待连接就绪
    try:
        print("   正在握手 (timeout=5s)...")
        grpc.channel_ready_future(channel).result(timeout=5.0)
        print("   ✅ TCP/gRPC 握手成功！通道已就绪。")
    except grpc.FutureTimeoutError:
        print("   ❌ 握手超时！服务端依然没响应。")
        return

    # 3. 发起 Generic 调用
    # 因为服务端是 GenericRpcHandler，我们可以随便调个方法名，比如 "SayHello"
    print("   发送 Generic 请求...")
    try:
        unary_unary = channel.unary_unary('/TestService/SayHello')
        response = unary_unary(b'Are you there?', timeout=5.0)
        print(f"   🎉 收到响应: {response}")
        print("   ✅ 全链路测试通过！环境没问题！")
    except grpc.RpcError as e:
        print(f"   ⚠️ 调用报错: {e}")
        # 只要有报错（比如 Unimplemented），说明连接也是通的，只是方法名对不上
        # 但如果是 DeadlineExceeded，那就是还没通
        if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
            print("   ❌ 依然是超时。")
        else:
            print("   ✅ 连接是通的！(报错是因为服务端逻辑，但这说明网络OK)")

if __name__ == "__main__":
    run()


