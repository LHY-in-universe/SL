"""
极简 gRPC 连接诊断脚本
"""
import sys
import os
import grpc
import time

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'splitlearn-comm', 'src'))

from splitlearn_comm.protocol import compute_service_pb2_grpc, compute_service_pb2

def test_connection(target):
    print(f"正在测试连接到 {target} ...")
    
    # 1. 创建 Channel
    channel = grpc.insecure_channel(target)
    
    # 2. 尝试等待连接就绪 (5秒超时)
    print("尝试建立连接 (waitForConnected)...")
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
        print("✅ Channel 连接成功！")
    except grpc.FutureTimeoutError:
        print("❌ Channel 连接超时！")
        return False
    except Exception as e:
        print(f"❌ Channel 错误: {e}")
        return False

    # 3. 尝试调用服务
    print("尝试调用 Compute 服务...")
    stub = compute_service_pb2_grpc.ComputeServiceStub(channel)
    
    # 构造一个空的请求 (虽然会报错，但能证明连上了)
    request = compute_service_pb2.ComputeRequest(
        model_id="gpt2-trunk",
        data=b"test",
        shape=[1]
    )
    
    try:
        stub.Compute(request, timeout=5)
        print("✅ 调用成功 (意外地)")
    except grpc.RpcError as e:
        print(f"收到服务器响应 (这是好事): Code={e.code()}")
        if e.code() == grpc.StatusCode.INTERNAL or e.code() == grpc.StatusCode.UNKNOWN:
            # 如果服务器因为数据格式错误而报错，说明连接是通的！
            print("✅ 连接是通的！服务器拒绝了我们的无效数据，这很正常。")
            return True
        elif e.code() == grpc.StatusCode.UNAVAILABLE:
            print("❌ 服务不可用 (UNAVAILABLE)")
            return False
        else:
            print(f"⚠️ 其他错误: {e}")
            return True

if __name__ == "__main__":
    targets = ["127.0.0.1:50053", "localhost:50053", "0.0.0.0:50053"]
    
    for target in targets:
        print("\n" + "-"*50)
        test_connection(target)
