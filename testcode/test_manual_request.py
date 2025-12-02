"""
手动构造 ComputeRequest 测试脚本
完全按照服务端文档要求进行编码
"""
import sys
import os
import torch
import numpy as np
import grpc
import time

# 引入 splitlearn_comm 路径
proj_root = "/Users/lhy/Desktop/Git/SL"
sys.path.insert(0, os.path.join(proj_root, "splitlearn-comm", "src"))

try:
    # 尝试导入底层 protobuf 定义
    from splitlearn_comm.proto import compute_service_pb2
    from splitlearn_comm.proto import compute_service_pb2_grpc
except ImportError:
    print("❌ 无法导入 proto 定义，请检查 splitlearn-comm 是否编译正确")
    sys.exit(1)

SERVER_ADDRESS = "192.168.216.129:50053"
MODEL_ID = "gpt2-trunk"

def run_test():
    print(f"🔗 正在连接 gRPC 服务器: {SERVER_ADDRESS}")
    
    # 1. 创建 Channel
    channel = grpc.insecure_channel(SERVER_ADDRESS)
    stub = compute_service_pb2_grpc.ComputeServiceStub(channel)
    
    # 等待 Channel 就绪
    try:
        grpc.channel_ready_future(channel).result(timeout=5.0)
        print("✅ gRPC Channel 连接就绪 (TCP握手成功)")
    except grpc.FutureTimeoutError:
        print("❌ gRPC Channel 连接超时 (TCP握手后gRPC协议未响应)")
        return

    # 2. 准备数据 (完全照文档)
    print("📦 准备数据...")
    input_tensor = torch.randn(1, 10, 768)
    
    # 转换: Tensor -> numpy(float32) -> bytes
    array = input_tensor.detach().cpu().numpy().astype(np.float32)
    data_bytes = array.tobytes()
    shape_list = list(input_tensor.shape)
    
    print(f"   - Shape: {shape_list}")
    print(f"   - Bytes len: {len(data_bytes)}")

    # 3. 构造 ComputeRequest
    request = compute_service_pb2.ComputeRequest(
        data=data_bytes,
        shape=shape_list,
        model_id=MODEL_ID,
        request_id=int(time.time())
    )

    # 4. 发送请求
    print(f"🚀 发送 ComputeRequest (model_id={MODEL_ID})...")
    try:
        # 设置较长的超时时间
        response = stub.Compute(request, timeout=15.0)
        
        print("🎉 收到响应！")
        print(f"   - 计算耗时: {response.compute_time_ms:.2f} ms")
        print(f"   - 输出 Shape: {response.shape}")
        
        # 5. 解析响应 (可选)
        output_array = np.frombuffer(response.data, dtype=np.float32)
        output_array = output_array.reshape(response.shape)
        print(f"   - 输出张量均值: {output_array.mean():.4f}")
        print("✅ 测试完全通过")
        
    except grpc.RpcError as e:
        print(f"❌ gRPC 调用失败: {e.code()}")
        print(f"   Details: {e.details()}")

if __name__ == "__main__":
    run_test()

手动构造 ComputeRequest 测试脚本
完全按照服务端文档要求进行编码
"""
import sys
import os
import torch
import numpy as np
import grpc
import time

# 引入 splitlearn_comm 路径
proj_root = "/Users/lhy/Desktop/Git/SL"
sys.path.insert(0, os.path.join(proj_root, "splitlearn-comm", "src"))

try:
    # 尝试导入底层 protobuf 定义
    from splitlearn_comm.proto import compute_service_pb2
    from splitlearn_comm.proto import compute_service_pb2_grpc
except ImportError:
    print("❌ 无法导入 proto 定义，请检查 splitlearn-comm 是否编译正确")
    sys.exit(1)

SERVER_ADDRESS = "192.168.216.129:50053"
MODEL_ID = "gpt2-trunk"

def run_test():
    print(f"🔗 正在连接 gRPC 服务器: {SERVER_ADDRESS}")
    
    # 1. 创建 Channel
    channel = grpc.insecure_channel(SERVER_ADDRESS)
    stub = compute_service_pb2_grpc.ComputeServiceStub(channel)
    
    # 等待 Channel 就绪
    try:
        grpc.channel_ready_future(channel).result(timeout=5.0)
        print("✅ gRPC Channel 连接就绪 (TCP握手成功)")
    except grpc.FutureTimeoutError:
        print("❌ gRPC Channel 连接超时 (TCP握手后gRPC协议未响应)")
        return

    # 2. 准备数据 (完全照文档)
    print("📦 准备数据...")
    input_tensor = torch.randn(1, 10, 768)
    
    # 转换: Tensor -> numpy(float32) -> bytes
    array = input_tensor.detach().cpu().numpy().astype(np.float32)
    data_bytes = array.tobytes()
    shape_list = list(input_tensor.shape)
    
    print(f"   - Shape: {shape_list}")
    print(f"   - Bytes len: {len(data_bytes)}")

    # 3. 构造 ComputeRequest
    request = compute_service_pb2.ComputeRequest(
        data=data_bytes,
        shape=shape_list,
        model_id=MODEL_ID,
        request_id=int(time.time())
    )

    # 4. 发送请求
    print(f"🚀 发送 ComputeRequest (model_id={MODEL_ID})...")
    try:
        # 设置较长的超时时间
        response = stub.Compute(request, timeout=15.0)
        
        print("🎉 收到响应！")
        print(f"   - 计算耗时: {response.compute_time_ms:.2f} ms")
        print(f"   - 输出 Shape: {response.shape}")
        
        # 5. 解析响应 (可选)
        output_array = np.frombuffer(response.data, dtype=np.float32)
        output_array = output_array.reshape(response.shape)
        print(f"   - 输出张量均值: {output_array.mean():.4f}")
        print("✅ 测试完全通过")
        
    except grpc.RpcError as e:
        print(f"❌ gRPC 调用失败: {e.code()}")
        print(f"   Details: {e.details()}")

if __name__ == "__main__":
    run_test()


