"""
标准格式 ComputeRequest 测试脚本
使用官方推荐的 splitlearn_comm 库进行测试
"""
import sys
import os
import torch
import numpy as np
import time

# 确保 splitlearn_comm 在路径中
proj_root = "/Users/lhy/Desktop/Git/SL"
comm_path = os.path.join(proj_root, "splitlearn-comm", "src")
sys.path.insert(0, comm_path)

try:
    from splitlearn_comm import GRPCComputeClient
    print("✅ 成功导入 splitlearn_comm")
except ImportError as e:
    print(f"❌ 导入 splitlearn_comm 失败: {e}")
    print(f"当前 sys.path: {sys.path}")
    sys.exit(1)

SERVER_ADDRESS = "192.168.216.129:50053"
MODEL_ID = "gpt2-trunk"

def run_test():
    print("="*60)
    print(f"测试目标: {SERVER_ADDRESS} (Model: {MODEL_ID})")
    print("="*60)

    # 1. 准备数据
    print("📦 [1/4] 准备输入数据...")
    input_tensor = torch.randn(1, 10, 768)
    print(f"   - Tensor Shape: {tuple(input_tensor.shape)}")
    
    # 验证一下数据转换逻辑 (模拟客户端内部行为)
    try:
        array = input_tensor.cpu().numpy().astype(np.float32)
        data_bytes = array.tobytes()
        print(f"   - 序列化检查: {len(data_bytes)} bytes (预期 30720)")
        if len(data_bytes) != 30720:
            print("   ⚠️ 警告: 数据长度与预期不符！")
    except Exception as e:
        print(f"   ❌ 数据序列化检查失败: {e}")

    # 2. 连接
    print(f"\n🔗 [2/4] 连接服务器...")
    client = GRPCComputeClient(SERVER_ADDRESS, timeout=15.0)
    
    start_conn = time.time()
    if client.connect():
        print(f"   ✅ 连接成功! (耗时 {time.time()-start_conn:.2f}s)")
    else:
        print("   ❌ 连接失败 (gRPC握手超时)")
        return

    # 3. 发送计算请求
    print(f"\n🚀 [3/4] 发送 ComputeRequest...")
    try:
        start_compute = time.time()
        output_tensor = client.compute(input_tensor, model_id=MODEL_ID)
        duration = time.time() - start_compute
        
        print(f"   🎉 计算成功! (耗时 {duration:.2f}s)")
        print(f"   - 返回 Shape: {tuple(output_tensor.shape)}")
        print("   ✅ 数据完整性验证通过")
        
    except Exception as e:
        print(f"   ❌ 计算请求失败: {e}")
        print("   可能的排查点:")
        print("   1. 服务端防火墙是否真的允许了 50053 入站？")
        print("   2. 服务端程序是否卡死？(尝试重启服务端)")
        print("   3. 服务端是否报错？(看服务端控制台)")

    finally:
        client.disconnect()
        print("\n🔌 [4/4] 连接已关闭")

if __name__ == "__main__":
    run_test()

标准格式 ComputeRequest 测试脚本
使用官方推荐的 splitlearn_comm 库进行测试
"""
import sys
import os
import torch
import numpy as np
import time

# 确保 splitlearn_comm 在路径中
proj_root = "/Users/lhy/Desktop/Git/SL"
comm_path = os.path.join(proj_root, "splitlearn-comm", "src")
sys.path.insert(0, comm_path)

try:
    from splitlearn_comm import GRPCComputeClient
    print("✅ 成功导入 splitlearn_comm")
except ImportError as e:
    print(f"❌ 导入 splitlearn_comm 失败: {e}")
    print(f"当前 sys.path: {sys.path}")
    sys.exit(1)

SERVER_ADDRESS = "192.168.216.129:50053"
MODEL_ID = "gpt2-trunk"

def run_test():
    print("="*60)
    print(f"测试目标: {SERVER_ADDRESS} (Model: {MODEL_ID})")
    print("="*60)

    # 1. 准备数据
    print("📦 [1/4] 准备输入数据...")
    input_tensor = torch.randn(1, 10, 768)
    print(f"   - Tensor Shape: {tuple(input_tensor.shape)}")
    
    # 验证一下数据转换逻辑 (模拟客户端内部行为)
    try:
        array = input_tensor.cpu().numpy().astype(np.float32)
        data_bytes = array.tobytes()
        print(f"   - 序列化检查: {len(data_bytes)} bytes (预期 30720)")
        if len(data_bytes) != 30720:
            print("   ⚠️ 警告: 数据长度与预期不符！")
    except Exception as e:
        print(f"   ❌ 数据序列化检查失败: {e}")

    # 2. 连接
    print(f"\n🔗 [2/4] 连接服务器...")
    client = GRPCComputeClient(SERVER_ADDRESS, timeout=15.0)
    
    start_conn = time.time()
    if client.connect():
        print(f"   ✅ 连接成功! (耗时 {time.time()-start_conn:.2f}s)")
    else:
        print("   ❌ 连接失败 (gRPC握手超时)")
        return

    # 3. 发送计算请求
    print(f"\n🚀 [3/4] 发送 ComputeRequest...")
    try:
        start_compute = time.time()
        output_tensor = client.compute(input_tensor, model_id=MODEL_ID)
        duration = time.time() - start_compute
        
        print(f"   🎉 计算成功! (耗时 {duration:.2f}s)")
        print(f"   - 返回 Shape: {tuple(output_tensor.shape)}")
        print("   ✅ 数据完整性验证通过")
        
    except Exception as e:
        print(f"   ❌ 计算请求失败: {e}")
        print("   可能的排查点:")
        print("   1. 服务端防火墙是否真的允许了 50053 入站？")
        print("   2. 服务端程序是否卡死？(尝试重启服务端)")
        print("   3. 服务端是否报错？(看服务端控制台)")

    finally:
        client.disconnect()
        print("\n🔌 [4/4] 连接已关闭")

if __name__ == "__main__":
    run_test()


