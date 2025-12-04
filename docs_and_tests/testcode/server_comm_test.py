#!/usr/bin/env python3
"""
gRPC 服务器测试脚本 - 显示数据传输详情

在终端运行此脚本启动服务器，可以看到：
- 接收到的请求数据
- 发送的响应数据
- 数据传输的详细信息
"""

import os
import sys
import time
import torch
import logging

# 设置环境变量（必须在导入 grpc 之前）
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearnComm', 'src'))
sys.path.insert(0, os.path.join(project_root, 'SplitLearnCore', 'src'))

# 配置详细日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

from splitlearn_comm import GRPCComputeServer
from splitlearn_comm.core import ModelComputeFunction, ComputeFunction

# 测试配置
PORT = 50055
HOST = "0.0.0.0"
MODEL_PATH = os.path.join(current_dir, "gpt2_trunk_full.pt")


class VerboseComputeFunction(ComputeFunction):
    """带详细输出的计算函数"""
    
    def __init__(self, model, device="cpu", model_name="test-model"):
        self.model = model.to(device).eval()
        self.device = device
        self.model_name = model_name
        self.request_count = 0
    
    def compute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """执行计算并显示详细信息"""
        self.request_count += 1
        req_id = self.request_count
        
        print("\n" + "=" * 70)
        print(f"📥 服务器收到请求 #{req_id}")
        print("=" * 70)
        
        # 显示输入数据信息
        print(f"\n📊 输入数据信息:")
        print(f"   形状: {input_tensor.shape}")
        print(f"   数据类型: {input_tensor.dtype}")
        print(f"   数据大小: {input_tensor.numel() * 4 / 1024:.2f} KB")
        print(f"   设备: {input_tensor.device}")
        
        # 显示输入数据的统计信息
        print(f"\n📈 输入数据统计:")
        print(f"   最小值: {input_tensor.min().item():.6f}")
        print(f"   最大值: {input_tensor.max().item():.6f}")
        print(f"   平均值: {input_tensor.mean().item():.6f}")
        print(f"   标准差: {input_tensor.std().item():.6f}")
        
        # 显示输入数据的部分值（前几个元素）
        flat_input = input_tensor.flatten()
        print(f"\n🔢 输入数据前10个值:")
        print(f"   {flat_input[:10].tolist()}")
        
        # 执行计算
        print(f"\n⚙️  开始计算...")
        start_time = time.time()
        
        input_on_device = input_tensor.to(self.device)
        with torch.no_grad():
            output = self.model(input_on_device)
        
        compute_time = (time.time() - start_time) * 1000
        
        # 显示输出数据信息
        print(f"\n📤 输出数据信息:")
        print(f"   形状: {output.shape}")
        print(f"   数据类型: {output.dtype}")
        print(f"   数据大小: {output.numel() * 4 / 1024:.2f} KB")
        print(f"   设备: {output.device}")
        print(f"   计算耗时: {compute_time:.2f} ms")
        
        # 显示输出数据的统计信息
        print(f"\n📈 输出数据统计:")
        print(f"   最小值: {output.min().item():.6f}")
        print(f"   最大值: {output.max().item():.6f}")
        print(f"   平均值: {output.mean().item():.6f}")
        print(f"   标准差: {output.std().item():.6f}")
        
        # 显示输出数据的部分值
        flat_output = output.flatten()
        print(f"\n🔢 输出数据前10个值:")
        print(f"   {flat_output[:10].tolist()}")
        
        # 数据传输信息
        input_size_kb = input_tensor.numel() * 4 / 1024
        output_size_kb = output.numel() * 4 / 1024
        total_size_kb = input_size_kb + output_size_kb
        
        print(f"\n📡 数据传输统计:")
        print(f"   接收数据: {input_size_kb:.2f} KB")
        print(f"   发送数据: {output_size_kb:.2f} KB")
        print(f"   总传输: {total_size_kb:.2f} KB")
        print(f"   总耗时: {compute_time:.2f} ms")
        print(f"   吞吐量: {total_size_kb / (compute_time / 1000):.2f} KB/s")
        
        print("=" * 70)
        
        return output.cpu()
    
    def get_info(self):
        return {
            "name": self.model_name,
            "device": self.device,
            "total_requests": self.request_count
        }


def main():
    print("\n" + "=" * 70)
    print("🚀 gRPC 服务器启动")
    print("=" * 70)
    
    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ 模型文件不存在: {MODEL_PATH}")
        print("   请先运行: python testcode/prepare_models.py")
        return 1
    
    # 加载模型
    print(f"\n📦 加载模型: {MODEL_PATH}")
    try:
        from splitlearn_core.models.gpt2 import GPT2TrunkModel
        print("   ✓ GPT2TrunkModel 导入成功")
    except ImportError:
        print("   ⚠️  无法导入 GPT2TrunkModel，尝试直接加载...")
    
    print("   正在加载模型...")
    start_time = time.time()
    model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model.eval()
    load_time = time.time() - start_time
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ 模型加载完成 (耗时: {load_time:.2f} 秒)")
    print(f"   ✓ 参数量: {total_params:,}")
    
    # 创建计算函数
    print(f"\n🔧 创建计算函数...")
    compute_fn = VerboseComputeFunction(
        model=model,
        device="cpu",
        model_name="gpt2-trunk"
    )
    print("   ✓ 计算函数创建成功")
    
    # 创建服务器
    print(f"\n🌐 创建 gRPC 服务器...")
    print(f"   监听地址: {HOST}:{PORT}")
    print(f"   最大工作线程: 1 (单线程模式)")
    
    server = GRPCComputeServer(
        compute_fn=compute_fn,
        host=HOST,
        port=PORT,
        max_workers=1
    )
    print("   ✓ 服务器创建成功")
    
    # 启动服务器
    print(f"\n▶️  启动服务器...")
    server.start()
    print("   ✓ 服务器已启动")
    
    print("\n" + "=" * 70)
    print("✅ 服务器运行中，等待客户端连接...")
    print("=" * 70)
    print(f"\n📡 服务器地址: localhost:{PORT}")
    print(f"💡 在另一个终端运行客户端: python testcode/client_comm_test.py")
    print(f"⏹️  按 Ctrl+C 停止服务器\n")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n\n🛑 收到停止信号，正在关闭服务器...")
        server.stop(grace=2)
        print("   ✓ 服务器已关闭")
        print(f"\n📊 总共处理了 {compute_fn.request_count} 个请求")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

