"""
远程服务端（gRPC）：承载 Qwen3-VL trunk，端口 50051。

启动示例（远程服务器）：
    PYTHONPATH=./SplitLearnCore/src \
    ./.venv/bin/python qwen3_server_grpc.py

注意：
- 默认 trunk 在 CPU，dtype=float16（可改）
- 需要先安装依赖：pip install grpcio
"""
import os
import warnings

# 抑制警告
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*coroutine.*')
warnings.filterwarnings('ignore', message='.*mutex.*')

import torch
from splitlearn_core.quickstart import load_split_model
from splitlearn_comm.server import GRPCComputeServer
from splitlearn_comm.core import ComputeFunction

# 清理代理
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(k, None)

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
SPLIT_POINTS = [0, 27]  # trunk 包含前 27 层（layers 0-26）
DEVICE = "cpu"  # trunk 在 CPU，节省显存
DTYPE = torch.float16  # 可改 torch.float32


class Qwen3VLTrunkFunction(ComputeFunction):
    """
    包装 Qwen3-VL trunk 的计算函数。
    注意：compute() 只接受一个 input_tensor，这里假设是 [B, T, H]，
    attention_mask 默认为全 1（所有位置可见）。
    """
    
    def __init__(self, trunk_model):
        self.trunk = trunk_model
        self.device = next(trunk_model.parameters()).device
        
    def compute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        执行 trunk 前向。
        输入: [B, T, H]
        输出: [B, T, H]
        """
        input_tensor = input_tensor.to(self.device)
        # 创建默认 attention_mask（全 1）
        B, T, _ = input_tensor.shape
        attn_mask = torch.ones((B, T), device=self.device, dtype=torch.long)
        
        with torch.no_grad():
            output = self.trunk(input_tensor, attention_mask=attn_mask)
        return output
    
    def get_info(self):
        return {
            "name": "Qwen3VLTrunk",
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.trunk.parameters()),
        }


def main():
    print("=" * 70)
    print("Qwen3-VL Trunk gRPC 服务端")
    print("=" * 70)
    print(f"模型: {MODEL_ID}")
    print(f"组件: trunk (文本层 {SPLIT_POINTS[0]}-{SPLIT_POINTS[1]-1}，共 {SPLIT_POINTS[1]} 层)")
    print(f"设备: {DEVICE}")
    print(f"精度: {DTYPE}")
    print(f"端口: 50051")
    print("=" * 70)
    
    # 加载 trunk
    print("\n加载 trunk...")
    _, trunk, _ = load_split_model(
        model_type="qwen3_vl",
        split_points=SPLIT_POINTS,
        model_name_or_path=MODEL_ID,
        cache_dir="./models",
        device=DEVICE,
        torch_dtype=DTYPE,
        parts=["trunk"],
    )
    print("✓ Trunk 加载完成")
    
    # 包装为 ComputeFunction
    compute_fn = Qwen3VLTrunkFunction(trunk)
    
    # 创建并启动 gRPC 服务器
    print("\n启动 gRPC 服务器...")
    server = GRPCComputeServer(
        compute_fn=compute_fn,
        host="0.0.0.0",
        port=50051,
        max_workers=1,  # 单线程避免 mutex 警告
    )
    
    print("✓ 服务器已启动，等待客户端连接...")
    print("  地址: 0.0.0.0:50051")
    print("  按 Ctrl+C 停止\n")
    
    try:
        server.start()
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n\n服务器已停止")
        server.stop()


if __name__ == "__main__":
    main()

