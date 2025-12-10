"""
GPT-2 分拆模型服务端（gRPC）

职责：
  - 加载 Trunk 模型（中间 8 层，layers 2-9）
  - 处理 KV Cache
  - 推送监控数据
  - 记录详细日志

用法：
    PYTHONPATH=./SplitLearnCore/src:./SplitLearnComm/src python gpt2_server_grpc.py
"""

import os
import sys
import logging
import time
from pathlib import Path

import torch

# 添加路径
sys.path.insert(0, str(Path(__file__).parent / "SplitLearnCore" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "SplitLearnComm" / "src"))

from splitlearn_core.quickstart import load_split_model
from splitlearn_comm.server import GRPCComputeServer
from splitlearn_comm.core import ComputeFunction

# 配置日志
log_dir = Path("./logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "gpt2_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GPT2TrunkFunction(ComputeFunction):
    """GPT-2 Trunk 计算函数，支持 KV Cache"""

    def __init__(self, trunk_model, device):
        self.trunk = trunk_model
        self.device = device
        self.request_count = 0

        logger.info(f"GPT2TrunkFunction initialized on {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in trunk_model.parameters())/1e6:.2f}M")

    def compute(
        self,
        input_tensor: torch.Tensor,
        past_key_values=None,
        use_cache: bool = True,
        **kwargs
    ):
        """执行 Trunk 前向传播"""
        self.request_count += 1

        # 移动到设备
        input_tensor = input_tensor.to(self.device)

        # 日志记录
        logger.info(f"Request #{self.request_count}: input_shape={input_tensor.shape}")
        if past_key_values:
            logger.debug(f"  Using KV cache: {len(past_key_values)} layers")

        start_time = time.time()

        with torch.no_grad():
            # 执行 Trunk 前向
            # 注意：GPT2TrunkModel.forward 接受 hidden_states 作为输入
            output = self.trunk(
                input_tensor,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

            # Trunk 可能返回 (hidden_states, present_key_values)
            if isinstance(output, tuple):
                hidden_states, present_kv = output
            else:
                hidden_states = output
                present_kv = None

        compute_time = time.time() - start_time

        logger.info(f"Request #{self.request_count}: completed in {compute_time*1000:.2f}ms")

        # 返回结果和 KV cache
        if use_cache and present_kv is not None:
            return hidden_states, present_kv
        else:
            return hidden_states

    def get_info(self):
        """返回服务信息"""
        return {
            "name": "GPT2Trunk",
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.trunk.parameters()),
            "layers": "2-9 (8 layers)",
            "total_requests": self.request_count,
        }


def main():
    # 配置
    model_id = "gpt2"
    split_points = [2, 10]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    port = 50051

    logger.info("=" * 70)
    logger.info("GPT-2 Trunk 服务端（gRPC）")
    logger.info("=" * 70)
    logger.info(f"设备: {device}")
    logger.info(f"端口: {port}")
    logger.info(f"拆分点: {split_points} (Trunk 包含 layers 2-9)")

    # 加载 Trunk 模型
    logger.info("加载 Trunk 模型...")
    _, trunk, _ = load_split_model(
        model_type="gpt2",
        split_points=split_points,
        model_name_or_path=model_id,
        cache_dir="./models",
        device=device,
        parts=["trunk"],  # 只加载 trunk
    )

    # 应用 torch.compile() 优化（如果支持）
    if hasattr(torch, 'compile') and device == "cuda":
        logger.info("应用 torch.compile() 优化...")
        try:
            trunk = torch.compile(trunk, mode="reduce-overhead")
            logger.info("✓ torch.compile() 优化已应用")
        except Exception as e:
            logger.warning(f"torch.compile() 优化失败: {e}")
            logger.warning("继续使用未编译的模型")

    logger.info("✓ Trunk 模型加载完成")

    # 创建计算函数
    compute_fn = GPT2TrunkFunction(trunk, device)

    # 创建 gRPC 服务器
    logger.info(f"创建 gRPC 服务器...")
    server = GRPCComputeServer(
        compute_fn=compute_fn,
        host="0.0.0.0",
        port=port,
        max_workers=4,
        version="1.0.0",
    )

    logger.info(f"启动 gRPC 服务器在 0.0.0.0:{port}...")
    server.start()

    logger.info("✓ 服务器已启动，等待客户端连接...")
    logger.info("按 Ctrl+C 停止服务器")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("\n收到中断信号，关闭服务器...")
        server.stop()
        logger.info("服务器已关闭")


if __name__ == "__main__":
    main()
