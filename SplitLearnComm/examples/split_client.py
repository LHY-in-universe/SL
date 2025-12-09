"""
客户端：处理视觉前端（ViT）并保留 LLM 的最后一层（top），
中段 trunk 通过调用服务端接口完成。这里用轻量模块模拟，便于快速跑通。
"""
import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from split_server import server_forward  # 同目录下的模拟服务端


class VisionFront(nn.Module):
    """
    简化的视觉前端：Conv2d -> flatten -> projection
    模拟 ViT 编码视频帧/图像后输出 patch 序列。
    """
    def __init__(self, hidden_size=512, patch=16):
        super().__init__()
        self.conv = nn.Conv2d(3, hidden_size, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values: torch.Tensor):
        # pixel_values: [B, 3, H, W]
        x = self.conv(pixel_values)              # [B, H', W', C]
        x = x.flatten(2).transpose(1, 2)         # [B, T, C]
        x = self.norm(x)
        return x


class TopLayer(nn.Module):
    """
    模拟 LLM 的最后一层：RMSNorm + 线性头。
    """
    def __init__(self, hidden_size=512, vocab_size=32000):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        x = self.norm(x)
        logits = self.head(x)  # [B, T, V]
        return logits


def run_once(device="cpu", dtype=torch.float32):
    # 1) 准备模型
    vision = VisionFront().to(device).to(dtype)
    top = TopLayer().to(device).to(dtype)

    # 2) 构造假视频帧（1 张 256x256 RGB 图片）
    pixel = torch.randn(1, 3, 256, 256, device=device, dtype=dtype)

    # 3) 视觉前端 -> patch 序列
    feats = vision(pixel)  # [B, T, C]

    # 4) 发送到服务端做 trunk
    trunk_out = server_forward(feats)  # [B, T, C]

    # 5) 客户端 top 层
    logits = top(trunk_out)  # [B, T, V]

    print("视觉前端输出:", tuple(feats.shape))
    print("服务端 trunk 输出:", tuple(trunk_out.shape))
    print("top logits:", tuple(logits.shape))


if __name__ == "__main__":
    # 清理代理，避免网络环境干扰
    for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        os.environ.pop(k, None)
    run_once(device="cpu", dtype=torch.float32)
