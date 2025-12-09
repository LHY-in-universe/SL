"""
服务端：承载中段 trunk（除去视觉前端和 LLM 最后一层）。

为了便于快速跑通，不依赖实际大模型，使用一个轻量的
TransformerBlock 仿真 trunk 前向。客户端把视频/视觉特征
编码后发送到 server_forward，服务端返回 trunk 输出。
"""
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, attn_mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = x + attn_out
        x = self.norm1(x)
        x = x + self.mlp(x)
        x = self.norm2(x)
        return x


class TrunkServer(nn.Module):
    """
    仅模拟 trunk，客户端负责视觉前端与最后一层。
    """

    def __init__(self, hidden_size=512, depth=4, num_heads=8):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads=num_heads) for _ in range(depth)]
        )

    @torch.inference_mode()
    def forward(self, x, attn_mask=None):
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        return x


@torch.inference_mode()
def server_forward(tensor: torch.Tensor, attn_mask=None):
    """
    对外暴露的服务端前向接口。
    输入 tensor 形状 [B, T, H]，返回同形输出。
    """
    global _SERVER
    if "_SERVER" not in globals():
        _SERVER = TrunkServer().to(tensor.device)
    return _SERVER(tensor, attn_mask=attn_mask)


if __name__ == "__main__":
    # 简单自测
    x = torch.randn(2, 16, 512)
    y = server_forward(x)
    print("input:", x.shape, "output:", y.shape)

