"""
服务端：承载 Qwen3-VL 的 trunk（文本中段层）。

说明：
- 依赖 transformers>=4.57.3，已内置 Qwen3VL 模型。
- 默认拆分：bottom=视觉塔，trunk=文本层0-13，top=文本层14-27+norm+lm_head。
- 这里仅加载 trunk，提供 server_forward 接口供客户端调用。

注意：
- 为了节省显存，默认 trunk 放在 CPU，dtype=float16 也可按需调整。
"""
import os
import torch
from splitlearn_core.quickstart import load_split_model

# 清理代理，避免下载受限
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(k, None)

# 全局保存 trunk 实例
_TRUNK = None
_TRUNK_DEVICE = None


def load_trunk(
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    split_points=(0, 14),
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.float16,
):
    global _TRUNK, _TRUNK_DEVICE
    if _TRUNK is not None:
        return _TRUNK

    # 只加载 trunk
    bottom, trunk, top = load_split_model(
        model_type="qwen3_vl",
        split_points=list(split_points),
        model_name_or_path=model_name,
        cache_dir="./models",
        device=device,
        torch_dtype=torch_dtype,
        parts=["trunk"],
    )
    _TRUNK = trunk
    _TRUNK_DEVICE = device
    return _TRUNK


@torch.inference_mode()
def server_forward(hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
    """
    对外暴露的 trunk 前向。
    输入:
        hidden_states: [B, T, H]，来自视觉前端或上游拼接
        attention_mask: [B, T] 或已扩展 mask
    返回:
        trunk 输出，同形 [B, T, H]
    """
    trunk = load_trunk()
    # 确保在 trunk 设备
    hs = hidden_states.to(_TRUNK_DEVICE)
    am = attention_mask.to(_TRUNK_DEVICE) if attention_mask is not None else None
    out = trunk(hs, attention_mask=am)
    return out


if __name__ == "__main__":
    # 简单自测
    dummy = torch.randn(1, 16, 2048)
    out = server_forward(dummy)
    print("trunk out shape:", tuple(out.shape))
