#!/usr/bin/env python3
"""
使用 SplitLearnCore API 演示 qwen3vl 2B 视觉模型在 ViT → Transformer 交界处的拆分与推理。

说明：
- 假设模型名称为 "qwen3vl-2b-vision"（请根据你本地/私有权重替换）。
- 拆分点放在 ViT 前端输出序列 `[B, N+1, D]` 与 Transformer Encoder 之间，使用 split_points=[1] 作为示例。
- 运行脚本时会：
  1) 加载模型并拆分 bottom / trunk / top
  2) 用随机图像跑一遍前向，打印各阶段张量形状

依赖：SplitLearnCore 已在本仓库。确保安装 transformers 和对应模型权重可用。
"""

import argparse
import torch
from splitlearn_core.quickstart import load_split_model


def parse_args():
    parser = argparse.ArgumentParser(description="qwen3vl 2B ViT-Transformer 拆分示例")
    parser.add_argument("--model", default="qwen3vl-2b-vision", help="模型名称或本地路径")
    parser.add_argument("--device", default=None, help="cpu/cuda/mps，可留空自动检测")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"], help="权重精度")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--image-size", type=int, default=224, help="输入图像尺寸，默认 224")
    parser.add_argument("--patch-size", type=int, default=16, help="patch 大小，默认 16")
    parser.add_argument("--num-patches", type=int, default=196, help="patch 数量，默认 196 (224/16)²")
    return parser.parse_args()


def get_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def main():
    args = parse_args()
    torch_dtype = get_dtype(args.dtype)

    print("=" * 70)
    print("加载并拆分 qwen3vl 2B 视觉模型 (示例)")
    print("模型:", args.model)
    print("设备:", args.device or "auto")
    print("精度:", args.dtype)
    print("拆分点: ViT → Transformer 交界 (split_points=[1])")
    print("=" * 70)

    try:
        bottom, trunk, top = load_split_model(
            args.model,
            split_points=[1],          # ViT 与 Transformer 交界
            cache_dir="./models",
            device=args.device,
            dtype=torch_dtype,
        )
    except Exception as e:
        print("❌ 加载/拆分失败：", e)
        print("请确认：")
        print("  1) 模型名称或路径是否正确 (args.model)")
        print("  2) 是否已下载 qwen3vl 2B 视觉权重")
        print("  3) 环境已安装 transformers 并可加载该模型")
        return

    # 构造随机输入
    B = args.batch
    H = W = args.image_size
    x = torch.randn(B, 3, H, W, device=bottom.parameters().__next__().device, dtype=torch_dtype)

    print("\n开始前向：")
    with torch.no_grad():
        # 前端 (ViT 序列化输出)
        seq = bottom(x)
        print(f"bottom 输出: {tuple(seq.shape)}  (预期 [B, N+1, D])")

        # 后端 Transformer 编码
        trunk_out = trunk(seq)
        print(f"trunk 输出: {tuple(trunk_out.shape)}  (预期 [B, N+1, D])")

        # 顶端任务头（如分类/生成）
        out = top(trunk_out)
        print(f"top 输出: {tuple(out.shape)}")

    print("\n前向完成。")
    print("提示：如果需要在客户端/服务器拆分部署，保持该张量接口一致即可。")


if __name__ == "__main__":
    main()
