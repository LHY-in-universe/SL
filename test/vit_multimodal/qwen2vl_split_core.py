#!/usr/bin/env python3
"""
Qwen2-VL-2B 拆分示例（视觉塔 → 文本 Transformer）

说明：
- 使用 SplitLearnCore 的 quickstart，根据 split_points 切分 Qwen2-VL：
  - bottom：视觉塔（ViT+Merger），输出视觉特征序列
  - trunk/top：文本 Transformer 层与 lm_head
- 示例使用单张图像、batch=1，演示接口连通性。
"""
import argparse
import torch
from splitlearn_core.quickstart import load_split_model
from transformers import Qwen2VLProcessor


def parse_args():
    p = argparse.ArgumentParser(description="qwen2-vl 拆分示例")
    p.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct", help="模型名称或本地路径")
    p.add_argument("--device", default=None, help="cpu/cuda/mps，可留空自动检测")
    p.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="权重精度",
    )
    p.add_argument("--image", required=False, help="测试图片路径；不填则用随机图")
    p.add_argument("--split-points", type=int, nargs=2, default=[0, 14], help="拆分点 [bottom_end, trunk_end]，默认[0,14]表示文本层一半一半")
    return p.parse_args()


def get_dtype(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def main():
    args = parse_args()
    torch_dtype = get_dtype(args.dtype)

    print("=" * 70)
    print("加载并拆分 Qwen2-VL")
    print("模型:", args.model)
    print("设备:", args.device or "auto")
    print("精度:", args.dtype)
    print("split_points:", args.split_points)
    print("=" * 70)

    # 加载 processor（生成 pixel_values 与 grid_thw）
    processor = Qwen2VLProcessor.from_pretrained(args.model)

    # 构造视觉输入
    if args.image:
        from PIL import Image
        image = Image.open(args.image).convert("RGB")
    else:
        # 随机图像，shape 按官方处理要求
        from PIL import Image
        import numpy as np
        image = Image.fromarray((np.random.rand(448, 448, 3) * 255).astype('uint8'))

    # processor 会返回 pixel_values (B, C, H, W) 和 vision_infos（含 grid_thw）
    # 需要提供文本（即使是空字符串）以满足 processor 要求
    inputs = processor(images=[image], text=[""], return_tensors="pt")
    pixel_values = inputs["pixel_values"]  # [B, C, H, W]
    image_grid_thw = inputs["image_grid_thw"]  # [num_images, 3]

    # 载入拆分模型
    bottom, trunk, top = load_split_model(
        model_type="qwen2_vl",
        split_points=args.split_points,
        model_name_or_path=args.model,
        cache_dir="./models",
        device=args.device,
        torch_dtype=torch_dtype,
    )

    device = next(bottom.parameters()).device
    pixel_values = pixel_values.to(device=device, dtype=torch_dtype)
    image_grid_thw = image_grid_thw.to(device)

    with torch.no_grad():
        # 前端：视觉塔 -> 视觉特征序列（无 batch 维度，需加 batch）
        vision_feats = bottom(pixel_values, grid_thw=image_grid_thw)  # [N_vision, hidden]
        vision_feats = vision_feats.unsqueeze(0)  # [1, N_vision, hidden]

        # 构造注意力 mask：全 1
        attn_mask = torch.ones(vision_feats.shape[:2], device=device, dtype=torch.long)

        # 中段 Transformer
        trunk_out = trunk(vision_feats, attention_mask=attn_mask)

        # 顶端 + lm_head
        top_out = top(trunk_out, attention_mask=attn_mask)

    print("\n张量形状：")
    print("bottom 输出:", tuple(vision_feats.shape))
    print("trunk 输出:", tuple(trunk_out.shape))
    print("top logits:", tuple(top_out.logits.shape))
    print("\n完成。注意：此示例未拼接文本，仅演示视觉链路拆分接口。实际应用需将视觉特征按 token 位置嵌入文本序列。")


if __name__ == "__main__":
    main()

