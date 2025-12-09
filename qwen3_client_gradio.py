"""
客户端：使用 Qwen3-VL 真实模型，拆分为 bottom（视觉塔）+ top（末端层），
服务端承载 trunk（中段文本层）。客户端通过 Gradio 提供上传图片接口，
演示端到端前向（不含文本，仅视觉链路）。

依赖：
- transformers>=4.57.3
- 已下载模型 Qwen/Qwen3-VL-2B-Instruct（缓存放 ./models）
- 需要本地/公网可访问，如环境限制 localhost，可用 share=True。
"""
import os
import torch
import gradio as gr
from PIL import Image
import numpy as np

from splitlearn_core.quickstart import load_split_model
from transformers import AutoProcessor
from qwen3_server import server_forward  # 同目录的服务端前向

# 清理代理
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(k, None)

model_id = "Qwen/Qwen3-VL-2B-Instruct"
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16  # 可改 torch.float32

print("加载 bottom/top ...")
bottom, _, top = load_split_model(
    model_type="qwen3_vl",
    split_points=[0, 14],
    model_name_or_path=model_id,
    cache_dir="./models",
    device=device,
    torch_dtype=dtype,
    parts=["bottom", "top"],
)
processor = AutoProcessor.from_pretrained(model_id)
print("bottom/top ready on", device)


def run(image):
    if image is None:
        return {"error": "请上传图片"}
    inputs = processor(images=[Image.fromarray(image)], text=[""], return_tensors="pt")
    pixel = inputs["pixel_values"].to(device=device, dtype=dtype)
    grid = inputs["image_grid_thw"].to(device)

    with torch.no_grad():
        # bottom：视觉塔 -> 视觉特征序列 [N_vision, hidden]
        feats = bottom(pixel, grid_thw=grid)
        if isinstance(feats, tuple):
            feats = feats[0]
        feats = feats.unsqueeze(0)  # [1, N, H]

        attn = torch.ones(feats.shape[:2], device=feats.device, dtype=torch.long)

        # 发送到服务端做 trunk（在 CPU，节省显存）
        trunk_out = server_forward(feats.to("cpu"), attention_mask=attn.to("cpu"))

        # 回到客户端设备，跑 top + lm_head
        trunk_out = trunk_out.to(device)
        attn = attn.to(device)
        top_out = top(trunk_out, attention_mask=attn)

    logits = top_out.logits  # [1, N, vocab]
    return {
        "vision_feats_shape": str(tuple(feats.shape)),
        "trunk_out_shape": str(tuple(trunk_out.shape)),
        "logits_shape": str(tuple(logits.shape)),
        "logits_dtype": str(logits.dtype),
    }


with gr.Blocks() as demo:
    gr.Markdown("# Qwen3-VL 拆分示例（客户端：bottom+top，服务端：trunk）")
    img = gr.Image(label="上传图片")
    out = gr.JSON(label="张量信息")
    img.change(run, inputs=img, outputs=out)

if __name__ == "__main__":
    print("启动 Gradio ...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,          # 若本地受限可保持 True
        inbrowser=False,
    )
