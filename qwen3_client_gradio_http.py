"""
客户端（Gradio）：本地跑 Qwen3-VL bottom + top，远程 HTTP 服务跑 trunk。

用法（本地）：
    # 确保已安装 transformers>=4.57.3, torch, gradio, requests
    PYTHONPATH=./SplitLearnCore/src \
    ./ .venv/bin/python qwen3_client_gradio_http.py

依赖环境：
- 远程需先启动 `qwen3_server_http.py`，默认 http://<server>:8000/forward
- 本地默认 device=mps（可改 cpu），dtype=float16（显存不够可改 float32）
"""
import base64
import io
import os
import requests
import torch
import gradio as gr
from PIL import Image

from splitlearn_core.quickstart import load_split_model
from transformers import AutoProcessor

# 远程服务地址
SERVER_URL = os.environ.get("QWEN3_TRUNK_URL", "http://127.0.0.1:8000/forward")

# 清理代理
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(k, None)

model_id = "Qwen/Qwen3-VL-2B-Instruct"
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16  # 显存不足可改 torch.float32

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


def tensor_to_base64(t: torch.Tensor) -> str:
    buf = io.BytesIO()
    torch.save(t, buf)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_tensor(data: str, device: str) -> torch.Tensor:
    raw = base64.b64decode(data.encode("utf-8"))
    buf = io.BytesIO(raw)
    t = torch.load(buf, map_location=device)
    return t


def call_trunk_remote(hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
    payload = {
        "hidden_states": tensor_to_base64(hidden_states),
        "attention_mask": tensor_to_base64(attention_mask) if attention_mask is not None else None,
    }
    r = requests.post(SERVER_URL, json=payload, timeout=300)
    r.raise_for_status()
    out_b64 = r.json()["hidden_states"]
    return base64_to_tensor(out_b64, device="cpu")


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

        # 远程 trunk（在 CPU），返回 CPU 张量
        trunk_out = call_trunk_remote(feats.to("cpu"), attention_mask=attn.to("cpu"))

        # 回客户端设备跑 top
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
    gr.Markdown("# Qwen3-VL 拆分（客户端 bottom+top，本地，trunk 远程 HTTP）")
    img = gr.Image(label="上传图片")
    out = gr.JSON(label="张量信息")
    img.change(run, inputs=img, outputs=out)

if __name__ == "__main__":
    print("启动 Gradio ... 远程 trunk URL:", SERVER_URL)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,          # 如本地可访问可改 False
        inbrowser=False,
    )

