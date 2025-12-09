"""
远程服务端（HTTP）：承载 Qwen3-VL trunk。

- 启动后提供 /health 与 /forward 接口。
- /forward 接收 base64 序列化的 hidden_states / attention_mask，返回 trunk 输出。
- 依赖 transformers>=4.57.3，已内置 Qwen3-VL。
- 默认 trunk 在 CPU，dtype=float16，可按需调整 DEVICE/DTYPE。

启动示例：
    PYTHONPATH=./SplitLearnCore/src \
    ./ .venv/bin/python qwen3_server_http.py
"""
import base64
import io
import os
from typing import Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from splitlearn_core.quickstart import load_split_model

# 清理代理
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(k, None)

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
SPLIT_POINTS = [0, 14]
DEVICE = "cpu"
DTYPE = torch.float16  # 可改 torch.float32

app = FastAPI()
_TRUNK = None


def _load_trunk():
    global _TRUNK
    if _TRUNK is not None:
        return _TRUNK
    _, trunk, _ = load_split_model(
        model_type="qwen3_vl",
        split_points=SPLIT_POINTS,
        model_name_or_path=MODEL_ID,
        cache_dir="./models",
        device=DEVICE,
        torch_dtype=DTYPE,
        parts=["trunk"],
    )
    _TRUNK = trunk
    return _TRUNK


def tensor_to_base64(t: torch.Tensor) -> str:
    buf = io.BytesIO()
    torch.save(t, buf)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_tensor(data: str, device: Optional[str] = None) -> torch.Tensor:
    raw = base64.b64decode(data.encode("utf-8"))
    buf = io.BytesIO(raw)
    t = torch.load(buf, map_location=device or "cpu")
    return t


class ForwardRequest(BaseModel):
    hidden_states: str  # base64 of torch tensor [B, T, H]
    attention_mask: Optional[str] = None  # base64 of torch tensor [B, T] or expanded


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forward")
def forward(req: ForwardRequest):
    trunk = _load_trunk()
    hs = base64_to_tensor(req.hidden_states, device=DEVICE).to(DEVICE)
    am = base64_to_tensor(req.attention_mask, device=DEVICE) if req.attention_mask else None
    with torch.no_grad():
        out = trunk(hs, attention_mask=am)
    return {"hidden_states": tensor_to_base64(out)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

