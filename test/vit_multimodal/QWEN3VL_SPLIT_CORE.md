# 使用 SplitLearnCore API 拆分 qwen3vl 2B（ViT → Transformer 交界）

本文档演示如何基于 SplitLearnCore 的核心 API，按“ViT 前端 → Transformer 后端”的边界拆分 qwen3vl 2B 视觉模型。仅作为参考示例，实际模型名称/权重路径请替换为你本地可用的 qwen3vl 2B 视觉检查点。

## 拆分思路

- **拆分点**：在 ViT 前端完成 Patch Embedding、位置编码、CLS 拼接后，将序列张量 `[B, N+1, D]` 发送给后端 Transformer Encoder。
- **前端（Bottom/Trunk）职责**：图像 → patch → embedding → 位置/CLS → 序列化输出。
- **后端（Trunk/Top）职责**：Transformer Encoder 及任务头，输入序列不再重复位置/CLS。

## 使用核心 API 的两种方式

### 方式 A：高层 quickstart（适合已有内置模型）
如果已有与 qwen3vl 对应的模型工厂定义，可直接：
```python
from splitlearn_core.quickstart import load_split_model

# 模型名称/路径请替换为你的 qwen3vl 2B 视觉权重
model_name = "qwen3vl-2b-vision"

# split_points 需与模型定义一致，这里示例：
#  - 0: 仅前端（patch+pos+cls）作为 bottom
#  - 1: Transformer 层起点作为后端
bottom, trunk, top = load_split_model(
    model_name,
    split_points=[1],      # ViT → Transformer 的交界
    cache_dir="./models", # 可选
    device="cpu",         # 按需切换 "cuda"/"mps"
    dtype=None             # 按权重精度设置 torch.float16/bfloat16/float32
)
```
> 注意：`split_points` 的具体含义取决于模型工厂的实现；若内置模型未覆盖，需要自定义模型类（方式 B）。

### 方式 B：自定义 Bottom / Trunk / Top（通用）
适用于自定义拆分点或尚未在工厂注册的模型：
1. **定义前端（Bottom 或 Trunk）**：
   - 输入：图像 `[B, 3, H, W]`
   - 输出：序列 `[B, N+1, D]`（含 CLS，已加位置编码）
2. **定义后端（Trunk 或 Top）**：
   - 输入：序列 `[B, N+1, D]`
   - 输出：Transformer 编码后的特征或最终 logits
3. **注册/组合**：
   - 可通过自定义类继承 `BaseBottomModel` / `BaseTrunkModel` / `BaseTopModel`，或在工厂中注册；也可直接在客户端/服务器显式加载自定义模块。

示例框架（伪代码，需替换为你的 qwen3vl 实现）：
```python
import torch
from splitlearn_core.core import BaseBottomModel, BaseTrunkModel

class Qwen3VLBotto​m(BaseBottomModel):
    def __init__(self, vit_frontend):
        super().__init__()
        self.vit_frontend = vit_frontend  # patch + pos + cls
    def forward(self, x):  # x: [B, 3, H, W]
        return self.vit_frontend(x)      # [B, N+1, D]

class Qwen3VLTrunk(BaseTrunkModel):
    def __init__(self, transformer_encoder):
        super().__init__()
        self.transformer = transformer_encoder
    def forward(self, seq):  # seq: [B, N+1, D]
        return self.transformer(seq)     # [B, N+1, D]

# 装配
bottom = Qwen3VLBotto​m(vit_frontend)
trunk  = Qwen3VLTrunk(transformer_encoder)
```
> 核心要点：前端输出序列的顺序/精度/维度要与后端期望完全一致；后端不重复位置编码和 CLS。

## 接口/对齐要求
- **张量形状**：`[B, N+1, D]`，CLS 在 index 0，后跟 N 个 patch。
- **精度**：保持 dtype 一致（fp16/bf16/fp32），按权重配置。
- **mask（可选）**：如果使用可变长度或 padding，需一并传输并与序列长度对齐。
- **位置编码**：在前端已添加，后端不再重复。

## 传输规模参考
- 224×224 图像，patch=16，N=196，假设 D=768，B=32，fp16：
  - 元素数：32 × 197 × 768 ≈ 4.84M
  - 大小：≈ 9.7 MB / 前向（4.84M × 2 bytes）

## 建议流程
1. 确认 qwen3vl 2B 视觉权重名称/路径，及前端/后端模块拆分方式。
2. 若已在工厂注册，优先用 `load_split_model` + `split_points`；否则按自定义 Bottom/Trunk 实现。
3. 在客户端加载前端，在服务器加载后端，保持张量维度、精度和位置/CLS 对齐；可选传递 mask。
4. 先在单机验证前后端拼接的一致性，再接入 gRPC/通信层。

