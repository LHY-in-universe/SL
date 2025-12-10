"""
客户端（Gradio + gRPC）：本地跑 Qwen3-VL bottom + top，远程 gRPC 服务跑 trunk。

用法（本地）：
    # 设置远程服务地址（可选，默认 192.168.0.144:50051）
    export QWEN3_TRUNK_SERVER="192.168.0.144:50051"
    
    PYTHONPATH=./SplitLearnCore/src \
    ./.venv/bin/python qwen3_client_gradio_grpc.py

依赖：
- 远程需先启动 `qwen3_server_grpc.py`（端口 50051）
- 本地需要：transformers>=4.57.3, torch, gradio, grpcio
"""
import os
import warnings

# 抑制警告
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*coroutine.*')
warnings.filterwarnings('ignore', message='.*mutex.*')

import torch
import gradio as gr
from PIL import Image

from splitlearn_core.quickstart import load_split_model
from splitlearn_core.utils.shard_loader import ShardLoader
from transformers import AutoProcessor
from splitlearn_comm.client import GRPCComputeClient

# 远程服务地址
SERVER_ADDRESS = os.environ.get("QWEN3_TRUNK_SERVER", "192.168.0.144:50051")

# 清理代理
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(k, None)

model_id = "Qwen/Qwen3-VL-2B-Instruct"
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16  # 显存不足可改 torch.float32

print("=" * 70)
print("Qwen3-VL 客户端（Gradio + gRPC）")
print("=" * 70)
print(f"远程服务: {SERVER_ADDRESS}")
print(f"本地设备: {device}")
print(f"精度: {dtype}")
print("=" * 70)

print("\n【客户端只加载以下组件】")
print("  - bottom: 视觉层（vision_tower）")
print("  - top: 最后一层 LLM（layer 27）+ norm + lm_head")
print("  - trunk: 在远程服务器运行（layers 0-26，共 27 层）")
print("\n加载 bottom 和 top...")
bottom, trunk, top = load_split_model(
    model_type="qwen3_vl",
    split_points=[0, 27],  # top 只包含最后一层（layer 27）
    model_name_or_path=model_id,
    cache_dir="./models",
    device=device,
    torch_dtype=dtype,
    parts=["bottom", "top"],  # 只加载 bottom 和 top，trunk 为 None
)
processor = AutoProcessor.from_pretrained(model_id)

# 加载 embed_tokens 用于处理文本输入（高效提取，无需加载完整模型）
print("\n加载 embed_tokens（用于文本处理）...")
try:
    embed_tokens = ShardLoader.extract_embed_tokens(
        model_path=model_id,
        model_type="qwen3_vl",
        device=device,
        torch_dtype=dtype,
        cache_dir="./models",
    )
    print("✓ embed_tokens 加载完成（高效提取，无需完整模型）")
except Exception as e:
    print(f"\n❌ 加载 embed_tokens 失败！")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误消息: {str(e)}")
    import traceback
    print("\n完整堆栈跟踪:")
    traceback.print_exc()
    print("\n回退到传统加载方式...")
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
    full_model_temp = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir="./models",
        dtype=dtype,
        device_map="cpu",
    )
    embed_tokens = full_model_temp.model.language_model.embed_tokens.to(device)
    del full_model_temp
    import gc
    gc.collect()
    print("✓ embed_tokens 加载完成（使用传统方式）")

print("✓ bottom（视觉层）加载完成")
print("✓ top（最后一层 LLM layer 27 + norm + lm_head）加载完成")
if trunk is None:
    print("✓ trunk 未加载（将在远程服务器运行）")
else:
    print("⚠ 警告: trunk 被加载了，应该为 None")

# 连接远程 gRPC 服务
print(f"\n连接远程服务 {SERVER_ADDRESS}...")
client = GRPCComputeClient(server_address=SERVER_ADDRESS, timeout=300.0)
if not client.connect():
    raise RuntimeError(f"无法连接到远程服务 {SERVER_ADDRESS}")
print("✓ 已连接到远程服务")


def run(image, question, max_new_tokens=50):
    if image is None:
        return "请上传图片"
    
    if not question or question.strip() == "":
        question = "描述这张图片"
    
    try:
        # 处理图片和文本
        inputs = processor(images=[Image.fromarray(image)], text=[question], return_tensors="pt")
        pixel = inputs["pixel_values"].to(device=device, dtype=dtype)
        grid = inputs["image_grid_thw"].to(device)
        input_ids = inputs["input_ids"].to(device)  # [1, text_len]
        
        with torch.no_grad():
            # 1. bottom：视觉塔 -> 视觉特征序列 [N_vision, hidden]
            vision_feats = bottom(pixel, grid_thw=grid)
            if isinstance(vision_feats, tuple):
                vision_feats = vision_feats[0]
            vision_feats = vision_feats.unsqueeze(0)  # [1, N_vision, H]
            
            # 2. 处理文本：input_ids -> embeddings
            text_embeds = embed_tokens(input_ids)  # [1, text_len, H]
            
            # 3. 拼接视觉和文本特征
            # 注意：Qwen3-VL 中，视觉特征需要插入到文本序列的特定位置
            # 这里简化处理：将视觉特征放在文本之前
            combined_embeds = torch.cat([vision_feats, text_embeds], dim=1)  # [1, N_vision + text_len, H]
            combined_len = combined_embeds.shape[1]
            
            # 4. 发送到远程 trunk（通过 gRPC）
            trunk_out = client.compute(combined_embeds.to("cpu"))  # gRPC 传输用 CPU

            # 5. 回到客户端设备跑 top
            trunk_out = trunk_out.to(device)
            attn = torch.ones(trunk_out.shape[:2], device=device, dtype=torch.long)
            top_out = top(trunk_out, attention_mask=attn)

            # 6. 生成文本（自回归）
            generated_ids = [input_ids[0, -1].item()]  # 从最后一个输入 token 开始
            current_embeds = combined_embeds
            
            for _ in range(max_new_tokens):
                # 获取最后一个位置的 logits
                logits = top_out.logits[0, -1, :]  # [vocab_size]
                next_token_id = torch.argmax(logits, dim=-1).item()
                generated_ids.append(next_token_id)
                
                # 检查是否遇到结束 token
                if next_token_id == processor.tokenizer.eos_token_id:
                    break
                
                # 准备下一轮输入：添加新 token 的 embedding
                next_token_embed = embed_tokens(torch.tensor([[next_token_id]], device=device))
                current_embeds = torch.cat([current_embeds, next_token_embed], dim=1)
                
                # 再次通过 trunk 和 top
                trunk_out = client.compute(current_embeds.to("cpu"))
                trunk_out = trunk_out.to(device)
                attn = torch.ones(trunk_out.shape[:2], device=device, dtype=torch.long)
                top_out = top(trunk_out, attention_mask=attn)

        # 解码生成的文本
        generated_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return f"**问题**: {question}\n\n**回答**: {generated_text}"
        
    except Exception as e:
        import traceback
        error_msg = f"错误: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg


with gr.Blocks() as demo:
    gr.Markdown("# Qwen3-VL 拆分客户端")
    gr.Markdown("""
    **本地组件**:
    - 视觉层（bottom）：处理图片输入
    - 最后一层 LLM（top）：layer 27 + norm + lm_head，生成最终输出
    
    **远程组件**:
    - 前 27 层 LLM（trunk）：layers 0-26，在远程服务器运行
    """)
    gr.Markdown(f"**远程服务**: `{SERVER_ADDRESS}`")
    img = gr.Image(label="上传图片")
    question = gr.Textbox(
        label="问题（可选）",
        placeholder="例如：描述这张图片",
        value="描述这张图片"
    )
    max_tokens = gr.Slider(
        minimum=1,
        maximum=200,
        value=50,
        step=1,
        label="最大生成 token 数"
    )
    out = gr.Textbox(label="回答", lines=10)
    
    def process(image, question, max_tokens):
        return run(image, question, max_tokens)
    
    btn = gr.Button("生成回答", variant="primary")
    btn.click(process, inputs=[img, question, max_tokens], outputs=out)
    img.change(process, inputs=[img, question, max_tokens], outputs=out)

if __name__ == "__main__":
    print("\n启动 Gradio ...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,          # 如本地可访问可改 False
        inbrowser=False,
    )

