"""
GPT-2 完整模型部署（对照组）

用途：
  - 作为分拆模型的性能对照
  - 测试分拆是否引入额外开销
  - 验证生成质量一致性

架构：
  - 单机运行完整的 GPT-2 模型
  - 同样的 KV Cache 和优化
  - 同样的性能统计

用法：
    PYTHONPATH=./SplitLearnCore/src:./SplitLearnComm/src python gpt2_full_model_gradio.py
"""

import os
import sys
import time
import logging
from pathlib import Path

import torch
import gradio as gr
import plotly.graph_objects as go
import pandas as pd

# 添加路径
sys.path.insert(0, str(Path(__file__).parent / "SplitLearnCore" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "SplitLearnComm" / "src"))

from transformers import GPT2LMHeadModel, AutoTokenizer

# 配置日志
log_dir = Path("./logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "gpt2_full.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 全局变量（模型加载）
# ============================================================================

model_id = "gpt2"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

logger.info("=" * 70)
logger.info("GPT-2 完整模型（对照组）")
logger.info("=" * 70)
logger.info(f"设备: {device}")
logger.info(f"模型: {model_id}")

logger.info("加载完整 GPT-2 模型...")

# 加载模型和分词器
model = GPT2LMHeadModel.from_pretrained(model_id, cache_dir="./models")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./models")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.to(device)
model.eval()

# 应用 torch.compile() 优化
if hasattr(torch, 'compile') and device == "cuda":
    logger.info("应用 torch.compile() 优化...")
    try:
        model = torch.compile(model, mode="reduce-overhead")
        logger.info("✓ torch.compile() 优化已应用")
    except Exception as e:
        logger.warning(f"torch.compile() 优化失败: {e}")

logger.info(f"✓ 模型加载完成")
logger.info(f"✓ 参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# 全局统计变量
all_token_stats = []


# ============================================================================
# 核心生成函数
# ============================================================================

def generate_with_kv_cache(
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
):
    """使用 KV Cache 的完整模型生成（生成器函数，流式输出）"""
    global all_token_stats

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # 统计
    token_times = []
    total_start = time.time()

    generated_tokens = []
    past_key_values = None

    with torch.no_grad():
        for step in range(max_new_tokens):
            step_start = time.time()

            # 输入
            if step == 0:
                current_input_ids = input_ids
            else:
                current_input_ids = torch.tensor([[next_token_id]], device=device)

            # 前向传播（带 KV Cache）
            outputs = model(
                current_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs.logits[0, -1, :]
            past_key_values = outputs.past_key_values

            # 采样
            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token_id = logits.argmax(dim=-1).item()

            generated_tokens.append(next_token_id)

            # 记录时间
            token_time = time.time() - step_start
            token_times.append(token_time * 1000)

            # 记录详细统计
            all_token_stats.append({
                "step": step,
                "token_id": next_token_id,
                "time_ms": token_time * 1000,
            })

            if next_token_id == tokenizer.eos_token_id:
                break

            # 实时输出
            current_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            elapsed = time.time() - total_start

            stats_text = f"""🔄 生成中...

Token数: {len(generated_tokens)}/{max_new_tokens}
速度: {len(generated_tokens)/elapsed:.2f} tokens/s
平均延迟: {sum(token_times)/len(token_times):.2f}ms/token
"""

            yield prompt + current_text, stats_text

    # 最终统计
    total_time = time.time() - total_start
    final_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    stats_text = f"""✅ 生成完成

总Token数: {len(generated_tokens)}
总时间: {total_time:.2f}s
平均速度: {len(generated_tokens)/total_time:.2f} tokens/s
平均延迟: {sum(token_times)/len(token_times):.2f}ms/token

最小延迟: {min(token_times):.2f}ms
最大延迟: {max(token_times):.2f}ms
"""

    yield prompt + final_text, stats_text


# ============================================================================
# 统计分析
# ============================================================================

def update_stats():
    """更新统计信息"""
    if not all_token_stats:
        empty_df = pd.DataFrame(columns=["指标", "值"])
        empty_fig = go.Figure()
        empty_fig.update_layout(title="暂无数据")
        return empty_df, empty_fig, empty_fig

    # 创建统计表
    times = [s['time_ms'] for s in all_token_stats]

    df = pd.DataFrame({
        "指标": [
            "总Token数",
            "平均延迟(ms)",
            "最小延迟(ms)",
            "最大延迟(ms)",
            "标准差(ms)",
        ],
        "值": [
            len(all_token_stats),
            f"{sum(times) / len(times):.2f}",
            f"{min(times):.2f}",
            f"{max(times):.2f}",
            f"{pd.Series(times).std():.2f}",
        ]
    })

    # 延迟分布直方图
    fig_dist = go.Figure(data=[go.Histogram(x=times, nbinsx=20)])
    fig_dist.update_layout(
        title="Token 延迟分布",
        xaxis_title="延迟 (ms)",
        yaxis_title="频次"
    )

    # Token 生成时间线
    steps = [s['step'] for s in all_token_stats[-50:]]  # 最近50个
    times_recent = [s['time_ms'] for s in all_token_stats[-50:]]

    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(x=steps, y=times_recent, name="延迟", mode='lines+markers'))
    fig_timeline.update_layout(
        title="Token 生成时间线（最近50个）",
        xaxis_title="Token 序号",
        yaxis_title="时间 (ms)"
    )

    return df, fig_dist, fig_timeline


# ============================================================================
# Gradio 界面
# ============================================================================

with gr.Blocks(title="GPT-2 完整模型", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# GPT-2 完整模型（对照组）")
    gr.Markdown(f"""
    **架构**: 完整的 GPT-2 (12 层 transformer)
    **设备**: {device}
    **优化**: KV Cache + torch.compile()
    **用途**: 性能对照基线
    """)

    with gr.Tab("📝 文本生成"):
        with gr.Row():
            with gr.Column(scale=3):
                prompt_input = gr.Textbox(
                    label="输入提示 (Prompt)",
                    placeholder="例如: The future of AI is",
                    lines=3,
                    value="Once upon a time"
                )
                gr.Examples(
                    examples=[
                        ["Once upon a time"],
                        ["The future of AI is"],
                        ["In the year 2050,"],
                        ["Hello, my name is"],
                    ],
                    inputs=prompt_input,
                )

            with gr.Column(scale=2):
                max_tokens = gr.Slider(1, 200, value=50, step=1, label="最大生成 tokens")
                temperature = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Temperature")
                top_k = gr.Slider(0, 100, value=50, step=1, label="Top-K")

        generate_btn = gr.Button("🚀 开始生成", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=3):
                output_text = gr.Textbox(label="生成结果", lines=12, interactive=False)

            with gr.Column(scale=1):
                stats_display = gr.Textbox(label="生成统计", lines=12, interactive=False)

        # 按钮事件
        generate_btn.click(
            fn=generate_with_kv_cache,
            inputs=[prompt_input, max_tokens, temperature, top_k],
            outputs=[output_text, stats_display],
        )

    with gr.Tab("📊 性能统计"):
        gr.Markdown("Token 生成性能分析（每5秒刷新）")

        stats_table = gr.DataFrame(label="统计摘要")

        with gr.Row():
            stats_dist_plot = gr.Plot(label="延迟分布")
            stats_timeline_plot = gr.Plot(label="时间线")

        # 自动刷新统计
        demo.load(
            fn=update_stats,
            outputs=[stats_table, stats_dist_plot, stats_timeline_plot],
            every=5
        )


if __name__ == "__main__":
    logger.info("\n启动 Gradio 界面...")
    demo.queue()  # Gradio 6.0 推荐添加队列
    demo.launch(
        share=True,  # 使用 Gradio 公网分享链接
        show_error=True,
        inbrowser=True,
    )
