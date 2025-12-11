"""
GPT-2 分拆模型客户端（Gradio + gRPC）

架构：
  - Bottom (本地): 前 2 层 (layers 0-1) + embeddings
  - Trunk (远程 gRPC): 中间 8 层 (layers 2-9)
  - Top (本地): 后 2 层 (layers 10-11) + final norm + LM head

用法：
    export GPT2_TRUNK_SERVER="localhost:50051"
    PYTHONPATH=./SplitLearnCore/src:./SplitLearnComm/src python gpt2_client_gradio_grpc.py
"""

import os
import sys
import time
import socket
import logging
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")

import torch
import gradio as gr
import plotly.graph_objects as go
import pandas as pd

# 添加路径
sys.path.insert(0, str(Path(__file__).parent / "SplitLearnCore" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "SplitLearnComm" / "src"))

from transformers import AutoTokenizer
from splitlearn_core.quickstart import load_split_model
from splitlearn_comm.client import GRPCComputeClient

# 配置日志
log_dir = Path("./logs")
log_dir.mkdir(exist_ok=True)

LOG_LEVEL_NAME = os.environ.get("GPT2_LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.INFO)
GRPC_VERBOSE = os.environ.get("GPT2_GRPC_LOG", "1") == "1"

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "gpt2_client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 全局变量（模型加载）
# ============================================================================

model_id = "gpt2"
split_points = [2, 10]  # Bottom: 0-1, Trunk: 2-9, Top: 10-11
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
SERVER_ADDRESS = os.environ.get("GPT2_TRUNK_SERVER", "localhost:50051")
CLIENT_PORT = int(os.environ.get("GPT2_CLIENT_PORT", "7860"))
SHARE_ENABLED = os.environ.get("GRADIO_SHARE", "1") == "1"
MODEL_CACHE = str(Path("./models").resolve())

logger.info("=" * 70)
logger.info("GPT-2 分拆客户端（Gradio + gRPC）")
logger.info("=" * 70)
logger.info(f"日志级别: {LOG_LEVEL_NAME}")
logger.info(f"本地设备: {device}")
logger.info(f"远程服务: {SERVER_ADDRESS}")
logger.info(f"拆分点: {split_points}")
logger.info(f"Gradio 端口: {CLIENT_PORT}")
logger.info(f"公网分享: {SHARE_ENABLED}")
logger.info(f"模型缓存目录: {MODEL_CACHE}")
logger.info(f"gRPC 传输日志: {GRPC_VERBOSE}")

# 加载分拆模型（只加载 bottom 和 top）
logger.info("加载 Bottom 和 Top 模型...")
bottom, _, top = load_split_model(
    model_type="gpt2",
    split_points=split_points,
    model_name_or_path=model_id,
    cache_dir="./models",
    device=device,
    parts=["bottom", "top"],  # trunk 不加载
)

# 应用 torch.compile() 优化
if hasattr(torch, 'compile') and device == "cuda":
    logger.info("应用 torch.compile() 优化...")
    try:
        bottom = torch.compile(bottom, mode="reduce-overhead")
        top = torch.compile(top, mode="reduce-overhead")
        logger.info("✓ torch.compile() 优化已应用")
    except Exception as e:
        logger.warning(f"torch.compile() 优化失败: {e}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=MODEL_CACHE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

logger.info("✓ Bottom 和 Top 模型加载完成")
logger.info(f"  Bottom 参数: {sum(p.numel() for p in bottom.parameters())/1e6:.2f}M")
logger.info(f"  Top 参数: {sum(p.numel() for p in top.parameters())/1e6:.2f}M")

# 连接 gRPC 服务
logger.info(f"连接远程服务 {SERVER_ADDRESS}...")
client = GRPCComputeClient(server_address=SERVER_ADDRESS, timeout=60.0)
if not client.connect():
    raise RuntimeError(f"无法连接到服务器 {SERVER_ADDRESS}")
logger.info("✓ 已连接到远程服务")

# 全局统计变量
all_token_stats = []
generation_history = []


# ============================================================================
# 核心生成函数
# ============================================================================

def generate_text_with_kv_cache(
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
):
    """使用 KV Cache 的高效文本生成（生成器函数，流式输出）"""
    global all_token_stats

    logger.info(
        f"[generate] 收到生成请求 "
        f"prompt_len={len(prompt)} max_new_tokens={max_new_tokens} "
        f"temp={temperature} top_k={top_k}"
    )

    # 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # 统计变量
    token_times = []  # 每个 token 的生成时间
    server_compute_times = []  # 服务端计算时间
    network_times = []  # 网络往返时间
    bottom_times = []
    top_times = []

    generated_tokens = []
    past_key_values = None  # KV cache

    total_start = time.time()

    def _tensor_bytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    with torch.no_grad():
        for step in range(max_new_tokens):
            step_start = time.time()

            # 1. Bottom: 嵌入 + 前 2 层
            bottom_start = time.time()
            if step == 0:
                # 首次：处理完整提示
                current_input_ids = input_ids
            else:
                # 后续：只处理新 token
                current_input_ids = torch.tensor([[next_token_id]], device=device)

            bottom_out = bottom(current_input_ids)
            bottom_time = time.time() - bottom_start
            bottom_times.append(bottom_time * 1000)

            # 2. Trunk (远程 gRPC): 中间 8 层 + KV Cache
            trunk_start = time.time()
            try:
                if GRPC_VERBOSE:
                    logger.info(
                        f"[gRPC->trunk] step={step} "
                        f"input_shape={list(bottom_out.shape)} "
                        f"bytes={_tensor_bytes(bottom_out)} "
                        f"dtype={bottom_out.dtype} "
                        f"use_cache={past_key_values is not None}"
                    )
                trunk_out, present_kv, timing = client.compute_with_cache(
                    bottom_out.to("cpu"),
                    past_key_values=past_key_values,  # 传入历史 KV
                    use_cache=True,           # 返回新 KV
                    model_id="gpt2"
                )
                if GRPC_VERBOSE:
                    logger.info(
                        f"[gRPC<-trunk] step={step} "
                        f"output_shape={list(trunk_out.shape)} "
                        f"bytes={_tensor_bytes(trunk_out)} "
                        f"server_ms={timing.get('server_compute_ms', 0)} "
                        f"network_ms={timing.get('network_overhead_ms', 0)}"
                    )
            except Exception as e:
                logger.error(f"gRPC 调用失败: {e}")
                yield {
                    "text": f"错误：无法连接到服务器\n{str(e)}",
                    "stats": "生成失败"
                }
                return

            trunk_time = time.time() - trunk_start

            # 记录服务端计算时间
            server_compute_times.append(timing.get("server_compute_ms", 0))
            network_times.append(timing.get("network_overhead_ms", 0))

            # 更新 KV cache
            past_key_values = present_kv

            # 3. Top (本地): 后 2 层 + LM head
            top_start = time.time()
            trunk_out = trunk_out.to(device)
            output = top(trunk_out)
            logits = output.logits[0, -1, :]  # 最后一个位置的 logits
            top_time = time.time() - top_start
            top_times.append(top_time * 1000)

            # 4. 采样下一个 token
            sampling_start = time.time()
            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token_id = logits.argmax(dim=-1).item()
            sampling_time = time.time() - sampling_start

            generated_tokens.append(next_token_id)

            # 5. 记录 token 生成时间
            token_time = time.time() - step_start
            token_times.append(token_time * 1000)  # 转换为毫秒

            # 记录详细统计
            token_text = tokenizer.decode([next_token_id])
            all_token_stats.append({
                "step": step,
                "token_id": next_token_id,
                "token_text": token_text,
                "total_ms": token_time * 1000,
                "bottom_ms": bottom_time * 1000,
                "trunk_ms": trunk_time * 1000,
                "top_ms": top_time * 1000,
                "sampling_ms": sampling_time * 1000,
                "server_compute_ms": timing.get("server_compute_ms", 0),
                "network_ms": timing.get("network_overhead_ms", 0),
            })

            # 6. 检查结束符
            if next_token_id == tokenizer.eos_token_id:
                break

            # 7. 实时输出（流式生成）
            current_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            elapsed = time.time() - total_start

            # 记录每个 token 的详细信息到日志
            token_text = tokenizer.decode([next_token_id])
            logger.info(f"Token #{len(generated_tokens)}: '{token_text}' (ID={next_token_id}) | "
                       f"Total={token_time*1000:.2f}ms, Server={timing.get('server_compute_ms', 0):.2f}ms, "
                       f"Network={timing.get('network_overhead_ms', 0):.2f}ms")

            stats_text = f"""🔄 生成中...

Token数: {len(generated_tokens)}/{max_new_tokens}
最新Token: '{token_text}'
速度: {len(generated_tokens)/elapsed:.2f} tokens/s
平均延迟: {sum(token_times)/len(token_times):.2f}ms/token
服务端计算: {sum(server_compute_times)/len(server_compute_times):.2f}ms
网络开销: {sum(network_times)/len(network_times):.2f}ms
"""

            yield prompt + current_text, stats_text

    # 最终统计
    total_time = time.time() - total_start
    final_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # 获取最近生成的token详情（用于最终展示）
    recent_stats = all_token_stats[-len(generated_tokens):] if len(all_token_stats) >= len(generated_tokens) else all_token_stats
    token_list = "\n".join([
        f"  {i+1}. '{s['token_text']}' (ID={s['token_id']}, {s['total_ms']:.1f}ms)"
        for i, s in enumerate(recent_stats[:10])  # 显示前10个token
    ])
    if len(recent_stats) > 10:
        token_list += f"\n  ... (还有 {len(recent_stats)-10} 个token)"

    stats_text = f"""✅ 生成完成

总Token数: {len(generated_tokens)}
总时间: {total_time:.2f}s
平均速度: {len(generated_tokens)/total_time:.2f} tokens/s
平均延迟: {sum(token_times)/len(token_times):.2f}ms/token

组件耗时:
  - Bottom: {sum(bottom_times)/len(bottom_times):.2f}ms
  - Trunk (含网络): {sum(token_times)/len(token_times):.2f}ms
  - Top: {sum(top_times)/len(top_times):.2f}ms

服务端统计:
  - 纯计算时间: {sum(server_compute_times)/len(server_compute_times):.2f}ms
  - 网络开销: {sum(network_times)/len(network_times):.2f}ms

生成的Token (前10个):
{token_list}

提示: 查看 '🔍 Token详情' 标签页查看完整token列表
"""

    # 记录最终统计到日志
    logger.info(f"✅ 生成完成: {len(generated_tokens)} tokens in {total_time:.2f}s ({len(generated_tokens)/total_time:.2f} tokens/s)")
    logger.info(f"   平均延迟: Bottom={sum(bottom_times)/len(bottom_times):.2f}ms, "
               f"Server={sum(server_compute_times)/len(server_compute_times):.2f}ms, "
               f"Network={sum(network_times)/len(network_times):.2f}ms")

    yield prompt + final_text, stats_text


# ============================================================================
# 监控数据获取
# ============================================================================

def update_monitoring():
    """从服务端获取监控数据"""
    try:
        # 获取最新的服务端监控快照
        monitoring_data = client.get_server_monitoring_data()

        if not monitoring_data or len(monitoring_data) == 0:
            return "N/A", "N/A", "N/A", "无监控数据"

        latest = monitoring_data[-1]

        cpu = f"{latest.get('cpu_percent', 0):.1f}%"
        gpu = f"{latest.get('gpu_utilization', 0):.1f}%" if latest.get('gpu_available') else "N/A"
        memory = f"{latest.get('memory_mb', 0):.0f} MB ({latest.get('memory_percent', 0):.1f}%)"

        # 格式化历史记录
        log_lines = []
        for snapshot in monitoring_data[-10:]:  # 最近 10 条
            ts = snapshot.get('timestamp', time.time())
            log_lines.append(
                f"[{time.strftime('%H:%M:%S', time.localtime(ts))}] "
                f"CPU: {snapshot.get('cpu_percent', 0):.1f}% | "
                f"GPU: {snapshot.get('gpu_utilization', 0):.1f}% | "
                f"MEM: {snapshot.get('memory_mb', 0):.0f}MB"
            )

        return cpu, gpu, memory, "\n".join(log_lines)

    except Exception as e:
        return "错误", "错误", "错误", f"获取监控数据失败: {str(e)}"


def update_latency_stats():
    """更新延迟统计"""
    if not all_token_stats:
        empty_df = pd.DataFrame(columns=["指标", "值"])
        empty_fig = go.Figure()
        empty_fig.update_layout(title="暂无数据")
        return empty_df, empty_fig, empty_fig

    # 创建统计表
    df = pd.DataFrame({
        "指标": [
            "总请求数",
            "平均总延迟(ms)",
            "平均Bottom(ms)",
            "平均Trunk+网络(ms)",
            "平均Top(ms)",
            "平均服务端计算(ms)",
            "平均网络开销(ms)",
        ],
        "值": [
            len(all_token_stats),
            f"{sum(s['total_ms'] for s in all_token_stats) / len(all_token_stats):.2f}",
            f"{sum(s['bottom_ms'] for s in all_token_stats) / len(all_token_stats):.2f}",
            f"{sum(s['trunk_ms'] for s in all_token_stats) / len(all_token_stats):.2f}",
            f"{sum(s['top_ms'] for s in all_token_stats) / len(all_token_stats):.2f}",
            f"{sum(s['server_compute_ms'] for s in all_token_stats) / len(all_token_stats):.2f}",
            f"{sum(s['network_ms'] for s in all_token_stats) / len(all_token_stats):.2f}",
        ]
    })

    # 延迟分布直方图
    total_times = [s['total_ms'] for s in all_token_stats]
    fig_dist = go.Figure(data=[go.Histogram(x=total_times, nbinsx=20)])
    fig_dist.update_layout(
        title="Token 延迟分布",
        xaxis_title="延迟 (ms)",
        yaxis_title="频次"
    )

    # Token 生成时间线
    steps = [s['step'] for s in all_token_stats[-50:]]  # 最近50个
    total_ms = [s['total_ms'] for s in all_token_stats[-50:]]
    server_ms = [s['server_compute_ms'] for s in all_token_stats[-50:]]
    network_ms = [s['network_ms'] for s in all_token_stats[-50:]]

    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(x=steps, y=total_ms, name="总延迟", mode='lines+markers'))
    fig_timeline.add_trace(go.Scatter(x=steps, y=server_ms, name="服务端计算", mode='lines'))
    fig_timeline.add_trace(go.Scatter(x=steps, y=network_ms, name="网络开销", mode='lines'))
    fig_timeline.update_layout(
        title="Token 生成时间线（最近50个）",
        xaxis_title="Token 序号",
        yaxis_title="时间 (ms)"
    )

    return df, fig_dist, fig_timeline


def update_token_details():
    """更新Token详情表格（最近50个token）"""
    if not all_token_stats:
        return pd.DataFrame(columns=["序号", "Token", "ID", "总延迟(ms)", "Bottom(ms)", "Trunk(ms)", "Top(ms)", "服务端(ms)", "网络(ms)"])

    # 获取最近50个token
    recent_tokens = all_token_stats[-50:]

    df = pd.DataFrame({
        "序号": [s['step'] for s in recent_tokens],
        "Token": [s['token_text'] for s in recent_tokens],
        "ID": [s['token_id'] for s in recent_tokens],
        "总延迟(ms)": [f"{s['total_ms']:.2f}" for s in recent_tokens],
        "Bottom(ms)": [f"{s['bottom_ms']:.2f}" for s in recent_tokens],
        "Trunk(ms)": [f"{s['trunk_ms']:.2f}" for s in recent_tokens],
        "Top(ms)": [f"{s['top_ms']:.2f}" for s in recent_tokens],
        "服务端(ms)": [f"{s['server_compute_ms']:.2f}" for s in recent_tokens],
        "网络(ms)": [f"{s['network_ms']:.2f}" for s in recent_tokens],
    })

    return df


# ============================================================================
# Gradio 界面
# ============================================================================

with gr.Blocks(title="GPT-2 分拆客户端", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# GPT-2 分拆学习客户端")
    gr.Markdown(f"""
    **架构**:
    - 🖥️ **本地 Bottom**: 前 2 层 (layers 0-1) + embeddings
    - ☁️ **远程 Trunk**: 中间 8 层 (layers 2-9) - `{SERVER_ADDRESS}`
    - 🖥️ **本地 Top**: 后 2 层 (layers 10-11) + final norm + LM head
    - ⚡ **优化**: KV Cache + torch.compile()
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
            fn=generate_text_with_kv_cache,
            inputs=[prompt_input, max_tokens, temperature, top_k],
            outputs=[output_text, stats_display],
        )

    with gr.Tab("📊 服务端监控"):
        gr.Markdown("实时监控服务端资源使用情况（每2秒刷新）")

        with gr.Row():
            cpu_display = gr.Textbox(label="CPU 使用率", interactive=False)
            gpu_display = gr.Textbox(label="GPU 使用率", interactive=False)
            memory_display = gr.Textbox(label="内存使用", interactive=False)

        monitoring_log = gr.Textbox(label="监控历史（最近10条）", lines=12, interactive=False)

    with gr.Tab("⏱️ 延迟分析"):
        gr.Markdown("Token 生成性能分析（每5秒刷新）")

        latency_table = gr.DataFrame(label="延迟统计")

        with gr.Row():
            latency_plot = gr.Plot(label="延迟分布")
            token_timeline = gr.Plot(label="Token 生成时间线")

    with gr.Tab("🔍 Token详情"):
        gr.Markdown("逐Token生成详情（最近50个，每3秒刷新）")
        gr.Markdown("""
        **说明**：此表格显示每个生成的token及其详细时间分解
        - **Token**: 生成的文本token
        - **总延迟**: 从输入到输出的完整时间
        - **Bottom**: 本地前2层计算时间
        - **Trunk**: 远程8层计算+网络传输时间
        - **Top**: 本地后2层计算时间
        - **服务端**: 远程服务器纯计算时间
        - **网络**: 网络往返开销
        """)

        token_details_table = gr.DataFrame(label="Token生成详情", wrap=True)

    # 自动刷新监控数据
    demo.load(
        fn=update_monitoring,
        outputs=[cpu_display, gpu_display, memory_display, monitoring_log],
        every=2
    )

    # 自动刷新延迟统计
    demo.load(
        fn=update_latency_stats,
        outputs=[latency_table, latency_plot, token_timeline],
        every=5
    )

    # 自动刷新Token详情
    demo.load(
        fn=update_token_details,
        outputs=[token_details_table],
        every=3
    )


if __name__ == "__main__":
    logger.info("\n启动 Gradio 界面...")
    demo.queue()  # Gradio 6.0 推荐添加队列
    logger.info(
        f"启动 Gradio: share={SHARE_ENABLED}, server_name=0.0.0.0, "
        f"port={CLIENT_PORT}"
    )
    launch_kwargs = dict(
        share=SHARE_ENABLED,
        server_name="127.0.0.1",  # 使用 127.0.0.1 而不是 0.0.0.0，避免 localhost 访问问题
        server_port=CLIENT_PORT,
        show_error=True,
        inbrowser=SHARE_ENABLED,
    )
    local_url = share_url = None
    try:
        _, local_url, share_url = demo.launch(**launch_kwargs)
    except (OSError, ValueError) as e:
        # 处理端口被占用或 localhost 不可访问的情况
        if "localhost" in str(e).lower() or "share" in str(e).lower():
            # localhost 不可访问，强制启用 share
            logger.warning(f"localhost 不可访问，自动启用公网分享: {e}")
            launch_kwargs["share"] = True
            launch_kwargs["server_name"] = "127.0.0.1"
            _, local_url, share_url = demo.launch(**launch_kwargs)
        else:
            # 端口被占用，自动切换端口
            logger.warning(f"端口 {CLIENT_PORT} 被占用，尝试自动切换: {e}")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                free_port = s.getsockname()[1]
            launch_kwargs["server_port"] = free_port
            logger.info(f"改用空闲端口: {free_port}")
            _, local_url, share_url = demo.launch(**launch_kwargs)

    logger.info(f"Gradio 本地地址: {local_url}")
    logger.info(f"Gradio 分享链接: {share_url if share_url else '未开启或创建失败'}")
