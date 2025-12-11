"""
GPT-2 完整模型部署（对照组）

用途：
  - 作为分拆模型的性能对照
  - 测试分拆是否引入额外开销
  - 验证生成质量一致性

架构：
  - 单机运行完整的 GPT-2 模型
  - 使用 Hugging Face 官方优化
  - KV Cache 流式生成
  - 详细的性能统计

优化：
  - Hugging Face: low_cpu_mem_usage, torch_dtype, device_map, flash_attention
  - PyTorch: torch.inference_mode(), torch.compile(), 矩阵乘法精度优化
  - 生成: KV Cache, 优化的采样过程, 流式输出

用法：
    # 默认启动
    PYTHONPATH=./SplitLearnCore/src:./SplitLearnComm/src python gpt2_full_model_gradio.py
    
    # 使用优化选项
    GPT2_TORCH_DTYPE=float16 GPT2_FLASH_ATTN=1 python gpt2_full_model_gradio.py
    
环境变量：
    GPT2_FULL_PORT: Gradio 端口（默认 7861）
    GRADIO_SHARE: 是否启用公网分享（默认 1）
    GPT2_TORCH_DTYPE: 数据类型 float32/float16/bfloat16（默认 float32）
    GPT2_LOW_MEM: 低内存加载（默认 1）
    GPT2_USE_DEVICE_MAP: 自动设备映射（默认 0，仅 CUDA）
    GPT2_FLASH_ATTN: Flash Attention 2（默认 0）
    GPT2_ENABLE_COMPILE: torch.compile（默认 1，仅 CUDA）
    GPT2_MATMUL_PRECISION: 矩阵乘法精度 high/medium/low（默认 medium）
    GPT2_LOG_LEVEL: 日志级别 DEBUG/INFO/WARNING/ERROR（默认 INFO）
"""

import os
import sys
import time
import logging
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")

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

LOG_LEVEL_NAME = os.environ.get("GPT2_LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.INFO)

logging.basicConfig(
    level=LOG_LEVEL,
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
FULL_PORT = int(os.environ.get("GPT2_FULL_PORT", "7861"))
SHARE_ENABLED = os.environ.get("GRADIO_SHARE", "1") == "1"
MODEL_CACHE = str(Path("./models").resolve())

# 优化选项（可通过环境变量控制）
ENABLE_TORCH_COMPILE = os.environ.get("GPT2_ENABLE_COMPILE", "1") == "1"  # 是否启用 torch.compile
MATMUL_PRECISION = os.environ.get("GPT2_MATMUL_PRECISION", "medium")  # 矩阵乘法精度: high/medium/low

logger.info("=" * 70)
logger.info("GPT-2 完整模型（对照组）")
logger.info("=" * 70)
logger.info(f"日志级别: {LOG_LEVEL_NAME}")
logger.info(f"设备: {device}")
logger.info(f"模型: {model_id}")
logger.info(f"Gradio 端口: {FULL_PORT}")
logger.info(f"公网分享: {SHARE_ENABLED}")
logger.info(f"模型缓存目录: {MODEL_CACHE}")

logger.info("加载完整 GPT-2 模型...")

# ============================================================================
# Hugging Face 官方优化配置
# ============================================================================

# 1. 低内存加载（减少峰值内存使用）
LOW_CPU_MEM_USAGE = os.environ.get("GPT2_LOW_MEM", "1") == "1"

# 2. 数据类型优化（可选：float16, bfloat16）
TORCH_DTYPE = os.environ.get("GPT2_TORCH_DTYPE", "float32")
if TORCH_DTYPE == "float16":
    torch_dtype = torch.float16
elif TORCH_DTYPE == "bfloat16":
    torch_dtype = torch.bfloat16
else:
    torch_dtype = None  # 使用默认 float32

# 3. 设备映射（自动选择设备）
USE_DEVICE_MAP = os.environ.get("GPT2_USE_DEVICE_MAP", "0") == "1" and device == "cuda"

# 4. 优化的注意力实现（如果可用）
USE_FLASH_ATTENTION = os.environ.get("GPT2_FLASH_ATTN", "0") == "1"

logger.info("Hugging Face 优化配置:")
logger.info(f"  - low_cpu_mem_usage: {LOW_CPU_MEM_USAGE}")
logger.info(f"  - torch_dtype: {TORCH_DTYPE}")
logger.info(f"  - device_map: {USE_DEVICE_MAP}")
logger.info(f"  - flash_attention: {USE_FLASH_ATTENTION}")

# 加载模型和分词器（使用 Hugging Face 官方优化）
model_kwargs = {
    "cache_dir": MODEL_CACHE,
    "low_cpu_mem_usage": LOW_CPU_MEM_USAGE,
}

# 添加数据类型（如果指定）
if torch_dtype is not None and device != "cpu":  # CPU 通常不支持 float16/bfloat16
    model_kwargs["torch_dtype"] = torch_dtype
    logger.info(f"  使用 {TORCH_DTYPE} 精度（可能提升性能）")

# 添加设备映射（仅 CUDA）
if USE_DEVICE_MAP:
    model_kwargs["device_map"] = "auto"
    logger.info("  使用自动设备映射")

# 添加优化的注意力实现
if USE_FLASH_ATTENTION:
    try:
        # 检查是否安装了 flash-attn
        import flash_attn
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("  使用 Flash Attention 2（已安装）")
    except ImportError:
        logger.warning("  Flash Attention 2 未安装，使用默认实现")
        logger.info("  提示: 安装 flash-attn 可提升性能: pip install flash-attn")
    except Exception as e:
        logger.warning(f"  Flash Attention 2 配置失败: {e}")

# 使用 Hugging Face 官方方法加载模型（自动应用所有优化）
try:
    model = GPT2LMHeadModel.from_pretrained(model_id, **model_kwargs)
    logger.info("✓ 模型加载成功（使用 Hugging Face 优化）")
except Exception as e:
    logger.error(f"模型加载失败: {e}", exc_info=True)
    raise

# 加载分词器（使用优化配置）
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir=MODEL_CACHE,
    use_fast=True,  # 使用快速分词器（如果可用）
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 如果未使用 device_map，手动移动到设备
if not USE_DEVICE_MAP:
    model.to(device)

model.eval()

# ============================================================================
# PyTorch 标准库加速优化
# ============================================================================

# 1. 矩阵乘法精度优化（PyTorch 2.0+）
if hasattr(torch, 'set_float32_matmul_precision'):
    try:
        torch.set_float32_matmul_precision(MATMUL_PRECISION)
        logger.info(f"✓ 矩阵乘法精度优化: {MATMUL_PRECISION}")
    except Exception as e:
        logger.warning(f"矩阵乘法精度优化失败: {e}")

# 2. CUDA 优化（如果使用 CUDA）
if device == "cuda":
    # CUDNN 基准测试（自动选择最优算法）
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # 允许非确定性算法以获得更好性能
    logger.info("✓ CUDA 优化已启用 (cudnn.benchmark)")

# 3. MPS 优化（如果使用 MPS）
if device == "mps":
    # MPS 特定优化
    if hasattr(torch.backends, 'mps'):
        logger.info("✓ MPS 设备优化已启用")
    # 注意：MPS 不支持 torch.compile，但可以使用其他优化

# 4. torch.compile() 优化（仅 CUDA，PyTorch 2.0+）
if ENABLE_TORCH_COMPILE and hasattr(torch, 'compile') and device == "cuda":
    logger.info("应用 torch.compile() 优化...")
    try:
        # reduce-overhead: 减少 Python 开销，适合重复调用
        # 其他模式: "default" (平衡), "max-autotune" (最优化但编译慢)
        model = torch.compile(model, mode="reduce-overhead")
        logger.info("✓ torch.compile() 优化已应用 (mode=reduce-overhead)")
    except Exception as e:
        logger.warning(f"torch.compile() 优化失败: {e}")
elif device == "mps":
    logger.info("ℹ MPS 设备不支持 torch.compile()，使用其他优化")

logger.info(f"✓ 模型加载完成")
logger.info(f"✓ 参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
if torch_dtype is not None:
    actual_dtype = next(model.parameters()).dtype
    logger.info(f"✓ 实际精度: {actual_dtype}")
logger.info("")
logger.info("=" * 70)
logger.info("已启用的优化:")
logger.info("")
logger.info("Hugging Face 官方优化:")
logger.info(f"  - low_cpu_mem_usage: {LOW_CPU_MEM_USAGE}")
logger.info(f"  - torch_dtype: {TORCH_DTYPE}")
logger.info(f"  - device_map: {USE_DEVICE_MAP}")
logger.info(f"  - flash_attention: {USE_FLASH_ATTENTION}")
logger.info("")
logger.info("PyTorch 标准库优化:")
logger.info(f"  - 推理模式: torch.inference_mode() (比 torch.no_grad() 更快)")
logger.info(f"  - 矩阵乘法精度: {MATMUL_PRECISION}")
if device == "cuda":
    logger.info(f"  - CUDA 优化: cudnn.benchmark=True")
    if ENABLE_TORCH_COMPILE and hasattr(torch, 'compile'):
        logger.info(f"  - torch.compile(): 已启用 (mode=reduce-overhead)")
    else:
        logger.info(f"  - torch.compile(): 未启用")
elif device == "mps":
    logger.info(f"  - MPS 设备: 使用 MPS 后端优化")
    logger.info(f"  - torch.compile(): MPS 不支持")
else:
    logger.info(f"  - CPU 设备: 使用 CPU 后端")
logger.info("")
logger.info("生成优化:")
logger.info(f"  - KV Cache: 已启用（避免重复计算历史 token）")
logger.info(f"  - output_attentions: False（节省内存）")
logger.info(f"  - output_hidden_states: False（节省内存）")
logger.info("=" * 70)
logger.info("")

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
    """
    使用 KV Cache 的完整模型生成（生成器函数，流式输出）
    
    优化说明：
    - 使用手动循环 + KV Cache 实现流式生成
    - 每个 token 实时输出，提供更好的用户体验
    - 使用 Hugging Face 官方优化参数
    
    性能说明：
    - 在 MPS 设备上，完整模型可能比分拆模型慢，因为：
      1. 所有计算都在本地，没有并行化
      2. MPS 对长序列处理性能不稳定
      3. KV cache 随序列增长，内存压力增加
    - 在 CUDA 设备上，torch.compile() 会显著提升性能
    """
    global all_token_stats

    # ========================================================================
    # 关键调试：立即输出日志，确保函数被调用时能立即看到
    # ========================================================================
    logger.info("=" * 70)
    logger.info("[GENERATE] ========== 函数被调用！==========")
    logger.info(f"[GENERATE] 收到生成请求")
    logger.info(f"[GENERATE]   - prompt: '{prompt}' (长度: {len(prompt) if prompt else 0})")
    logger.info(f"[GENERATE]   - max_new_tokens: {max_new_tokens} (类型: {type(max_new_tokens)})")
    logger.info(f"[GENERATE]   - temperature: {temperature} (类型: {type(temperature)})")
    logger.info(f"[GENERATE]   - top_k: {top_k} (类型: {type(top_k)})")
    logger.info("=" * 70)
    
    # 确保参数类型正确（Gradio 可能传递字符串）
    try:
        max_new_tokens = int(max_new_tokens)
        temperature = float(temperature)
        top_k = int(top_k)
        prompt = str(prompt) if prompt else ""
    except (ValueError, TypeError) as e:
        logger.error(f"参数类型转换失败: {e}, prompt={type(prompt)}, max_new_tokens={type(max_new_tokens)}, temperature={type(temperature)}, top_k={type(top_k)}")
        yield f"错误：参数类型无效\n{str(e)}", "生成失败"
        return

    # 编码输入（使用 tokenizer 的优化方法）
    try:
        # 使用 tokenizer 的优化编码方法
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,  # 不需要 padding
            truncation=False,  # 不截断
            add_special_tokens=True,  # 添加特殊 token
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
    except Exception as e:
        logger.error(f"[generate] 编码输入失败: {e}", exc_info=True)
        yield f"错误：编码输入失败\n{str(e)}", "生成失败"
        return

    # 统计
    token_times = []
    total_start = time.time()

    generated_tokens = []
    past_key_values = None

    # 使用 torch.inference_mode() 替代 torch.no_grad()
    # inference_mode 比 no_grad 更快，因为它完全禁用梯度计算和 autograd
    with torch.inference_mode():
        for step in range(max_new_tokens):
            step_start = time.time()

            # 输入：首次处理完整 prompt，后续只处理新 token（KV Cache 优化）
            if step == 0:
                current_input_ids = input_ids
                current_attention_mask = attention_mask
            else:
                # 优化：复用 tensor，避免重复创建，使用 long 类型
                current_input_ids = torch.tensor([[next_token_id]], device=device, dtype=torch.long)
                current_attention_mask = None  # 单 token 不需要 attention_mask

            # 前向传播（带 KV Cache）- 使用 Hugging Face 官方优化
            # KV Cache 工作原理：
            # - step=0: 处理完整 prompt，生成 KV cache
            # - step>0: 只处理新 token，复用之前的 KV cache，避免重复计算
            # 
            # Hugging Face 优化参数：
            # - use_cache=True: 启用 KV cache（默认已启用，显式指定确保优化）
            # - output_attentions=False: 不输出注意力权重（节省内存和计算）
            # - output_hidden_states=False: 不输出隐藏状态（节省内存）
            # - attention_mask: 仅在首次需要，后续单 token 不需要
            model_kwargs = {
                "past_key_values": past_key_values,
                "use_cache": True,  # 启用 KV cache 优化
                "output_attentions": False,  # 不输出注意力权重，节省内存
                "output_hidden_states": False,  # 不输出隐藏状态，节省内存
            }
            
            if current_attention_mask is not None:
                model_kwargs["attention_mask"] = current_attention_mask
            
            outputs = model(current_input_ids, **model_kwargs)

            logits = outputs.logits[0, -1, :]
            past_key_values = outputs.past_key_values

            # 优化采样过程
            if temperature > 0:
                # 温度采样
                logits = logits / temperature
                if top_k > 0:
                    # Top-k 采样：只保留 top-k 个最高概率的 token
                    # 优化：使用 torch.topk 一次性获取，避免多次计算
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    # 创建 mask，只保留 top-k
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered.scatter_(0, top_k_indices, top_k_logits)
                    logits = logits_filtered
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
            else:
                # 贪婪采样：直接取最大值（最快）
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

            # 记录每个 token 的详细信息到日志
            token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
            logger.info(f"Token #{len(generated_tokens)}: '{token_text}' (ID={next_token_id}) | "
                       f"Time={token_time*1000:.2f}ms")

            if next_token_id == tokenizer.eos_token_id:
                break

            # 实时输出（流式生成）
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

    # 记录最终统计到日志
    if generated_tokens:
        logger.info(f"✅ 生成完成: {len(generated_tokens)} tokens in {total_time:.2f}s "
                   f"({len(generated_tokens)/total_time:.2f} tokens/s)")
        if token_times:
            avg_latency = sum(token_times) / len(token_times)
            logger.info(f"   平均延迟: {avg_latency:.2f}ms/token, "
                       f"最小: {min(token_times):.2f}ms, 最大: {max(token_times):.2f}ms")
        
        stats_text = f"""✅ 生成完成

总Token数: {len(generated_tokens)}
总时间: {total_time:.2f}s
平均速度: {len(generated_tokens)/total_time:.2f} tokens/s
平均延迟: {sum(token_times)/len(token_times):.2f}ms/token

最小延迟: {min(token_times):.2f}ms
最大延迟: {max(token_times):.2f}ms
"""
    else:
        logger.warning("⚠ 未生成任何 token")
        stats_text = "⚠ 未生成任何 token（可能遇到 EOS 或错误）"
        final_text = ""

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

        with gr.Row():
            generate_btn = gr.Button("🚀 开始生成", variant="primary", size="lg")
            test_btn = gr.Button("🧪 测试连接", variant="secondary", size="lg")
        
        # 测试输出组件（必须在按钮之后定义，在生成输出之前）
        test_output = gr.Textbox(label="测试结果", lines=2, interactive=False, visible=True)
        
        def test_connection():
            """测试函数，验证 Gradio 是否能正常调用后端函数"""
            logger.info("=" * 70)
            logger.info("[TEST] ========== 测试按钮被点击！==========")
            logger.info("[TEST] 如果看到这条日志，说明 Gradio 可以正常调用后端函数")
            logger.info("=" * 70)
            return "✅ 测试成功！后端函数可以正常调用。如果看到这条消息，说明 Gradio 连接正常。"

        with gr.Row():
            with gr.Column(scale=3):
                output_text = gr.Textbox(label="生成结果", lines=12, interactive=False)

            with gr.Column(scale=1):
                stats_display = gr.Textbox(label="生成统计", lines=12, interactive=False)

        # 按钮事件 - 直接使用函数，与客户端保持一致
        # 添加调试日志，确认按钮绑定
        logger.info("=" * 70)
        logger.info("[UI] 正在绑定按钮事件...")
        logger.info(f"[UI] 生成按钮: fn={generate_with_kv_cache.__name__}")
        logger.info(f"[UI] 输入组件: prompt_input, max_tokens, temperature, top_k")
        logger.info(f"[UI] 输出组件: output_text, stats_display")
        
        generate_btn.click(
            fn=generate_with_kv_cache,
            inputs=[prompt_input, max_tokens, temperature, top_k],
            outputs=[output_text, stats_display],
        )
        
        logger.info("[UI] ✓ 按钮事件绑定完成")
        logger.info("=" * 70)
        
        # 测试按钮事件
        test_btn.click(
            fn=test_connection,
            inputs=[],
            outputs=[test_output],
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
    # ========================================================================
    # 关键诊断：确认按钮绑定是否已执行
    # ========================================================================
    logger.info("=" * 70)
    logger.info("[启动检查] 开始启动 Gradio 服务器...")
    logger.info(f"[启动检查] demo 对象: {demo}")
    logger.info(f"[启动检查] 按钮对象: generate_btn={generate_btn if 'generate_btn' in globals() else '未定义'}")
    logger.info("=" * 70)
    logger.info("\n启动 Gradio 界面...")
    
    # ========================================================================
    # 重要：启用队列以支持并发请求和更好的错误处理
    # ========================================================================
    logger.info("[UI] 启用 Gradio 队列...")
    demo.queue(
        max_size=10,  # 最大队列长度
        default_concurrency_limit=1,  # 并发限制（生成任务通常串行执行）
    )
    logger.info("[UI] ✓ 队列已启用")
    
    # 预先输出地址信息（因为 launch() 是阻塞的）
    expected_local_url = f"http://127.0.0.1:{FULL_PORT}/"
    logger.info("=" * 70)
    logger.info(f"预期本地地址: {expected_local_url}")
    if SHARE_ENABLED:
        logger.info("公网分享: 将自动创建分享链接")
    logger.info("=" * 70)
    logger.info("")
    logger.info(
        f"启动 Gradio: share={SHARE_ENABLED}, server_name=127.0.0.1, "
        f"port={FULL_PORT}"
    )
    
    local_url = share_url = None
    try:
        logger.info("正在启动 Gradio 服务器...")
        logger.info("提示: demo.launch() 是阻塞调用，服务器启动后地址信息会在控制台显示")
        
        # 直接启动，Gradio 会在控制台打印地址信息
        # 注意：launch() 是阻塞的，不会返回，除非服务器关闭
        launch_result = demo.launch(
            share=SHARE_ENABLED,  # 可通过环境变量 GRADIO_SHARE 控制
            server_name="127.0.0.1",  # 使用 127.0.0.1 避免 localhost 访问问题
            server_port=FULL_PORT,
            show_error=True,
            inbrowser=SHARE_ENABLED,
        )
        # 注意：如果 launch() 返回了（通常不会），处理返回值
        if launch_result:
            if isinstance(launch_result, tuple) and len(launch_result) >= 3:
                _, local_url, share_url = launch_result
            else:
                local_url = f"http://127.0.0.1:{FULL_PORT}/"
    except ValueError as e:
        # localhost 不可访问时，自动启用 share
        if "localhost" in str(e).lower() or "share" in str(e).lower():
            logger.warning(f"localhost 不可访问，自动启用公网分享: {e}")
            try:
                launch_result = demo.launch(
                    share=True,
                    server_name="127.0.0.1",
                    server_port=FULL_PORT,
                    show_error=True,
                    inbrowser=True,
                )
                if isinstance(launch_result, tuple) and len(launch_result) >= 3:
                    _, local_url, share_url = launch_result
                else:
                    local_url = f"http://127.0.0.1:{FULL_PORT}/"
            except Exception as e2:
                logger.error(f"启动公网分享失败: {e2}")
                local_url = f"http://127.0.0.1:{FULL_PORT}/"
        else:
            raise
    except Exception as e:
        logger.error(f"启动 Gradio 失败: {e}", exc_info=True)
        local_url = f"http://127.0.0.1:{FULL_PORT}/"
    
    logger.info("=" * 70)
    logger.info(f"✓ Gradio 本地地址: {local_url}")
    logger.info(f"✓ Gradio 分享链接: {share_url if share_url else '未开启或创建失败'}")
    logger.info("=" * 70)
    logger.info("提示: 如果点击生成按钮后没有日志，请确认:")
    logger.info("  1. 访问的是正确的地址（上面显示的本地地址或分享链接）")
    logger.info("  2. 页面已完全加载（没有 502 错误）")
    logger.info("  3. 浏览器控制台没有 JavaScript 错误")
    logger.info("=" * 70)
