"""
带 Gradio 界面的 Split Learning 客户端 (改进版)
"""
import torch
import gradio as gr
import time
import os
import sys
from transformers import AutoTokenizer

# 添加路径以便导入本地模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'SplitLearning', 'src'))
sys.path.append(os.path.join(project_root, 'splitlearn-comm', 'src'))

from splitlearn_comm import GRPCComputeClient

# 全局变量
bottom_model = None
top_model = None
tokenizer = None
client = None
stop_generation = False  # 停止生成标志

def load_models():
    """加载本地模型和连接服务器"""
    global bottom_model, top_model, tokenizer, client

    try:
        # 检查是否已经有缓存的模型文件
        bottom_path = os.path.join(current_dir, "gpt2_bottom_cached.pt")
        top_path = os.path.join(current_dir, "gpt2_top_cached.pt")

        if os.path.exists(bottom_path) and os.path.exists(top_path):
            print("使用缓存的模型文件...")
            bottom_model = torch.load(bottom_path, map_location='cpu', weights_only=False)
            top_model = torch.load(top_path, map_location='cpu', weights_only=False)
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            print("模型加载完成")
        else:
            # 需要下载并拆分
            from splitlearn import ModelFactory
            print("首次运行：正在下载并拆分 GPT-2 模型...")
            print("这可能需要几分钟，请耐心等待...")
            bottom, trunk, top = ModelFactory.create_split_models(
                model_type='gpt2',
                model_name_or_path='gpt2',
                split_point_1=2,
                split_point_2=10,
                device='cpu'
            )
            bottom_model = bottom
            top_model = top
            tokenizer = AutoTokenizer.from_pretrained('gpt2')

            # 保存以便下次使用
            torch.save(bottom_model, bottom_path)
            torch.save(top_model, top_path)
            print("模型已缓存到本地")

        # 连接服务器
        print("正在连接服务器 192.168.0.144:50053...")
        client = GRPCComputeClient("192.168.0.144:50053", timeout=20.0)
        if not client.connect():
            return "❌ 无法连接到服务器！\n请确保服务器正在运行：\n  python testcode/start_server.py\n\n(尝试了连接 127.0.0.1:50053)\n\n或运行准备脚本：\n  python testcode/prepare_models.py"

        return "✅ 初始化成功！\n\n✓ Bottom 模型已加载\n✓ Top 模型已加载\n✓ 服务器已连接 (localhost:50053)\n\n现在可以开始生成文本了！"
    except Exception as e:
        import traceback
        error_msg = f"❌ 初始化失败:\n\n{str(e)}\n\n详细错误:\n{traceback.format_exc()}"
        return error_msg


def check_models_loaded():
    """检查模型是否已加载"""
    if bottom_model is None or top_model is None or tokenizer is None:
        return False, "⚠️ 请先点击'初始化模型并连接服务器'按钮！"
    if client is None or not client.channel:
        return False, "⚠️ 服务器未连接！请重新初始化。"
    return True, ""


def stop_generation_fn():
    """停止生成"""
    global stop_generation
    stop_generation = True
    return "🛑 正在停止生成..."


def generate_text(prompt, max_length=20, temperature=1.0, top_k=50, show_speed=True):
    """生成文本的核心逻辑（改进版）"""
    global stop_generation
    stop_generation = False

    # 检查模型和服务器状态
    is_ready, error_msg = check_models_loaded()
    if not is_ready:
        yield error_msg, ""
        return

    # 验证输入
    if not prompt or len(prompt.strip()) == 0:
        yield "⚠️ 请输入一个有效的 prompt！", ""
        return

    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        generated_text = prompt
        start_time = time.time()
        tokens_generated = 0

        yield generated_text, "🔄 生成中..."

        for step in range(max_length):
            if stop_generation:
                elapsed_time = time.time() - start_time
                avg_speed = tokens_generated / elapsed_time if elapsed_time > 0 else 0
                stats = f"🛑 生成已停止\n\n生成了 {tokens_generated} 个 tokens\n平均速度: {avg_speed:.2f} tokens/s"
                yield generated_text, stats
                break

            # 1. Bottom (本地)
            with torch.no_grad():
                hidden_bottom = bottom_model(input_ids)

            # 2. Trunk (远程服务器)
            try:
                hidden_trunk = client.compute(hidden_bottom, model_id="gpt2-trunk")
            except Exception as e:
                error_stats = f"❌ 服务器通信错误\n\n{str(e)}\n\n已生成 {tokens_generated} tokens"
                yield generated_text + f"\n\n[错误: {str(e)}]", error_stats
                break

            # 3. Top (本地)
            with torch.no_grad():
                output = top_model(hidden_trunk)
                logits = output.logits[:, -1, :] / temperature

            # 4. Sampling (改进的采样策略)
            if top_k > 0:
                # Top-K 采样
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                probs = torch.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1)
                next_token_id = top_k_indices.gather(-1, next_token_idx)
            else:
                # Greedy 采样
                next_token_id = logits.argmax(dim=-1).unsqueeze(-1)

            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            tokens_generated += 1

            # Decode and update UI
            new_word = tokenizer.decode(next_token_id[0])
            generated_text += new_word

            # 计算统计信息
            elapsed_time = time.time() - start_time
            avg_speed = tokens_generated / elapsed_time if elapsed_time > 0 else 0

            if show_speed:
                stats = f"⏱️ 生成统计\n\nTokens: {tokens_generated}/{max_length}\n速度: {avg_speed:.2f} tokens/s\n耗时: {elapsed_time:.2f}s"
            else:
                stats = f"Tokens: {tokens_generated}/{max_length}"

            yield generated_text, stats

            # 检查是否生成了结束标记
            if tokenizer.eos_token_id and next_token_id.item() == tokenizer.eos_token_id:
                final_stats = f"✅ 生成完成 (遇到 EOS)\n\n生成了 {tokens_generated} 个 tokens\n平均速度: {avg_speed:.2f} tokens/s\n总耗时: {elapsed_time:.2f}s"
                yield generated_text, final_stats
                break

            time.sleep(0.02)  # 减少延迟以提升体验
        else:
            # 正常完成
            elapsed_time = time.time() - start_time
            avg_speed = tokens_generated / elapsed_time if elapsed_time > 0 else 0
            final_stats = f"✅ 生成完成\n\n生成了 {tokens_generated} 个 tokens\n平均速度: {avg_speed:.2f} tokens/s\n总耗时: {elapsed_time:.2f}s"
            yield generated_text, final_stats

    except Exception as e:
        import traceback
        error_msg = f"❌ 生成过程出错:\n\n{str(e)}\n\n{traceback.format_exc()}"
        yield generated_text, error_msg


# 创建 Gradio 界面 (改进版)
with gr.Blocks(
    title="Split Learning Demo",
    theme=gr.themes.Soft(),
    css="""
    .status-box {font-family: monospace; font-size: 14px;}
    .stats-box {font-family: monospace; font-size: 12px;}
    """
) as demo:

    gr.Markdown(
        """
        # 🚀 Split Learning 分布式推理演示

        **架构**: Bottom(本地) → Trunk(远程服务器) → Top(本地)

        这个演示展示了如何将大型语言模型拆分到多个设备上进行推理。
        """
    )

    # 初始化区域
    with gr.Group():
        gr.Markdown("### 1️⃣ 初始化")
        status_box = gr.Textbox(
            label="系统状态",
            value="⚪ 未初始化 - 请点击下方按钮开始",
            lines=6,
            elem_classes=["status-box"]
        )
        init_btn = gr.Button("🔌 初始化模型并连接服务器", variant="primary", size="lg")

    gr.Markdown("---")

    # 生成区域
    with gr.Group():
        gr.Markdown("### 2️⃣ 文本生成")

        with gr.Row():
            with gr.Column(scale=3):
                input_box = gr.Textbox(
                    label="输入 Prompt",
                    placeholder="例如: The future of AI is",
                    lines=3
                )

            with gr.Column(scale=2):
                gr.Markdown("**生成参数**")
                max_length_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=20,
                    step=1,
                    label="最大生成长度 (tokens)"
                )
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Temperature (创造性)"
                )
                top_k_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top-K (0=贪婪采样)"
                )
                show_speed_check = gr.Checkbox(
                    label="显示生成速度",
                    value=True
                )

        with gr.Row():
            generate_btn = gr.Button("▶️ 开始生成", variant="primary", size="lg")
            stop_btn = gr.Button("⏹️ 停止生成", variant="stop", size="lg")

        with gr.Row():
            with gr.Column(scale=3):
                output_box = gr.Textbox(
                    label="生成结果",
                    lines=8,
                    show_copy_button=True
                )
            with gr.Column(scale=1):
                stats_box = gr.Textbox(
                    label="生成统计",
                    lines=8,
                    elem_classes=["stats-box"]
                )

    # 示例
    gr.Markdown("### 💡 示例 Prompts")
    gr.Examples(
        examples=[
            ["The future of AI is"],
            ["Once upon a time, in a land far away,"],
            ["To be or not to be,"],
            ["In the beginning, there was"],
            ["The quick brown fox"],
        ],
        inputs=input_box,
        label="点击使用示例"
    )

    # 使用说明
    with gr.Accordion("📖 使用说明", open=False):
        gr.Markdown(
            """
            1. **初始化**: 点击"初始化模型并连接服务器"按钮
               - 首次运行会下载 GPT-2 模型（约 500MB）
               - 后续运行使用缓存，启动更快

            2. **生成文本**:
               - 输入一个 prompt
               - 调整生成参数（可选）
               - 点击"开始生成"

            3. **停止生成**: 点击"停止生成"按钮可随时中止

            4. **参数说明**:
               - **最大长度**: 生成的 token 数量
               - **Temperature**: 控制随机性（越高越随机）
               - **Top-K**: 限制采样范围（0=贪婪采样）

            5. **服务器要求**:
               - 确保运行了 `python testcode/start_server.py`
               - 服务器地址: localhost:50053
            """
        )

    # 事件绑定
    init_btn.click(
        fn=load_models,
        inputs=[],
        outputs=status_box
    )

    generate_btn.click(
        fn=generate_text,
        inputs=[
            input_box,
            max_length_slider,
            temperature_slider,
            top_k_slider,
            show_speed_check
        ],
        outputs=[output_box, stats_box],
        show_progress="full",  # 显示进度
        api_name="generate"    # 允许 API 调用
    )

    stop_btn.click(
        fn=stop_generation_fn,
        inputs=[],
        outputs=stats_box
    )

if __name__ == "__main__":
    # 从环境变量读取配置
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    server_port = int(os.environ.get("GRADIO_PORT", "7861"))

    print("=" * 70)
    print("Split Learning Gradio 客户端")
    print("=" * 70)
    print(f"启动地址: http://127.0.0.1:{server_port}")
    print(f"Share 模式: {'启用' if share else '禁用'}")
    print("=" * 70)

    demo.queue()  # 启用队列以支持流式输出
    demo.launch(
        server_name="0.0.0.0",  # 监听所有网络接口
        server_port=7870,
        share=False,  # 禁用公共链接，只使用本地访问
        show_error=True,
        inbrowser=True  # 自动打开浏览器
    )
