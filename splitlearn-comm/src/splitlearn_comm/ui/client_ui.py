"""
Client-side Gradio UI for interactive text generation with Split Learning
"""

import time
import threading
from typing import Any, Optional

import gradio as gr
import torch

from .components import get_theme, DEFAULT_CSS, StatsPanel


class ClientUI:
    """
    Gradio UI for client-side text generation

    Provides an interactive web interface for text generation using a split learning
    architecture where the model is distributed across client and server.

    Example:
        >>> from splitlearn_comm import GRPCComputeClient
        >>> from splitlearn_comm.ui import ClientUI
        >>>
        >>> client = GRPCComputeClient("localhost:50051")
        >>> client.connect()
        >>>
        >>> ui = ClientUI(
        ...     client=client,
        ...     bottom_model=bottom_model,
        ...     top_model=top_model,
        ...     tokenizer=tokenizer
        ... )
        >>> ui.launch(share=False, server_port=7860)
    """

    def __init__(
        self,
        client: Any,
        bottom_model: Any,
        top_model: Any,
        tokenizer: Any,
        theme: str = "default",
        model_id: Optional[str] = None,
    ):
        """
        Args:
            client: GRPCComputeClient instance (must be connected)
            bottom_model: Bottom part of the split model (runs locally)
            top_model: Top part of the split model (runs locally)
            tokenizer: Tokenizer for encoding/decoding text
            theme: UI theme variant ("default", "dark", "light")
            model_id: Optional model ID to send to server
        """
        self.client = client
        self.bottom_model = bottom_model
        self.top_model = top_model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.theme = get_theme(theme)

        # Generation control
        self.stop_generation = False

        # Build the interface
        self.demo = self._build_interface()

    def _stop_generation(self):
        """Stop the current generation"""
        self.stop_generation = True
        return "🛑 正在停止生成..."

    def _generate_text_streaming(
        self,
        prompt: str,
        max_length: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        show_speed: bool = True
    ):
        """
        Generate text with streaming output (generator function)

        Args:
            prompt: Input text prompt
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-K sampling parameter (0 = greedy)
            show_speed: Whether to show generation speed statistics

        Yields:
            Tuple of (generated_text, statistics_text)
        """
        self.stop_generation = False

        # Validate input
        if not prompt or len(prompt.strip()) == 0:
            yield "⚠️ 请输入一个有效的 prompt！", ""
            return

        try:
            # Encode input
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            generated_text = prompt
            start_time = time.time()
            tokens_generated = 0

            yield generated_text, "🔄 生成中..."

            # Generation loop
            for step in range(max_length):
                if self.stop_generation:
                    elapsed_time = time.time() - start_time
                    stats = StatsPanel.format_generation_stats(
                        tokens_generated, max_length, elapsed_time, status="stopped"
                    )
                    yield generated_text, stats
                    break

                # 1. Bottom model (local)
                with torch.no_grad():
                    hidden_bottom = self.bottom_model(input_ids)

                # 2. Trunk model (remote server)
                try:
                    hidden_trunk = self.client.compute(hidden_bottom, model_id=self.model_id)
                except Exception as e:
                    elapsed_time = time.time() - start_time
                    error_msg = f"❌ 服务器通信错误\n\n{str(e)}\n\n已生成 {tokens_generated} tokens"
                    yield generated_text + f"\n\n[错误: {str(e)}]", error_msg
                    break

                # 3. Top model (local)
                with torch.no_grad():
                    output = self.top_model(hidden_trunk)
                    logits = output.logits[:, -1, :] / temperature

                # 4. Sampling
                if top_k > 0:
                    # Top-K sampling
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    probs = torch.softmax(top_k_logits, dim=-1)
                    next_token_idx = torch.multinomial(probs, num_samples=1)
                    next_token_id = top_k_indices.gather(-1, next_token_idx)
                else:
                    # Greedy sampling
                    next_token_id = logits.argmax(dim=-1).unsqueeze(-1)

                # Update input_ids
                input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                tokens_generated += 1

                # Decode new token
                new_word = self.tokenizer.decode(next_token_id[0])
                generated_text += new_word

                # Calculate statistics
                elapsed_time = time.time() - start_time

                if show_speed:
                    stats = StatsPanel.format_generation_stats(
                        tokens_generated, max_length, elapsed_time, status="generating"
                    )
                else:
                    stats = f"Tokens: {tokens_generated}/{max_length}"

                yield generated_text, stats

                # Check for EOS token
                if self.tokenizer.eos_token_id and next_token_id.item() == self.tokenizer.eos_token_id:
                    stats = StatsPanel.format_generation_stats(
                        tokens_generated, max_length, elapsed_time, status="completed"
                    )
                    yield generated_text, stats
                    break

                time.sleep(0.02)  # Small delay for UI responsiveness

            else:
                # Normal completion (max_length reached)
                elapsed_time = time.time() - start_time
                stats = StatsPanel.format_generation_stats(
                    tokens_generated, max_length, elapsed_time, status="completed"
                )
                yield generated_text, stats

        except Exception as e:
            import traceback
            error_msg = f"❌ 生成过程出错:\n\n{str(e)}\n\n{traceback.format_exc()}"
            yield generated_text if 'generated_text' in locals() else prompt, error_msg

    def _build_interface(self) -> gr.Blocks:
        """Build the Gradio interface"""

        with gr.Blocks(
            title="Split Learning Client",
            theme=self.theme,
            css=DEFAULT_CSS
        ) as demo:

            gr.Markdown(
                """
                # 🚀 Split Learning 分布式推理客户端

                **架构**: Bottom(本地) → Trunk(远程服务器) → Top(本地)

                这个界面允许你使用分布式模型进行文本生成。
                """
            )

            # Generation area
            with gr.Group():
                gr.Markdown("### 📝 文本生成")

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
                            show_copy_button=True,
                            elem_classes=["output-box"]
                        )
                    with gr.Column(scale=1):
                        stats_box = gr.Textbox(
                            label="生成统计",
                            lines=8,
                            elem_classes=["stats-box"]
                        )

            # Examples
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

            # Usage instructions
            with gr.Accordion("📖 使用说明", open=False):
                gr.Markdown(
                    """
                    1. **输入文本**: 在输入框中输入你的 prompt
                    2. **调整参数** (可选):
                       - **最大长度**: 生成的 token 数量
                       - **Temperature**: 控制随机性（越高越随机）
                       - **Top-K**: 限制采样范围（0=贪婪采样）
                    3. **开始生成**: 点击"开始生成"按钮
                    4. **停止生成**: 点击"停止生成"按钮可随时中止

                    **注意**: 确保已连接到服务器并加载了模型。
                    """
                )

            # Event bindings
            generate_btn.click(
                fn=self._generate_text_streaming,
                inputs=[
                    input_box,
                    max_length_slider,
                    temperature_slider,
                    top_k_slider,
                    show_speed_check
                ],
                outputs=[output_box, stats_box],
                show_progress="full",
                api_name="generate"
            )

            stop_btn.click(
                fn=self._stop_generation,
                inputs=[],
                outputs=stats_box
            )

        return demo

    def launch(
        self,
        share: bool = False,
        server_name: str = "127.0.0.1",
        server_port: int = 7860,
        inbrowser: bool = True,
        blocking: bool = True,
        **kwargs
    ):
        """
        Launch the Gradio UI

        Args:
            share: Whether to create a public Gradio link
            server_name: Server hostname to bind to
            server_port: Server port to use
            inbrowser: Whether to automatically open in browser
            blocking: Whether to block the main thread (False = run in background)
            **kwargs: Additional arguments passed to demo.launch()

        Returns:
            If blocking=False, returns the running demo instance
        """
        self.demo.queue()  # Enable streaming support

        if blocking:
            self.demo.launch(
                share=share,
                server_name=server_name,
                server_port=server_port,
                inbrowser=inbrowser,
                show_error=True,
                **kwargs
            )
        else:
            # Run in background thread
            def _launch():
                self.demo.launch(
                    share=share,
                    server_name=server_name,
                    server_port=server_port,
                    inbrowser=inbrowser,
                    show_error=True,
                    **kwargs
                )

            thread = threading.Thread(target=_launch, daemon=True)
            thread.start()
            return self.demo
