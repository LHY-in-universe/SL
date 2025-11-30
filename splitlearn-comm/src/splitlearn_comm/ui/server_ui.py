"""
Server-side Gradio UI for monitoring and analytics
"""

import time
import threading
from typing import Any, Optional
from datetime import datetime

import gradio as gr
import pandas as pd

from .components import get_theme, DEFAULT_CSS, StatsPanel


class ServerMonitoringUI:
    """
    Gradio UI for server-side monitoring and analytics

    Provides a real-time dashboard for monitoring server metrics including
    request statistics, compute times, success rates, and request history.

    Example:
        >>> from splitlearn_comm import GRPCComputeServer
        >>> from splitlearn_comm.ui import ServerMonitoringUI
        >>>
        >>> server = GRPCComputeServer(compute_fn, port=50051)
        >>> server.start()
        >>>
        >>> ui = ServerMonitoringUI(servicer=server.servicer)
        >>> ui.launch(share=False, server_port=7861)
    """

    def __init__(
        self,
        servicer: Any,
        theme: str = "default",
        refresh_interval: int = 2,
    ):
        """
        Args:
            servicer: ComputeServicer instance (must be running)
            theme: UI theme variant ("default", "dark", "light")
            refresh_interval: Dashboard refresh interval in seconds
        """
        self.servicer = servicer
        self.theme = get_theme(theme)
        self.refresh_interval = refresh_interval

        # Build the interface
        self.demo = self._build_interface()

    def _get_stats_text(self):
        """Get current statistics as formatted text"""
        try:
            metrics = self.servicer.get_metrics()

            stats_text = StatsPanel.format_server_stats(
                total_requests=metrics["total_requests"],
                success_rate=metrics["success_rate"],
                avg_compute_time=metrics["avg_compute_time_ms"],
                uptime_seconds=metrics["uptime_seconds"],
                active_connections=0  # Could be enhanced
            )

            return stats_text
        except Exception as e:
            return f"❌ 获取统计信息失败: {str(e)}"

    def _get_request_history_dataframe(self):
        """Get request history as a pandas DataFrame"""
        try:
            metrics = self.servicer.get_metrics()
            history = metrics["request_history"]

            if not history:
                return pd.DataFrame({
                    "Request ID": [],
                    "Timestamp": [],
                    "Status": [],
                    "Compute Time (ms)": [],
                    "Error": []
                })

            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    "Request ID": req["request_id"],
                    "Timestamp": req["timestamp"].strftime("%H:%M:%S"),
                    "Status": "✅ Success" if req["success"] else "❌ Failed",
                    "Compute Time (ms)": f"{req['compute_time_ms']:.2f}",
                    "Error": req["error"] or ""
                }
                for req in history
            ])

            # Sort by Request ID descending (most recent first)
            df = df.sort_values("Request ID", ascending=False)

            return df

        except Exception as e:
            return pd.DataFrame({
                "Error": [str(e)]
            })

    def _get_compute_time_data(self):
        """Get compute time data for plotting"""
        try:
            metrics = self.servicer.get_metrics()
            history = metrics["request_history"]

            if not history:
                return pd.DataFrame({
                    "Request ID": [],
                    "Compute Time (ms)": []
                })

            # Get last 50 requests for plotting
            recent = list(history)[-50:]

            df = pd.DataFrame([
                {
                    "Request ID": req["request_id"],
                    "Compute Time (ms)": req["compute_time_ms"]
                }
                for req in recent
            ])

            return df

        except Exception as e:
            return pd.DataFrame({
                "Error": [str(e)]
            })

    def _build_interface(self) -> gr.Blocks:
        """Build the Gradio monitoring interface"""

        with gr.Blocks(
            title="Split Learning Server Monitoring",
            theme=self.theme,
            css=DEFAULT_CSS
        ) as demo:

            gr.Markdown(
                """
                # 📊 Split Learning 服务器监控

                实时监控服务器性能、请求统计和计算时间
                """
            )

            # Auto-refresh notice
            gr.Markdown(
                f"🔄 **自动刷新**: 每 {self.refresh_interval} 秒更新一次数据"
            )

            # Statistics Panel
            with gr.Group():
                gr.Markdown("### 📈 服务器统计")
                stats_box = gr.Textbox(
                    label="实时统计",
                    value=self._get_stats_text(),
                    lines=8,
                    elem_classes=["stats-box"],
                    interactive=False
                )

            gr.Markdown("---")

            # Compute Time Graph
            with gr.Group():
                gr.Markdown("### 📉 计算时间趋势")
                compute_time_plot = gr.LinePlot(
                    value=self._get_compute_time_data(),
                    x="Request ID",
                    y="Compute Time (ms)",
                    title="Recent Compute Times",
                    width=800,
                    height=300
                )

            gr.Markdown("---")

            # Request History Table
            with gr.Group():
                gr.Markdown("### 📋 请求历史")
                history_table = gr.Dataframe(
                    value=self._get_request_history_dataframe(),
                    label="最近的请求",
                    max_rows=20,
                    interactive=False
                )

            # Refresh button
            with gr.Row():
                refresh_btn = gr.Button("🔄 手动刷新", variant="primary")

            # Usage instructions
            with gr.Accordion("📖 使用说明", open=False):
                gr.Markdown(
                    """
                    ## 监控指标说明

                    - **总请求数**: 服务器处理的所有请求数量
                    - **成功率**: 成功处理的请求百分比
                    - **平均计算时间**: 每个请求的平均处理时间（毫秒）
                    - **运行时间**: 服务器启动后的运行时长

                    ## 图表说明

                    - **计算时间趋势**: 显示最近 50 个请求的计算时间变化
                    - **请求历史**: 显示最近 100 个请求的详细信息

                    ## 自动刷新

                    监控界面会每隔几秒自动更新，无需手动刷新。
                    """
                )

            # Event bindings - manual refresh
            refresh_btn.click(
                fn=lambda: (
                    self._get_stats_text(),
                    self._get_compute_time_data(),
                    self._get_request_history_dataframe()
                ),
                inputs=[],
                outputs=[stats_box, compute_time_plot, history_table]
            )

            # Auto-refresh using demo.load
            demo.load(
                fn=lambda: (
                    self._get_stats_text(),
                    self._get_compute_time_data(),
                    self._get_request_history_dataframe()
                ),
                inputs=[],
                outputs=[stats_box, compute_time_plot, history_table],
                every=self.refresh_interval
            )

        return demo

    def launch(
        self,
        share: bool = False,
        server_name: str = "127.0.0.1",
        server_port: int = 7861,
        inbrowser: bool = True,
        blocking: bool = True,
        **kwargs
    ):
        """
        Launch the monitoring UI

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
