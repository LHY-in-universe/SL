"""
Server-side Gradio UI for monitoring and analytics
"""

import time
import threading
from typing import Any, Optional
from datetime import datetime

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

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

    def _get_logs_text(self, level_filter: str = "ALL") -> str:
        """Get formatted logs as text"""
        try:
            logs = self.servicer.get_logs(
                level_filter=level_filter if level_filter != "ALL" else None,
                limit=100
            )

            if not logs:
                return f"No {level_filter} logs available"

            # Format logs
            formatted_lines = []
            for log in logs:
                timestamp_str = log['timestamp'].strftime("%H:%M:%S.%f")[:-3]
                level = log['level']
                message = log['message']

                # Add level emoji
                level_icon = {
                    'DEBUG': '🔍',
                    'INFO': 'ℹ️',
                    'WARNING': '⚠️',
                    'ERROR': '❌'
                }.get(level, '📝')

                line = f"[{timestamp_str}] {level_icon} {level:8} {message}"
                formatted_lines.append(line)

            return "\n".join(formatted_lines)

        except Exception as e:
            return f"❌ Error fetching logs: {str(e)}"

    def _get_latency_stats_dataframe(self) -> pd.DataFrame:
        """Get latency statistics as DataFrame"""
        try:
            metrics = self.servicer.get_metrics()
            stats = metrics.get("latency_stats", {})

            if stats.get('count', 0) == 0:
                return pd.DataFrame({
                    "Metric": ["No data"],
                    "Value": [""]
                })

            data = {
                "Metric": [
                    "Count",
                    "Mean",
                    "Median (P50)",
                    "P95",
                    "P99",
                    "Min",
                    "Max",
                    "Std Dev"
                ],
                "Value": [
                    f"{stats['count']} requests",
                    f"{stats['mean']:.2f} ms",
                    f"{stats['median']:.2f} ms",
                    f"{stats['p95']:.2f} ms",
                    f"{stats['p99']:.2f} ms",
                    f"{stats['min']:.2f} ms",
                    f"{stats['max']:.2f} ms",
                    f"{stats['std']:.2f} ms"
                ]
            }

            return pd.DataFrame(data)

        except Exception as e:
            return pd.DataFrame({
                "Error": [str(e)]
            })

    def _get_latency_distribution_plot(self):
        """Get latency distribution histogram"""
        try:
            metrics = self.servicer.get_metrics()
            dist = metrics.get("latency_distribution", {})

            bin_edges = dist.get("bin_edges", [])
            counts = dist.get("counts", [])

            if not bin_edges or not counts:
                # Return empty plot
                fig = go.Figure()
                fig.add_annotation(
                    text="No latency data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16)
                )
                fig.update_layout(
                    title="Latency Distribution",
                    xaxis_title="Latency (ms)",
                    yaxis_title="Count"
                )
                return fig

            # Calculate bin centers
            bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(counts))]

            fig = go.Figure(data=[
                go.Bar(
                    x=bin_centers,
                    y=counts,
                    name="Latency",
                    marker_color='rgb(55, 126, 184)'
                )
            ])

            fig.update_layout(
                title="Latency Distribution",
                xaxis_title="Latency (ms)",
                yaxis_title="Count",
                showlegend=False,
                height=300
            )

            return fig

        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

    def _get_latency_trend_plot(self):
        """Get latency trend over time"""
        try:
            metrics = self.servicer.get_metrics()
            history = metrics.get("latency_history", [])

            if not history:
                fig = go.Figure()
                fig.add_annotation(
                    text="No latency history available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16)
                )
                fig.update_layout(
                    title="Latency Trend",
                    xaxis_title="Request",
                    yaxis_title="Latency (ms)"
                )
                return fig

            timestamps = [h['timestamp'] for h in history]
            latencies = [h['latency_ms'] for h in history]

            fig = go.Figure(data=[
                go.Scatter(
                    x=list(range(len(latencies))),
                    y=latencies,
                    mode='lines+markers',
                    name="Latency",
                    line=dict(color='rgb(231, 99, 250)', width=2),
                    marker=dict(size=4)
                )
            ])

            fig.update_layout(
                title="Latency Trend (Last 50 Requests)",
                xaxis_title="Request Index",
                yaxis_title="Latency (ms)",
                showlegend=False,
                height=300
            )

            return fig

        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

    def _build_interface(self) -> gr.Blocks:
        """Build the Gradio monitoring interface"""

        with gr.Blocks(
            title="Split Learning Server Monitoring",
            theme=self.theme,
            css=DEFAULT_CSS
        ) as demo:

            gr.Markdown(
                """
                # 📊 Split Learning 服务器综合监控面板

                实时监控服务器性能、日志、延迟分析和请求统计
                """
            )

            # Auto-refresh notice
            gr.Markdown(
                f"🔄 **自动刷新**: 每 {self.refresh_interval} 秒更新一次数据"
            )

            # Tabs for different monitoring views
            with gr.Tabs():
                # Tab 1: Overview (existing functionality)
                with gr.Tab("📈 概览"):
                    # Statistics Panel
                    with gr.Group():
                        gr.Markdown("### 服务器统计")
                        stats_box = gr.Textbox(
                            label="实时统计",
                            value=self._get_stats_text(),
                            lines=8,
                            elem_classes=["stats-box"],
                            interactive=False
                        )

                    # Compute Time Graph
                    with gr.Group():
                        gr.Markdown("### 计算时间趋势")
                        compute_time_plot = gr.LinePlot(
                            value=self._get_compute_time_data(),
                            x="Request ID",
                            y="Compute Time (ms)",
                            title="Recent Compute Times",
                            width=800,
                            height=300
                        )

                    # Request History Table
                    with gr.Group():
                        gr.Markdown("### 请求历史")
                        history_table = gr.Dataframe(
                            value=self._get_request_history_dataframe(),
                            label="最近的请求",
                            max_rows=20,
                            interactive=False
                        )

                # Tab 2: Real-time Logs
                with gr.Tab("📝 实时日志"):
                    gr.Markdown("### 服务器日志")

                    with gr.Row():
                        log_level_filter = gr.Dropdown(
                            choices=["ALL", "DEBUG", "INFO", "WARNING", "ERROR"],
                            value="INFO",
                            label="日志级别",
                            scale=1
                        )

                    logs_display = gr.Textbox(
                        label="日志输出",
                        value=self._get_logs_text("INFO"),
                        lines=25,
                        max_lines=25,
                        interactive=False,
                        elem_classes=["stats-box"]
                    )

                    with gr.Accordion("📖 日志说明", open=False):
                        gr.Markdown(
                            """
                            ## 日志级别说明

                            - 🔍 **DEBUG**: 调试信息
                            - ℹ️ **INFO**: 一般信息（默认）
                            - ⚠️ **WARNING**: 警告信息
                            - ❌ **ERROR**: 错误信息

                            ## 功能

                            - 自动刷新最新日志
                            - 支持按级别过滤
                            - 显示最近 100 条日志
                            """
                        )

                # Tab 3: Detailed Latency Analysis
                with gr.Tab("⏱️ 延迟分析"):
                    gr.Markdown("### 详细延迟统计与分析")

                    # Latency Statistics Table
                    with gr.Group():
                        gr.Markdown("#### 延迟统计")
                        latency_stats_table = gr.Dataframe(
                            value=self._get_latency_stats_dataframe(),
                            label="统计指标",
                            interactive=False
                        )

                    # Latency Distribution Histogram
                    with gr.Group():
                        gr.Markdown("#### 延迟分布直方图")
                        latency_dist_plot = gr.Plot(
                            value=self._get_latency_distribution_plot(),
                            label="分布图"
                        )

                    # Latency Trend
                    with gr.Group():
                        gr.Markdown("#### 延迟趋势")
                        latency_trend_plot = gr.Plot(
                            value=self._get_latency_trend_plot(),
                            label="趋势图"
                        )

                    with gr.Accordion("📖 延迟分析说明", open=False):
                        gr.Markdown(
                            """
                            ## 统计指标说明

                            - **Mean**: 平均延迟
                            - **Median (P50)**: 中位数延迟，50% 的请求延迟低于此值
                            - **P95**: 95% 的请求延迟低于此值
                            - **P99**: 99% 的请求延迟低于此值
                            - **Min/Max**: 最小/最大延迟
                            - **Std Dev**: 标准差，表示延迟波动程度

                            ## 图表说明

                            - **分布直方图**: 显示延迟的分布情况
                            - **趋势图**: 显示最近 50 个请求的延迟变化
                            """
                        )

            # Refresh button (global)
            gr.Markdown("---")
            with gr.Row():
                refresh_btn = gr.Button("🔄 手动刷新所有数据", variant="primary")

            # Usage instructions
            with gr.Accordion("📖 使用说明", open=False):
                gr.Markdown(
                    """
                    ## 监控面板说明

                    本监控面板包含 3 个标签页：

                    1. **概览**: 服务器统计、计算时间趋势、请求历史
                    2. **实时日志**: 查看服务器实时日志，支持级别过滤
                    3. **延迟分析**: 详细的延迟统计（P95/P99）、分布图和趋势图

                    ## 自动刷新

                    所有数据每 {0} 秒自动更新，也可点击"手动刷新"按钮立即更新。
                    """.format(self.refresh_interval)
                )

            # Event bindings - log level filter
            log_level_filter.change(
                fn=self._get_logs_text,
                inputs=[log_level_filter],
                outputs=[logs_display]
            )

            # Event bindings - manual refresh all
            refresh_btn.click(
                fn=lambda level: (
                    self._get_stats_text(),
                    self._get_compute_time_data(),
                    self._get_request_history_dataframe(),
                    self._get_logs_text(level),
                    self._get_latency_stats_dataframe(),
                    self._get_latency_distribution_plot(),
                    self._get_latency_trend_plot()
                ),
                inputs=[log_level_filter],
                outputs=[
                    stats_box,
                    compute_time_plot,
                    history_table,
                    logs_display,
                    latency_stats_table,
                    latency_dist_plot,
                    latency_trend_plot
                ]
            )

            # Auto-refresh using demo.load
            demo.load(
                fn=lambda level: (
                    self._get_stats_text(),
                    self._get_compute_time_data(),
                    self._get_request_history_dataframe(),
                    self._get_logs_text(level),
                    self._get_latency_stats_dataframe(),
                    self._get_latency_distribution_plot(),
                    self._get_latency_trend_plot()
                ),
                inputs=[log_level_filter],
                outputs=[
                    stats_box,
                    compute_time_plot,
                    history_table,
                    logs_display,
                    latency_stats_table,
                    latency_dist_plot,
                    latency_trend_plot
                ],
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
