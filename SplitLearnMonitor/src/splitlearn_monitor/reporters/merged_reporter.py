"""
Merged HTML Reporter - 合并客户端和服务端监控数据的报告生成器
"""
from typing import Optional, List, Dict, Any
import numpy as np

from .html_reporter import HTMLReporter
from ..core import SystemMonitor, PerformanceTracker
from ..config import ResourceSnapshot
from ..utils import calculate_percentile


class MergedHTMLReporter(HTMLReporter):
    """
    合并客户端和服务端监控数据的 HTML 报告生成器

    Example:
        >>> client_monitor = ClientMonitor()
        >>> # ... 运行客户端 ...
        >>> server_snapshots = client.get_server_monitoring_data()
        >>>
        >>> reporter = MergedHTMLReporter(
        ...     client_system_monitor=client_monitor.system_monitor,
        ...     client_performance_tracker=client_monitor.performance_tracker,
        ...     server_monitoring_snapshots=server_snapshots
        ... )
        >>> reporter.generate_report("merged_report.html")
    """

    def __init__(
        self,
        client_system_monitor: Optional[SystemMonitor] = None,
        client_performance_tracker: Optional[PerformanceTracker] = None,
        server_monitoring_snapshots: Optional[List[Dict[str, Any]]] = None
    ):
        """
        初始化合并报告生成器

        Args:
            client_system_monitor: 客户端系统监控器
            client_performance_tracker: 客户端性能追踪器
            server_monitoring_snapshots: 服务端监控快照列表（字典格式）
        """
        super().__init__(client_system_monitor, client_performance_tracker)

        # 转换服务端快照为 ResourceSnapshot 对象
        self.server_snapshots = []
        if server_monitoring_snapshots:
            for snap_dict in server_monitoring_snapshots:
                snapshot = ResourceSnapshot(
                    timestamp=snap_dict["timestamp"],
                    cpu_percent=snap_dict["cpu_percent"],
                    memory_mb=snap_dict["memory_mb"],
                    memory_percent=snap_dict["memory_percent"],
                    gpu_available=snap_dict["gpu_available"],
                    gpu_utilization=snap_dict.get("gpu_utilization"),
                    gpu_memory_used_mb=snap_dict.get("gpu_memory_used_mb"),
                    gpu_memory_total_mb=snap_dict.get("gpu_memory_total_mb"),
                )
                self.server_snapshots.append(snapshot)

    def _build_html(self, title: str, include_raw_data: bool) -> str:
        """构建完整的 HTML 文档"""
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    {self._get_css()}
</head>
<body>
    <div class="container">
        {self._get_header(title)}
        {self._get_summary_section()}
        {self._get_client_resource_section()}
        {self._get_server_resource_section()}
        {self._get_comparison_section()}
        {self._get_performance_section()}
        {self._get_footer()}
    </div>
</body>
</html>"""

    def _get_client_resource_section(self) -> str:
        """获取客户端资源监控部分"""
        if not self.system_monitor:
            return ""

        sections = ['<h2 style="color: #2c5aa0; margin-top: 40px;">📱 客户端资源监控</h2>']

        # 获取客户端资源历史
        snapshots = self.system_monitor.get_history()
        if not snapshots:
            sections.append("<p>无客户端资源数据。</p>")
            return ''.join(sections)

        # 生成时序图
        try:
            fig = self.time_series_viz.plot_resource_timeline(
                snapshots,
                title="客户端资源使用时间线"
            )
            img_data = self._fig_to_base64(fig)
            sections.append(f"""<div class="chart-container">
    <img src="data:image/png;base64,{img_data}" alt="客户端资源时间线" style="max-width: 100%;">
</div>""")
            self.time_series_viz.close_figure(fig)
        except Exception as e:
            sections.append(f"<p style='color: red;'>无法生成图表: {e}</p>")

        # 添加统计表格
        stats = self.system_monitor.get_statistics()
        sections.append(self._get_resource_stats_table(stats, "客户端"))

        return ''.join(sections)

    def _get_server_resource_section(self) -> str:
        """获取服务端资源监控部分"""
        if not self.server_snapshots:
            return ""

        sections = ['<h2 style="color: #c44e00; margin-top: 40px;">🖥️ 服务端资源监控</h2>']

        # 生成时序图
        try:
            fig = self.time_series_viz.plot_resource_timeline(
                self.server_snapshots,
                title="服务端资源使用时间线"
            )
            img_data = self._fig_to_base64(fig)
            sections.append(f"""<div class="chart-container">
    <img src="data:image/png;base64,{img_data}" alt="服务端资源时间线" style="max-width: 100%;">
</div>""")
            self.time_series_viz.close_figure(fig)
        except Exception as e:
            sections.append(f"<p style='color: red;'>无法生成图表: {e}</p>")

        # 计算服务端统计数据
        cpu_values = [s.cpu_percent for s in self.server_snapshots]
        memory_values = [s.memory_mb for s in self.server_snapshots]

        cpu_mean = float(np.mean(cpu_values))
        cpu_max = float(np.max(cpu_values))
        cpu_p95 = calculate_percentile(cpu_values, 95)

        memory_mean = float(np.mean(memory_values))
        memory_max = float(np.max(memory_values))
        memory_p95 = calculate_percentile(memory_values, 95)

        # GPU 数据
        gpu_available = self.server_snapshots[0].gpu_available if self.server_snapshots else False
        gpu_mean = None
        gpu_max = None
        gpu_p95 = None

        if gpu_available:
            gpu_values = [s.gpu_utilization for s in self.server_snapshots if s.gpu_utilization is not None]
            if gpu_values:
                gpu_mean = float(np.mean(gpu_values))
                gpu_max = float(np.max(gpu_values))
                gpu_p95 = calculate_percentile(gpu_values, 95)

        # 构建统计表格
        sections.append("""<h3 style="color: #555;">服务端资源统计</h3>
<table>
    <tr><th>指标</th><th>平均值</th><th>最大值</th><th>P95</th></tr>""")

        sections.append(f"""<tr>
    <td>CPU 使用率 (%)</td>
    <td>{cpu_mean:.1f}</td>
    <td>{cpu_max:.1f}</td>
    <td>{cpu_p95:.1f}</td>
</tr>""")

        sections.append(f"""<tr>
    <td>内存 (MB)</td>
    <td>{memory_mean:.1f}</td>
    <td>{memory_max:.1f}</td>
    <td>{memory_p95:.1f}</td>
</tr>""")

        if gpu_available and gpu_mean is not None:
            sections.append(f"""<tr>
    <td>GPU 使用率 (%)</td>
    <td>{gpu_mean:.1f}</td>
    <td>{gpu_max:.1f}</td>
    <td>{gpu_p95:.1f}</td>
</tr>""")

        sections.append("</table>")

        return ''.join(sections)

    def _get_comparison_section(self) -> str:
        """获取客户端和服务端对比部分"""
        if not self.system_monitor or not self.server_snapshots:
            return ""

        sections = ['<h2 style="color: #137333; margin-top: 40px;">⚖️ 客户端 vs 服务端对比</h2>']

        client_stats = self.system_monitor.get_statistics()

        # 计算服务端统计
        cpu_values = [s.cpu_percent for s in self.server_snapshots]
        memory_values = [s.memory_mb for s in self.server_snapshots]

        server_cpu_mean = float(np.mean(cpu_values))
        server_memory_mean = float(np.mean(memory_values))

        # 对比表格
        sections.append("""<h3 style="color: #555;">平均资源使用对比</h3>
<table>
    <tr><th>指标</th><th>客户端</th><th>服务端</th><th>差异</th></tr>""")

        cpu_diff = server_cpu_mean - client_stats.cpu_mean
        cpu_diff_sign = "↑" if cpu_diff > 0 else "↓" if cpu_diff < 0 else "="
        cpu_diff_color = "#c44e00" if cpu_diff > 0 else "#137333" if cpu_diff < 0 else "#666"

        sections.append(f"""<tr>
    <td>CPU 使用率 (%)</td>
    <td>{client_stats.cpu_mean:.1f}</td>
    <td>{server_cpu_mean:.1f}</td>
    <td style="color: {cpu_diff_color};">{cpu_diff_sign} {abs(cpu_diff):.1f}</td>
</tr>""")

        memory_diff = server_memory_mean - client_stats.memory_mean_mb
        memory_diff_sign = "↑" if memory_diff > 0 else "↓" if memory_diff < 0 else "="
        memory_diff_color = "#c44e00" if memory_diff > 0 else "#137333" if memory_diff < 0 else "#666"

        sections.append(f"""<tr>
    <td>内存使用 (MB)</td>
    <td>{client_stats.memory_mean_mb:.1f}</td>
    <td>{server_memory_mean:.1f}</td>
    <td style="color: {memory_diff_color};">{memory_diff_sign} {abs(memory_diff):.1f}</td>
</tr>""")

        sections.append("</table>")

        # 添加说明
        sections.append("""<p style="margin-top: 20px; color: #666; font-size: 14px;">
<strong>提示：</strong>↑ 表示服务端资源使用更高，↓ 表示客户端资源使用更高。
</p>""")

        return ''.join(sections)

    def _get_resource_stats_table(self, stats, prefix: str) -> str:
        """生成资源统计表格"""
        sections = [f"""<h3 style="color: #555;">{prefix}资源统计</h3>
<table>
    <tr><th>指标</th><th>平均值</th><th>最大值</th><th>P95</th></tr>"""]

        sections.append(f"""<tr>
    <td>CPU 使用率 (%)</td>
    <td>{stats.cpu_mean:.1f}</td>
    <td>{stats.cpu_max:.1f}</td>
    <td>{stats.cpu_p95:.1f}</td>
</tr>""")

        sections.append(f"""<tr>
    <td>内存 (MB)</td>
    <td>{stats.memory_mean_mb:.1f}</td>
    <td>{stats.memory_max_mb:.1f}</td>
    <td>{stats.memory_p95_mb:.1f}</td>
</tr>""")

        if stats.gpu_available and stats.gpu_mean is not None:
            sections.append(f"""<tr>
    <td>GPU 使用率 (%)</td>
    <td>{stats.gpu_mean:.1f}</td>
    <td>{stats.gpu_max:.1f}</td>
    <td>{stats.gpu_p95:.1f}</td>
</tr>""")

        sections.append("</table>")

        return ''.join(sections)

    def _get_summary_section(self) -> str:
        """重写摘要部分，添加客户端/服务端信息"""
        sections = ['<div class="summary">']
        sections.append('<h2>📊 监控摘要</h2>')

        # 客户端摘要
        if self.system_monitor:
            stats = self.system_monitor.get_statistics()
            sections.append(f"""
<div style="margin-bottom: 20px; padding: 15px; background: #e8f0fe; border-radius: 8px;">
    <h3 style="margin-top: 0; color: #2c5aa0;">客户端</h3>
    <p><strong>监控时长:</strong> {stats.duration_seconds:.1f} 秒</p>
    <p><strong>采样数:</strong> {stats.sample_count}</p>
    <p><strong>平均 CPU:</strong> {stats.cpu_mean:.1f}% | <strong>平均内存:</strong> {stats.memory_mean_mb:.1f} MB</p>
</div>
""")

        # 服务端摘要
        if self.server_snapshots:
            cpu_values = [s.cpu_percent for s in self.server_snapshots]
            memory_values = [s.memory_mb for s in self.server_snapshots]
            cpu_mean = float(np.mean(cpu_values))
            memory_mean = float(np.mean(memory_values))

            sections.append(f"""
<div style="margin-bottom: 20px; padding: 15px; background: #fef7e0; border-radius: 8px;">
    <h3 style="margin-top: 0; color: #c44e00;">服务端</h3>
    <p><strong>快照数:</strong> {len(self.server_snapshots)}</p>
    <p><strong>平均 CPU:</strong> {cpu_mean:.1f}% | <strong>平均内存:</strong> {memory_mean:.1f} MB</p>
</div>
""")

        # 性能摘要
        if self.performance_tracker:
            all_stats = self.performance_tracker.get_all_statistics()
            if all_stats:
                sections.append(f"""
<div style="padding: 15px; background: #e6f4ea; border-radius: 8px;">
    <h3 style="margin-top: 0; color: #137333;">性能追踪</h3>
    <p><strong>追踪阶段数:</strong> {len(all_stats)}</p>
</div>
""")

        sections.append('</div>')
        return ''.join(sections)
