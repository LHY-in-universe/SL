"""
HTML Reporter - Generate comprehensive HTML reports with embedded charts
"""
import base64
from io import BytesIO
from typing import Optional
from datetime import datetime
from pathlib import Path

from ..core import SystemMonitor, PerformanceTracker
from ..visualizers import TimeSeriesVisualizer, ComparisonVisualizer, DistributionVisualizer
from ..utils import format_duration


class HTMLReporter:
    """
    Generate comprehensive HTML reports

    Creates standalone HTML files with embedded charts and statistics.

    Example:
        >>> reporter = HTMLReporter(system_monitor, perf_tracker)
        >>> reporter.generate_report("report.html")
    """

    def __init__(
        self,
        system_monitor: Optional[SystemMonitor] = None,
        performance_tracker: Optional[PerformanceTracker] = None
    ):
        """
        Initialize HTML reporter

        Args:
            system_monitor: SystemMonitor instance (optional)
            performance_tracker: PerformanceTracker instance (optional)
        """
        self.system_monitor = system_monitor
        self.performance_tracker = performance_tracker

        # Initialize visualizers
        self.time_series_viz = TimeSeriesVisualizer()
        self.comparison_viz = ComparisonVisualizer()
        self.distribution_viz = DistributionVisualizer()

    def generate_report(
        self,
        output_path: str,
        title: str = "Monitoring Report",
        include_raw_data: bool = False
    ):
        """
        Generate comprehensive HTML report

        Args:
            output_path: Path to output HTML file
            title: Report title
            include_raw_data: Include raw data tables
        """
        html_content = self._build_html(title, include_raw_data)

        with open(output_path, 'w') as f:
            f.write(html_content)

    def _build_html(self, title: str, include_raw_data: bool) -> str:
        """Build complete HTML document"""
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
        {self._get_resource_section()}
        {self._get_performance_section()}
        {self._get_footer()}
    </div>
</body>
</html>"""

    def _get_css(self) -> str:
        """Get embedded CSS styles"""
        return """<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f5f7fa;
}
.container {
    max-width: 1200px;
    margin: 0 auto;
    background: white;
    padding: 30px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
h1 {
    color: #2c3e50;
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
}
h2 {
    color: #34495e;
    margin-top: 30px;
    border-left: 4px solid #3498db;
    padding-left: 15px;
}
.header {
    margin-bottom: 30px;
}
.timestamp {
    color: #7f8c8d;
    font-size: 14px;
}
.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin: 20px 0;
}
.summary-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.summary-card h3 {
    margin: 0 0 10px 0;
    font-size: 14px;
    opacity: 0.9;
}
.summary-card .value {
    font-size: 28px;
    font-weight: bold;
}
.chart-container {
    margin: 30px 0;
    text-align: center;
}
.chart-container img {
    max-width: 100%;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
}
th, td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid #ecf0f1;
}
th {
    background-color: #3498db;
    color: white;
    font-weight: 600;
}
tr:hover {
    background-color: #f8f9fa;
}
.footer {
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #ecf0f1;
    text-align: center;
    color: #95a5a6;
    font-size: 12px;
}
</style>"""

    def _get_header(self, title: str) -> str:
        """Get header section"""
        return f"""<div class="header">
    <h1>{title}</h1>
    <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</div>"""

    def _get_summary_section(self) -> str:
        """Get summary cards section"""
        cards = []

        if self.system_monitor:
            stats = self.system_monitor.get_statistics()
            cards.append(f"""<div class="summary-card">
    <h3>Duration</h3>
    <div class="value">{format_duration(stats.duration_seconds)}</div>
</div>""")

            cards.append(f"""<div class="summary-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
    <h3>CPU Usage (Mean)</h3>
    <div class="value">{stats.cpu_mean:.1f}%</div>
</div>""")

            cards.append(f"""<div class="summary-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
    <h3>Memory (Mean)</h3>
    <div class="value">{stats.memory_mean_mb:.0f} MB</div>
</div>""")

        if self.performance_tracker:
            total_time = self.performance_tracker.get_total_time()
            cards.append(f"""<div class="summary-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
    <h3>Total Time</h3>
    <div class="value">{total_time:.1f} ms</div>
</div>""")

        if cards:
            return f"""<h2>Summary</h2>
<div class="summary-grid">
    {''.join(cards)}
</div>"""
        return ""

    def _get_resource_section(self) -> str:
        """Get resource monitoring section with charts"""
        if not self.system_monitor:
            return ""

        sections = ["""<h2>Resource Monitoring</h2>"""]

        # Get resource history
        snapshots = self.system_monitor.get_history()
        if not snapshots:
            sections.append("<p>No resource data collected.</p>")
            return ''.join(sections)

        # Generate time series chart
        try:
            fig = self.time_series_viz.plot_resource_timeline(
                snapshots,
                title="Resource Usage Timeline"
            )
            img_data = self._fig_to_base64(fig)
            sections.append(f"""<div class="chart-container">
    <img src="data:image/png;base64,{img_data}" alt="Resource Timeline">
</div>""")
            self.time_series_viz.close_figure(fig)
        except Exception:
            pass

        # Add statistics table
        stats = self.system_monitor.get_statistics()
        sections.append("""<h3>Resource Statistics</h3>
<table>
    <tr><th>Metric</th><th>Mean</th><th>Max</th><th>P95</th></tr>""")

        sections.append(f"""<tr>
    <td>CPU Usage (%)</td>
    <td>{stats.cpu_mean:.1f}</td>
    <td>{stats.cpu_max:.1f}</td>
    <td>{stats.cpu_p95:.1f}</td>
</tr>""")

        sections.append(f"""<tr>
    <td>Memory (MB)</td>
    <td>{stats.memory_mean_mb:.1f}</td>
    <td>{stats.memory_max_mb:.1f}</td>
    <td>{stats.memory_p95_mb:.1f}</td>
</tr>""")

        if stats.gpu_available and stats.gpu_mean is not None:
            sections.append(f"""<tr>
    <td>GPU Utilization (%)</td>
    <td>{stats.gpu_mean:.1f}</td>
    <td>{stats.gpu_max:.1f}</td>
    <td>{stats.gpu_p95:.1f}</td>
</tr>""")

        sections.append("</table>")

        return ''.join(sections)

    def _get_performance_section(self) -> str:
        """Get performance statistics section with charts"""
        if not self.performance_tracker:
            return ""

        sections = ["""<h2>Performance Statistics</h2>"""]

        all_stats = self.performance_tracker.get_all_statistics()
        if not all_stats:
            sections.append("<p>No performance data collected.</p>")
            return ''.join(sections)

        # Generate comparison chart
        try:
            fig = self.comparison_viz.plot_phase_comparison(
                all_stats,
                title="Phase Performance Comparison"
            )
            img_data = self._fig_to_base64(fig)
            sections.append(f"""<div class="chart-container">
    <img src="data:image/png;base64,{img_data}" alt="Performance Comparison">
</div>""")
            self.comparison_viz.close_figure(fig)
        except Exception:
            pass

        # Generate distribution chart
        try:
            fig = self.distribution_viz.plot_phase_distribution_pie(
                all_stats,
                title="Time Distribution by Phase"
            )
            img_data = self._fig_to_base64(fig)
            sections.append(f"""<div class="chart-container">
    <img src="data:image/png;base64,{img_data}" alt="Time Distribution">
</div>""")
            self.distribution_viz.close_figure(fig)
        except Exception:
            pass

        # Add statistics table
        sections.append("""<h3>Phase Statistics</h3>
<table>
    <tr>
        <th>Phase</th><th>Count</th><th>Total (ms)</th>
        <th>Mean (ms)</th><th>P95 (ms)</th><th>% of Total</th>
    </tr>""")

        sorted_stats = sorted(
            all_stats.items(),
            key=lambda x: x[1].total_time_ms,
            reverse=True
        )

        for phase_name, stats in sorted_stats:
            sections.append(f"""<tr>
    <td><strong>{phase_name}</strong></td>
    <td>{stats.count}</td>
    <td>{stats.total_time_ms:.1f}</td>
    <td>{stats.mean_ms:.1f}</td>
    <td>{stats.p95_ms:.1f}</td>
    <td>{stats.percentage_of_total:.1f}%</td>
</tr>""")

        sections.append("</table>")

        return ''.join(sections)

    def _get_footer(self) -> str:
        """Get footer section"""
        return """<div class="footer">
    <p>Generated by SplitLearnMonitor</p>
</div>"""

    @staticmethod
    def _fig_to_base64(fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_base64
