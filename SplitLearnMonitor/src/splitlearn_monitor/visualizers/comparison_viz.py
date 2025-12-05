"""
Comparison Visualizer - Compare performance across phases
"""
from typing import Dict, Optional, List
import matplotlib.pyplot as plt
import numpy as np

from ..config import PhaseStatistics


class ComparisonVisualizer:
    """
    Generate comparison charts for performance data

    Creates bar charts and other comparison visualizations to compare
    processing times across different phases (Bottom, Trunk, Top).

    Example:
        >>> viz = ComparisonVisualizer()
        >>> fig = viz.plot_phase_comparison(phase_stats)
        >>> viz.save_figure(fig, "comparison.png")
    """

    def __init__(
        self,
        figsize: tuple = (12, 6),
        dpi: int = 100,
        style: str = "default"
    ):
        """
        Initialize comparison visualizer

        Args:
            figsize: Figure size in inches (width, height)
            dpi: DPI for rasterized output
            style: Matplotlib style to use
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style = style

        if style != "default":
            plt.style.use(style)

    def plot_phase_comparison(
        self,
        phase_stats: Dict[str, PhaseStatistics],
        metric: str = "mean_ms",
        title: str = "Phase Performance Comparison",
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comparison of processing times across phases

        Args:
            phase_stats: Dictionary mapping phase names to PhaseStatistics
            metric: Metric to compare ("mean_ms", "median_ms", "p95_ms", etc.)
            title: Plot title
            output_path: If provided, save figure to this path

        Returns:
            Matplotlib Figure object
        """
        if not phase_stats:
            raise ValueError("No phase statistics provided")

        # Extract data
        phase_names = list(phase_stats.keys())
        values = [getattr(stats, metric) for stats in phase_stats.values()]

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Create bar chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(phase_names)))
        bars = ax.bar(phase_names, values, color=colors, edgecolor='black', linewidth=1.2)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}ms',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Formatting
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        if output_path:
            self.save_figure(fig, output_path)

        return fig

    def plot_phase_breakdown_bars(
        self,
        phase_stats: Dict[str, PhaseStatistics],
        title: str = "Time Breakdown by Phase",
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot stacked bar showing mean, min, max for each phase

        Args:
            phase_stats: Dictionary mapping phase names to PhaseStatistics
            title: Plot title
            output_path: If provided, save figure to this path

        Returns:
            Matplotlib Figure object
        """
        if not phase_stats:
            raise ValueError("No phase statistics provided")

        phase_names = list(phase_stats.keys())
        mean_values = [stats.mean_ms for stats in phase_stats.values()]
        min_values = [stats.min_ms for stats in phase_stats.values()]
        max_values = [stats.max_ms for stats in phase_stats.values()]
        p95_values = [stats.p95_ms for stats in phase_stats.values()]

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        x = np.arange(len(phase_names))
        width = 0.2

        # Create grouped bars
        ax.bar(x - 1.5*width, min_values, width, label='Min', color='#54A0FF')
        ax.bar(x - 0.5*width, mean_values, width, label='Mean', color='#48DBFB')
        ax.bar(x + 0.5*width, p95_values, width, label='P95', color='#FF9FF3')
        ax.bar(x + 1.5*width, max_values, width, label='Max', color='#FF6B6B')

        # Formatting
        ax.set_ylabel('Time (ms)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(phase_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if output_path:
            self.save_figure(fig, output_path)

        return fig

    def plot_latency_comparison(
        self,
        phase_timings: Dict[str, List[float]],
        title: str = "Latency Distribution Comparison",
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot box plots comparing latency distributions across phases

        Args:
            phase_timings: Dictionary mapping phase names to lists of timings
            title: Plot title
            output_path: If provided, save figure to this path

        Returns:
            Matplotlib Figure object
        """
        if not phase_timings:
            raise ValueError("No phase timings provided")

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Create box plot
        phase_names = list(phase_timings.keys())
        data = [phase_timings[name] for name in phase_names]

        bp = ax.boxplot(data, labels=phase_names, patch_artist=True,
                       showmeans=True, meanline=True)

        # Color the boxes
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(phase_names)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # Formatting
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        if output_path:
            self.save_figure(fig, output_path)

        return fig

    @staticmethod
    def save_figure(fig: plt.Figure, output_path: str, dpi: Optional[int] = None):
        """Save figure to file"""
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')

    @staticmethod
    def close_figure(fig: plt.Figure):
        """Close figure to free memory"""
        plt.close(fig)
