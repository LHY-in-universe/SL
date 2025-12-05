"""
Distribution Visualizer - Plot resource and time distributions
"""
from typing import Dict, Optional, List
import matplotlib.pyplot as plt
import numpy as np

from ..config import PhaseStatistics


class DistributionVisualizer:
    """
    Generate distribution visualizations

    Creates pie charts, histograms, and other distribution plots to show
    how time and resources are distributed across phases.

    Example:
        >>> viz = DistributionVisualizer()
        >>> fig = viz.plot_phase_distribution_pie(phase_stats)
        >>> viz.save_figure(fig, "distribution.png")
    """

    def __init__(
        self,
        figsize: tuple = (10, 8),
        dpi: int = 100,
        style: str = "default"
    ):
        """
        Initialize distribution visualizer

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

    def plot_phase_distribution_pie(
        self,
        phase_stats: Dict[str, PhaseStatistics],
        title: str = "Time Distribution by Phase",
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot pie chart showing time distribution across phases

        Args:
            phase_stats: Dictionary mapping phase names to PhaseStatistics
            title: Plot title
            output_path: If provided, save figure to this path

        Returns:
            Matplotlib Figure object
        """
        if not phase_stats:
            raise ValueError("No phase statistics provided")

        # Extract data
        phase_names = list(phase_stats.keys())
        percentages = [stats.percentage_of_total for stats in phase_stats.values()]

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Create pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(phase_names)))
        wedges, texts, autotexts = ax.pie(
            percentages,
            labels=phase_names,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'fontsize': 11}
        )

        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        # Equal aspect ratio ensures circular pie
        ax.axis('equal')

        # Title
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        if output_path:
            self.save_figure(fig, output_path)

        return fig

    def plot_phase_distribution_horizontal_bars(
        self,
        phase_stats: Dict[str, PhaseStatistics],
        title: str = "Time Distribution by Phase",
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot horizontal bar chart showing time distribution

        Args:
            phase_stats: Dictionary mapping phase names to PhaseStatistics
            title: Plot title
            output_path: If provided, save figure to this path

        Returns:
            Matplotlib Figure object
        """
        if not phase_stats:
            raise ValueError("No phase statistics provided")

        # Sort by total time (descending)
        sorted_stats = sorted(
            phase_stats.items(),
            key=lambda x: x[1].total_time_ms,
            reverse=True
        )

        phase_names = [name for name, _ in sorted_stats]
        total_times = [stats.total_time_ms for _, stats in sorted_stats]
        percentages = [stats.percentage_of_total for _, stats in sorted_stats]

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Create horizontal bars
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(phase_names)))
        bars = ax.barh(phase_names, total_times, color=colors, edgecolor='black', linewidth=1)

        # Add value labels
        for bar, time_ms, pct in zip(bars, total_times, percentages):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{time_ms:.1f}ms ({pct:.1f}%)',
                   ha='left', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Formatting
        ax.set_xlabel('Total Time (ms)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if output_path:
            self.save_figure(fig, output_path)

        return fig

    def plot_latency_histogram(
        self,
        latencies: List[float],
        phase_name: str = "Phase",
        bins: int = 50,
        title: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot histogram of latency distribution

        Args:
            latencies: List of latency measurements in milliseconds
            phase_name: Name of the phase
            bins: Number of histogram bins
            title: Plot title (auto-generated if None)
            output_path: If provided, save figure to this path

        Returns:
            Matplotlib Figure object
        """
        if not latencies:
            raise ValueError("No latency data provided")

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Create histogram
        n, bins_edges, patches = ax.hist(latencies, bins=bins, color='#54A0FF',
                                        edgecolor='black', alpha=0.7)

        # Add statistical lines
        mean_val = np.mean(latencies)
        median_val = np.median(latencies)
        p95_val = np.percentile(latencies, 95)

        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}ms')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}ms')
        ax.axvline(p95_val, color='orange', linestyle='--', linewidth=2, label=f'P95: {p95_val:.1f}ms')

        # Formatting
        ax.set_xlabel('Latency (ms)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)

        if title is None:
            title = f"Latency Distribution - {phase_name}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if output_path:
            self.save_figure(fig, output_path)

        return fig

    def plot_multiple_histograms(
        self,
        phase_timings: Dict[str, List[float]],
        bins: int = 30,
        title: str = "Latency Distributions by Phase",
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot overlapping histograms for multiple phases

        Args:
            phase_timings: Dictionary mapping phase names to lists of timings
            bins: Number of histogram bins
            title: Plot title
            output_path: If provided, save figure to this path

        Returns:
            Matplotlib Figure object
        """
        if not phase_timings:
            raise ValueError("No phase timings provided")

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot histograms
        colors = plt.cm.Set2(np.linspace(0, 1, len(phase_timings)))

        for (phase_name, timings), color in zip(phase_timings.items(), colors):
            ax.hist(timings, bins=bins, alpha=0.5, label=phase_name,
                   color=color, edgecolor='black', linewidth=0.5)

        # Formatting
        ax.set_xlabel('Latency (ms)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

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
