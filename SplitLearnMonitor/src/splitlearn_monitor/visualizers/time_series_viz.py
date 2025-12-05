"""
Time Series Visualizer - Plot resource usage over time
"""
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from ..config import ResourceSnapshot


class TimeSeriesVisualizer:
    """
    Generate time series plots for resource monitoring data

    Creates line plots showing CPU, memory, and GPU usage over time.

    Example:
        >>> viz = TimeSeriesVisualizer()
        >>> fig = viz.plot_resource_timeline(snapshots, metrics=["cpu", "memory"])
        >>> viz.save_figure(fig, "resources.png")
    """

    def __init__(
        self,
        figsize: tuple = (12, 8),
        dpi: int = 100,
        style: str = "default"
    ):
        """
        Initialize time series visualizer

        Args:
            figsize: Figure size in inches (width, height)
            dpi: DPI for rasterized output
            style: Matplotlib style to use
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style = style

        # Set style
        if style != "default":
            plt.style.use(style)

    def plot_resource_timeline(
        self,
        snapshots: List[ResourceSnapshot],
        metrics: List[str] = None,
        title: str = "Resource Usage Over Time",
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot resource usage over time

        Args:
            snapshots: List of ResourceSnapshot objects
            metrics: List of metrics to plot ("cpu", "memory", "gpu")
                    If None, plots all available metrics
            title: Plot title
            output_path: If provided, save figure to this path

        Returns:
            Matplotlib Figure object
        """
        if not snapshots:
            raise ValueError("No snapshots provided")

        # Default to all metrics if not specified
        if metrics is None:
            metrics = ["cpu", "memory"]
            if snapshots[0].gpu_available:
                metrics.append("gpu")

        # Extract data
        timestamps = [datetime.fromtimestamp(s.timestamp) for s in snapshots]
        cpu_data = [s.cpu_percent for s in snapshots]
        memory_data = [s.memory_mb for s in snapshots]

        # GPU data (may be None)
        gpu_available = snapshots[0].gpu_available
        if gpu_available:
            gpu_util_data = [s.gpu_utilization for s in snapshots if s.gpu_utilization is not None]
            gpu_mem_data = [s.gpu_memory_used_mb for s in snapshots if s.gpu_memory_used_mb is not None]
            gpu_timestamps = [datetime.fromtimestamp(s.timestamp) for s in snapshots
                            if s.gpu_utilization is not None]

        # Determine number of subplots
        n_plots = len([m for m in metrics if m in ["cpu", "memory", "gpu"] and
                      (m != "gpu" or gpu_available)])

        # Create figure and subplots
        fig, axes = plt.subplots(n_plots, 1, figsize=self.figsize, dpi=self.dpi)
        if n_plots == 1:
            axes = [axes]

        plot_idx = 0

        # Plot CPU
        if "cpu" in metrics:
            ax = axes[plot_idx]
            ax.plot(timestamps, cpu_data, linewidth=1.5, color='#2E86DE', label='CPU')
            ax.set_ylabel('CPU Usage (%)', fontsize=10)
            ax.set_ylim(0, max(cpu_data) * 1.1 if cpu_data else 100)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            plot_idx += 1

        # Plot Memory
        if "memory" in metrics:
            ax = axes[plot_idx]
            ax.plot(timestamps, memory_data, linewidth=1.5, color='#10AC84', label='Memory')
            ax.set_ylabel('Memory Usage (MB)', fontsize=10)
            ax.set_ylim(0, max(memory_data) * 1.1 if memory_data else 1000)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            plot_idx += 1

        # Plot GPU
        if "gpu" in metrics and gpu_available:
            ax = axes[plot_idx]
            # GPU Utilization
            ax.plot(gpu_timestamps, gpu_util_data, linewidth=1.5,
                   color='#EE5A6F', label='GPU Utilization')
            ax.set_ylabel('GPU Utilization (%)', fontsize=10)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)

            # GPU Memory on secondary y-axis
            ax2 = ax.twinx()
            ax2.plot(gpu_timestamps, gpu_mem_data, linewidth=1.5,
                    color='#FD9644', linestyle='--', label='GPU Memory')
            ax2.set_ylabel('GPU Memory (MB)', fontsize=10, color='#FD9644')
            ax2.tick_params(axis='y', labelcolor='#FD9644')

            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

            plot_idx += 1

        # Format x-axis for bottom plot
        axes[-1].set_xlabel('Time', fontsize=10)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Set title
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

        # Adjust layout
        plt.tight_layout()

        # Save if requested
        if output_path:
            self.save_figure(fig, output_path)

        return fig

    def plot_single_metric(
        self,
        snapshots: List[ResourceSnapshot],
        metric: str,
        title: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot a single metric over time

        Args:
            snapshots: List of ResourceSnapshot objects
            metric: Metric to plot ("cpu_percent", "memory_mb", "gpu_utilization", etc.)
            title: Plot title (auto-generated if None)
            output_path: If provided, save figure to this path

        Returns:
            Matplotlib Figure object
        """
        if not snapshots:
            raise ValueError("No snapshots provided")

        timestamps = [datetime.fromtimestamp(s.timestamp) for s in snapshots]
        values = [getattr(s, metric, None) for s in snapshots]

        # Filter out None values
        valid_data = [(t, v) for t, v in zip(timestamps, values) if v is not None]
        if not valid_data:
            raise ValueError(f"No valid data for metric: {metric}")

        timestamps, values = zip(*valid_data)

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot
        ax.plot(timestamps, values, linewidth=2)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Set title
        if title is None:
            title = f"{metric.replace('_', ' ').title()} Over Time"
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Save if requested
        if output_path:
            self.save_figure(fig, output_path)

        return fig

    @staticmethod
    def save_figure(fig: plt.Figure, output_path: str, dpi: Optional[int] = None):
        """
        Save figure to file

        Args:
            fig: Matplotlib Figure object
            output_path: Path to save figure
            dpi: DPI for output (uses figure's dpi if not specified)
        """
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')

    @staticmethod
    def close_figure(fig: plt.Figure):
        """
        Close figure to free memory

        Args:
            fig: Matplotlib Figure object
        """
        plt.close(fig)
