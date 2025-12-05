"""
Full Model Monitor - 完整模型推理监控
用于单服务器运行完整模型的场景
"""
from typing import Optional
from contextlib import contextmanager

from ..core import SystemMonitor, PerformanceTracker
from ..reporters import HTMLReporter, DataExporter


class FullModelMonitor:
    """
    完整模型推理监控器

    用于监控单服务器运行完整模型的场景，不区分 bottom/trunk/top。

    Example:
        >>> from splitlearn_monitor import FullModelMonitor
        >>>
        >>> # 初始化监控器
        >>> monitor = FullModelMonitor(
        ...     model_name="gpt2_full",
        ...     sampling_interval=0.1,
        ...     enable_gpu=True
        ... )
        >>>
        >>> # 启动监控
        >>> monitor.start()
        >>>
        >>> # 跟踪每次推理
        >>> for input_text in inputs:
        ...     with monitor.track_inference():
        ...         output = model(input_text)
        >>>
        >>> # 停止监控并生成报告
        >>> monitor.stop()
        >>> monitor.print_summary()
        >>> monitor.save_report("full_model_report.html")
    """

    def __init__(
        self,
        model_name: str = "full_model",
        sampling_interval: float = 0.1,
        enable_gpu: bool = True,
        auto_start: bool = False
    ):
        """
        初始化完整模型监控器

        Args:
            model_name: 模型名称（用于报告标题）
            sampling_interval: 资源采样间隔（秒）
            enable_gpu: 是否启用 GPU 监控
            auto_start: 是否自动启动监控
        """
        self.model_name = model_name

        # 初始化系统监控器（后台线程采样 CPU/GPU/内存）
        self.system_monitor = SystemMonitor(
            sampling_interval=sampling_interval,
            enable_gpu=enable_gpu
        )

        # 初始化性能追踪器（记录每次推理耗时）
        self.performance_tracker = PerformanceTracker()

        # 监控状态
        self._is_active = False

        if auto_start:
            self.start()

    def start(self):
        """启动监控"""
        if not self._is_active:
            self.system_monitor.start()
            self._is_active = True

    def stop(self):
        """停止监控"""
        if self._is_active:
            self.system_monitor.stop()
            self._is_active = False

    @contextmanager
    def track_inference(self, metadata: Optional[dict] = None):
        """
        跟踪一次推理

        Args:
            metadata: 可选的元数据（如 batch_size, sequence_length）

        Example:
            >>> with monitor.track_inference(metadata={"batch_size": 1}):
            ...     output = model(input_ids)
        """
        with self.performance_tracker.track_phase("inference", metadata):
            yield

    def record_inference(self, duration_ms: float, metadata: Optional[dict] = None):
        """
        手动记录一次推理

        Args:
            duration_ms: 推理耗时（毫秒）
            metadata: 可选的元数据
        """
        self.performance_tracker.record_phase("inference", duration_ms, metadata)

    def get_statistics(self) -> dict:
        """
        获取监控统计信息

        Returns:
            包含资源使用和性能统计的字典
        """
        stats = {
            "model_name": self.model_name,
            "is_active": self._is_active,
        }

        # 资源统计
        if self.system_monitor:
            resource_stats = self.system_monitor.get_statistics()
            stats["resource_stats"] = resource_stats.to_dict()

        # 性能统计
        if self.performance_tracker:
            phase_stats = self.performance_tracker.get_phase_statistics("inference")
            if phase_stats:
                stats["inference_stats"] = phase_stats.to_dict()
            stats["total_inferences"] = self.performance_tracker.get_phase_count("inference")
            stats["total_time_ms"] = self.performance_tracker.get_total_time()

        return stats

    def print_summary(self):
        """打印监控摘要到控制台"""
        print(f"\n{'='*80}")
        print(f"完整模型监控摘要 - {self.model_name}")
        print(f"{'='*80}")

        # 资源使用摘要
        if self.system_monitor:
            resource_stats = self.system_monitor.get_statistics()
            print(f"\n资源使用:")
            print(f"  监控时长: {resource_stats.duration_seconds:.1f} 秒")
            print(f"  采样数: {resource_stats.sample_count}")
            print(f"  CPU:    平均={resource_stats.cpu_mean:.1f}%, 最大={resource_stats.cpu_max:.1f}%, P95={resource_stats.cpu_p95:.1f}%")
            print(f"  内存:   平均={resource_stats.memory_mean_mb:.1f}MB, 最大={resource_stats.memory_max_mb:.1f}MB, P95={resource_stats.memory_p95_mb:.1f}MB")

            if resource_stats.gpu_available and resource_stats.gpu_mean is not None:
                print(f"  GPU:    平均={resource_stats.gpu_mean:.1f}%, 最大={resource_stats.gpu_max:.1f}%, P95={resource_stats.gpu_p95:.1f}%")
                if resource_stats.gpu_memory_mean_mb is not None:
                    print(f"  GPU内存: 平均={resource_stats.gpu_memory_mean_mb:.1f}MB, 最大={resource_stats.gpu_memory_max_mb:.1f}MB")

        # 推理性能摘要
        if self.performance_tracker:
            inference_stats = self.performance_tracker.get_phase_statistics("inference")
            if inference_stats:
                print(f"\n推理性能:")
                print(f"  总推理次数: {inference_stats.count}")
                print(f"  总耗时: {inference_stats.total_time_ms:.1f} ms")
                print(f"  平均耗时: {inference_stats.mean_ms:.1f} ms")
                print(f"  中位数: {inference_stats.median_ms:.1f} ms")
                print(f"  最小值: {inference_stats.min_ms:.1f} ms")
                print(f"  最大值: {inference_stats.max_ms:.1f} ms")
                print(f"  P50: {inference_stats.p50_ms:.1f} ms")
                print(f"  P95: {inference_stats.p95_ms:.1f} ms")
                print(f"  P99: {inference_stats.p99_ms:.1f} ms")
                print(f"  标准差: {inference_stats.std_ms:.1f} ms")

        print(f"{'='*80}\n")

    def save_report(
        self,
        output_path: Optional[str] = None,
        format: str = "html",
        include_raw_data: bool = False
    ) -> str:
        """
        保存监控报告

        Args:
            output_path: 输出文件路径（自动生成如果为 None）
            format: 报告格式（"html" 或 "json"）
            include_raw_data: 是否包含原始数据

        Returns:
            报告文件路径
        """
        # 自动生成输出路径
        if output_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.model_name}_{timestamp}_report.{format}"

        if format == "html":
            reporter = HTMLReporter(self.system_monitor, self.performance_tracker)
            reporter.generate_report(
                output_path,
                title=f"完整模型监控报告 - {self.model_name}",
                include_raw_data=include_raw_data
            )
        elif format == "json":
            exporter = DataExporter(self.system_monitor, self.performance_tracker)
            exporter.export_to_json(
                output_path,
                include_raw_snapshots=include_raw_data,
                include_raw_timings=include_raw_data
            )
        else:
            raise ValueError(f"不支持的格式: {format}")

        return output_path

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.stop()
        return False
