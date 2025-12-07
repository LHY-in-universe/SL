# SplitLearnMonitor

**Performance monitoring library for Split Learning systems**

SplitLearnMonitor provides comprehensive monitoring capabilities for Split Learning frameworks, including system resource tracking (CPU, GPU, memory) and phase-based performance analysis with beautiful visualizations.

## Features

- **System Resource Monitoring**
  - CPU utilization tracking
  - Memory usage monitoring
  - GPU monitoring with graceful fallback (NVIDIA GPUs via pynvml)
  - Background thread sampling with configurable intervals

- **Performance Tracking**
  - Phase-based timing with context managers
  - Statistical analysis (mean, median, P50/P95/P99, std dev)
  - Nested phase tracking support
  - Time breakdown and percentage analysis

- **Comprehensive Visualizations**
  - Time series plots for resource usage
  - Performance comparison charts
  - Distribution plots (pie charts, histograms)
  - Statistical tables

- **Report Generation**
  - HTML reports with embedded charts
  - Markdown summaries
  - JSON/CSV data export
  - Standalone files (no external dependencies)

- **Easy Integration**
  - Simple client and server wrappers
  - Context manager support
  - Minimal code changes required
  - Compatible with existing Split Learning code

## Installation / 依赖

- 已验证环境：Python 3.11.12、torch 2.9.1、psutil 5.9.5、matplotlib 3.7.1。
- 源码 + PYTHONPATH：`export PYTHONPATH=/Users/lhy/Desktop/Git/SL/SplitLearnMonitor/src:$PYTHONPATH`
- 或开发模式安装：`pip install -e /Users/lhy/Desktop/Git/SL/SplitLearnMonitor`
- GPU 监控需 `pynvml`，可选交互可安装 `[full]`。

## Quick Start

### Simple Client Monitoring

```python
from splitlearn_monitor import ClientMonitor

# Create and start monitor
monitor = ClientMonitor(session_name="my_session", enable_gpu=True)
monitor.start()

# Track your phases
with monitor.track_phase("bottom_model"):
    hidden_states = bottom_model(input_ids)

with monitor.track_phase("trunk_remote"):
    hidden_states = trunk_client.compute(hidden_states)

with monitor.track_phase("top_model"):
    output = top_model(hidden_states)

# Stop and save report
monitor.stop()
monitor.save_report("monitoring_report.html")
```

### Quick Monitoring (One-liner)

```python
from splitlearn_monitor.integrations.client_monitor import quick_monitor

with quick_monitor("task_name") as monitor:
    # Your code here
    with monitor.track_phase("computation"):
        result = expensive_computation()
# Report automatically generated!
```

### Manual Control

```python
from splitlearn_monitor import SystemMonitor, PerformanceTracker, HTMLReporter

# Initialize
sys_monitor = SystemMonitor(sampling_interval=0.1, enable_gpu=True)
perf_tracker = PerformanceTracker()

# Start monitoring
sys_monitor.start()

# Track phases
with perf_tracker.track_phase("phase1"):
    # Do work
    pass

# Stop and generate report
sys_monitor.stop()
reporter = HTMLReporter(sys_monitor, perf_tracker)
reporter.generate_report("report.html")
```

## Usage Examples

See the `examples/` directory for complete examples:

- `basic_client_monitoring.py` - Client-side monitoring
- `basic_server_monitoring.py` - Server-side monitoring
- `quick_monitor_demo.py` - Quick monitoring demo

### Running Examples

```bash
# Client monitoring example
python examples/basic_client_monitoring.py

# Quick monitor demo
python examples/quick_monitor_demo.py

# Server monitoring example
python examples/basic_server_monitoring.py
```

## Integration with Split Learning

### Client Integration

Integrate monitoring into your existing Split Learning client:

```python
from splitlearn_comm.quickstart import Client
from splitlearn_monitor import ClientMonitor

# Your existing setup
trunk_client = Client("localhost:50052")
monitor = ClientMonitor("client_session")
monitor.start()

# Wrap your inference loop
for step in range(max_steps):
    with monitor.track_phase("bottom_model"):
        hidden_1 = bottom(input_ids)

    with monitor.track_phase("trunk_remote"):
        hidden_2 = trunk_client.compute(hidden_1)

    with monitor.track_phase("top_model"):
        output = top(hidden_2)

monitor.stop()
monitor.save_report()
```

### Server Integration

```python
from splitlearn_manager.quickstart import ManagedServer
from splitlearn_monitor import ServerMonitor

# Create server with monitoring
server = ManagedServer(port=50052, ...)
monitor = ServerMonitor("trunk_server")
monitor.start()

# Server runs and monitor collects data...
# Generate report on demand
monitor.save_report("server_report.html")
```

## API Reference

### Core Classes

#### `SystemMonitor`
Background system resource monitor with configurable sampling.

```python
monitor = SystemMonitor(
    sampling_interval=0.1,  # Sample every 100ms
    max_samples=10000,      # Keep last 10k samples
    enable_gpu=True         # Enable GPU monitoring
)
monitor.start()
# ... work ...
monitor.stop()
stats = monitor.get_statistics()
```

#### `PerformanceTracker`
Phase-based performance tracking with statistical analysis.

```python
tracker = PerformanceTracker()

with tracker.track_phase("phase_name"):
    # Your code here
    pass

stats = tracker.get_phase_statistics("phase_name")
print(f"Mean: {stats.mean_ms}ms, P95: {stats.p95_ms}ms")
```

#### `HTMLReporter`
Generate comprehensive HTML reports with embedded charts.

```python
reporter = HTMLReporter(system_monitor, performance_tracker)
reporter.generate_report("report.html", title="My Report")
```

### Integration Classes

#### `ClientMonitor`
Simplified monitoring for Split Learning clients.

#### `ServerMonitor`
Simplified monitoring for Split Learning servers.

## Output Examples

### HTML Report
The HTML reporter generates a standalone file with:
- Summary dashboard with key metrics
- Time series charts showing resource usage over time
- Performance comparison charts
- Distribution analysis (pie charts, histograms)
- Detailed statistical tables
- All data embedded (no external dependencies)

### JSON Export
Complete data export including:
- Resource snapshots with timestamps
- Phase statistics (mean, median, percentiles)
- Raw timing data
- Metadata

## Configuration

```python
from splitlearn_monitor import MonitorConfig

config = MonitorConfig(
    sampling_interval=0.1,     # Sampling rate
    max_samples=10000,         # Buffer size
    enable_gpu=True,           # GPU monitoring
    viz_backend="matplotlib",  # Visualization backend
    output_format="png",       # Image format
)
```

## Performance Considerations

- **Monitoring Overhead**: Typically <2% CPU and memory overhead
- **Sampling Rate**: Default 0.1s (100ms) is recommended for most use cases
- **Memory Usage**: Circular buffer with configurable size (default: 10,000 samples)
- **Thread Safety**: All operations are thread-safe

## Requirements

### Core Dependencies
- Python >= 3.8
- psutil >= 5.9.0 (CPU/memory monitoring)
- numpy >= 1.24.0 (statistics)
- matplotlib >= 3.5.0 (visualization)

### Optional Dependencies
- pynvml >= 11.5.0 (GPU monitoring, NVIDIA only)
- plotly >= 5.0.0 (interactive visualizations)
- pandas >= 1.5.0 (data analysis)

## Minimal smoke test (CPU only)

```bash
export PYTHONPATH=/Users/lhy/Desktop/Git/SL/SplitLearnMonitor/src:${PYTHONPATH:-}
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 - <<'PY'
from splitlearn_monitor.core.performance_tracker import PerformanceTracker
from splitlearn_monitor.core.system_monitor import SystemMonitor
import time

pt = PerformanceTracker()
pt.record_phase("demo", 12.3)
print("perf keys:", list(pt.get_all_statistics().keys()))

sm = SystemMonitor(sampling_interval=0.1, enable_gpu=False)
sm.start()
time.sleep(0.3)
sm.stop()
print("snapshot:", sm.get_current_snapshot())
PY
```

## 简易 API 概览

- `SystemMonitor(sampling_interval=0.1, enable_gpu=True, max_samples=10000)`: 后台线程采样 CPU/内存/GPU；`start()` / `stop()` / `get_current_snapshot()` / `get_statistics()`。
- `PerformanceTracker()`: 记录阶段耗时；`track_phase(name)` 上下文或 `record_phase(name, duration_ms)`；`get_all_statistics()`。
- `HTMLReporter(system_monitor, performance_tracker)`: 生成 HTML 报告 `generate_report(path)`；还有 `MarkdownReporter/DataExporter` 等。
- 集成包装：`ClientMonitor`、`ServerMonitor`、`quick_monitor()`（便捷上下文）。

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Citation

If you use SplitLearnMonitor in your research, please cite:

```bibtex
@software{splitlearn_monitor,
  title={SplitLearnMonitor: Performance Monitoring for Split Learning},
  author={Split Learning Team},
  year={2024},
  url={https://github.com/split-learning/SplitLearnMonitor}
}
```

## Acknowledgments

Built for the Split Learning framework, designed to help developers and researchers monitor and optimize distributed machine learning systems.
