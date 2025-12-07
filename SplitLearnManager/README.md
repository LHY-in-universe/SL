# splitlearn-manager

A comprehensive model deployment and lifecycle management library for distributed deep learning.

## Features

### 🚀 **Model Lifecycle Management**
- Dynamic model loading/unloading
- Support for PyTorch, Hugging Face, and custom models
- Model warmup and optimization
- Least-recently-used (LRU) eviction policy

### 💾 **Resource Management**
- CPU, memory, and GPU monitoring
- Automatic resource allocation
- Memory limits and quotas
- Smart device selection

### 🔀 **Request Routing**
- Round-robin routing
- Least-loaded routing
- Custom routing strategies
- Load balancing across models

### 📊 **Monitoring & Metrics**
- Prometheus metrics integration
- Real-time resource usage tracking
- Model performance metrics
- Health checks

### ⚙️ **Configuration Management**
- YAML-based configuration
- Model-specific settings
- Server configuration
- Validation and error checking

## Installation / 依赖

- 已验证环境：Python 3.11.12、torch 2.9.1、splitlearn-comm 1.0.0（grpcio 1.69.0）。
- 源码 + PYTHONPATH：`export PYTHONPATH=/Users/lhy/Desktop/Git/SL/SplitLearnManager/src:$PYTHONPATH`
- 或开发模式安装：`pip install -e /Users/lhy/Desktop/Git/SL/SplitLearnManager`

## Quick Start

### Basic Server

```python
import torch.nn as nn
from splitlearn_manager import ManagedServer, ModelConfig

# Create a simple model
model = nn.Sequential(
    nn.Linear(768, 1024),
    nn.ReLU(),
    nn.Linear(1024, 768)
)

# Save model
torch.save(model, "model.pt")

# Create server
server = ManagedServer()

# Configure and load model
config = ModelConfig(
    model_id="my_model",
    model_path="model.pt",
    model_type="pytorch",
    device="cuda",
    batch_size=32
)

server.load_model(config)

# Start serving
server.start()
server.wait_for_termination()
```

### Client Connection

```python
from splitlearn_comm import GRPCComputeClient
import torch

# Connect to managed server
client = GRPCComputeClient("localhost:50051")
client.connect()

# Perform inference
input_tensor = torch.randn(1, 10, 768)
output_tensor = client.compute(input_tensor)

# Get statistics
stats = client.get_statistics()
print(f"Avg latency: {stats['avg_total_time_ms']:.2f}ms")

client.close()
```

### Configuration from YAML

```python
from splitlearn_manager import ModelConfig, ServerConfig

# Load model configuration
model_config = ModelConfig.from_yaml("model_config.yaml")

# Load server configuration
server_config = ServerConfig.from_yaml("server_config.yaml")

# Create server with configuration
server = ManagedServer(config=server_config)
server.load_model(model_config)
```

## Architecture

### Components

```
splitlearn-manager/
├── core/
│   ├── model_manager.py      # Model lifecycle management
│   ├── model_loader.py       # Model loading from various sources
│   └── resource_manager.py   # System resource management
├── config/
│   ├── model_config.py       # Model configuration
│   └── server_config.py      # Server configuration
├── monitoring/
│   ├── metrics.py            # Prometheus metrics
│   └── health.py             # Health checks
├── routing/
│   └── router.py             # Request routing
└── server/
    └── managed_server.py     # Main server implementation
```

### Model Lifecycle

```
┌─────────────┐
│   LOADING   │ ← load_model()
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    READY    │ ← Serving requests
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  UNLOADING  │ ← unload_model()
└─────────────┘
```

## Examples

### Multi-Model Server

```python
from splitlearn_manager import ManagedServer, ModelConfig

server = ManagedServer()

# Load multiple models
for model_id in ["model_a", "model_b", "model_c"]:
    config = ModelConfig(
        model_id=model_id,
        model_path=f"{model_id}.pt",
        device="cuda"
    )
    server.load_model(config)

server.start()
```

### Dynamic Model Management

```python
# Load a new model
server.load_model(new_config)

# Unload an old model
server.unload_model("old_model_id")

# Reload a model (e.g., after updating weights)
server.reload_model("model_id")

# Get model information
info = server.model_manager.get_model_info("model_id")
```

### Resource Management

```python
from splitlearn_manager import ResourceManager

rm = ResourceManager(max_memory_percent=80)

# Get current usage
usage = rm.get_current_usage()
print(f"CPU: {usage.cpu_percent}%")
print(f"Memory: {usage.memory_mb}MB")

# Find best device
device = rm.find_best_device(prefer_gpu=True)

# Check if resources available
if rm.check_available_resources(required_memory_mb=2048, required_gpu=True):
    # Load model
    pass
```

### Monitoring

```python
from splitlearn_manager import ManagedServer

server = ManagedServer()
# Metrics automatically exposed on http://localhost:8000

# Manual health check
health = server.health_checker.check_health()
print(f"Status: {health['status']}")

# Get server status
status = server.get_status()
print(f"Models loaded: {len(status['models'])}")
```

## Configuration

### Model Configuration

```yaml
# model_config.yaml
model_id: "my_model"
model_path: "/path/to/model.pt"
model_type: "pytorch"  # pytorch, huggingface, custom
device: "cuda:0"
batch_size: 32
max_memory_mb: 4096
warmup: true

config:
  input_shape: [1, 10, 768]
  max_sequence_length: 512
```

### Server Configuration

```yaml
# server_config.yaml
host: "0.0.0.0"
port: 50051
max_workers: 10
max_models: 5
metrics_port: 8000
health_check_interval: 30.0
enable_monitoring: true
log_level: "INFO"
```

## Prometheus Metrics

When monitoring is enabled, the following metrics are exposed:

- `model_load_total`: Total model loads
- `model_unload_total`: Total model unloads
- `models_loaded`: Currently loaded models
- `inference_requests_total`: Total inference requests
- `inference_duration_seconds`: Inference latency histogram
- `cpu_usage_percent`: CPU usage
- `memory_usage_mb`: Memory usage
- `gpu_memory_mb`: GPU memory usage

Access metrics at `http://localhost:8000/metrics`

## API Documentation

### ManagedServer

```python
class ManagedServer:
    def __init__(self, config: Optional[ServerConfig] = None)
    def load_model(self, config: ModelConfig) -> bool
    def unload_model(self, model_id: str) -> bool
    def start(self)
    def stop(self, grace: float = 5.0)
    def wait_for_termination(self)
    def get_status(self) -> Dict
```

### ModelManager

```python
class ModelManager:
    def load_model(self, config: ModelConfig) -> bool
    def unload_model(self, model_id: str) -> bool
    def get_model(self, model_id: str) -> Optional[ManagedModel]
    def list_models(self) -> List[Dict]
    def get_model_info(self, model_id: str) -> Dict
    def reload_model(self, model_id: str) -> bool
    def get_statistics(self) -> Dict
    def shutdown(self)
```

### ResourceManager

```python
class ResourceManager:
    def get_current_usage(self) -> ResourceUsage
    def check_available_resources(self, required_memory_mb, required_gpu) -> bool
    def find_best_device(self, prefer_gpu, min_free_memory_mb) -> str
    def get_gpu_info(self) -> List[Dict]
    def log_resource_usage(self)
```

## Best Practices

1. **Resource Limits**: Always set `max_memory_mb` to prevent OOM errors
2. **Model Warmup**: Enable warmup for production deployments
3. **Health Checks**: Monitor health endpoint regularly
4. **Graceful Shutdown**: Use `stop()` with appropriate grace period
5. **Metrics**: Monitor Prometheus metrics for performance insights
6. **Device Selection**: Let ResourceManager choose best device automatically

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- splitlearn-comm >= 1.0.0
- pyyaml >= 6.0
- psutil >= 5.9.0
- prometheus-client >= 0.16.0

## Development

```bash
# Clone repository
git clone https://github.com/yourusername/splitlearn-manager.git
cd splitlearn-manager

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run examples
python examples/basic_server.py
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Minimal smoke test (imports)

```bash
export PYTHONPATH=/Users/lhy/Desktop/Git/SL/SplitLearnManager/src:/Users/lhy/Desktop/Git/SL/SplitLearnComm/src:${PYTHONPATH:-}
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 - <<'PY'
from splitlearn_manager import ManagedServer, ModelConfig
print("imports ok")
PY
```

## 简易 API 概览

- `ManagedServer(config: Optional[ServerConfig]=None)`: 管理模型的 gRPC 服务端；`load_model(ModelConfig)` / `unload_model(model_id)` / `get_status()` / `start()` / `stop()`。
- `ModelConfig`: 描述模型元数据（路径、类型、设备、batch 等），支持 `from_yaml()`。
- `ServerConfig`: 服务器配置（端口、并发、监控开关等），支持 `from_yaml()`。
- `ResourceManager`: 资源查询/选择，`get_current_usage()`、`find_best_device()`、`check_available_resources()`。

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{splitlearn_manager2024,
  title = {splitlearn-manager: Model Management for Distributed Deep Learning},
  author = {SplitLearn Contributors},
  year = {2024},
  url = {https://github.com/yourusername/splitlearn-manager}
}
```

## See Also

- [splitlearn-comm](https://github.com/yourusername/splitlearn-comm) - Communication layer
- [Examples](examples/) - Usage examples
- [Documentation](docs/) - Detailed documentation
