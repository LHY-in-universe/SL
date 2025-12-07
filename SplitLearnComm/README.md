# splitlearn-comm

A high-performance gRPC-based communication library for distributed deep learning.

## Features

- 🚀 **High Performance**: Optimized tensor serialization using binary format
- 🔌 **Model Agnostic**: Completely decoupled from specific models via `ComputeFunction` abstraction
- 🔄 **Reliable**: Built-in retry mechanisms with exponential backoff
- 🛡️ **Robust**: Comprehensive error handling and health checks
- 📊 **Observable**: Performance metrics and statistics tracking
- 🌐 **Flexible**: Supports single-machine, LAN, and WAN deployments

## Installation / 依赖

- 已验证环境：Python 3.11.12、torch 2.9.1、grpcio 1.69.0（注意上限 `<1.70.0`）、protobuf 4.25.x。
- 源码 + PYTHONPATH（当前目录结构）：`export PYTHONPATH=/Users/lhy/Desktop/Git/SL/SplitLearnComm/src:$PYTHONPATH`
- 或开发模式安装：`pip install -e /Users/lhy/Desktop/Git/SL/SplitLearnComm`

## Quick Start

### Server Example

```python
import torch
from splitlearn_comm import GRPCComputeServer
from splitlearn_comm.core import ModelComputeFunction

# Create a simple model
model = torch.nn.Sequential(
    torch.nn.Linear(768, 768),
    torch.nn.ReLU(),
    torch.nn.Linear(768, 768)
)

# Wrap in ComputeFunction
compute_fn = ModelComputeFunction(model, device="cuda")

# Start server
server = GRPCComputeServer(
    compute_fn=compute_fn,
    host="0.0.0.0",
    port=50051
)

server.start()
server.wait_for_termination()
```

### Client Example

```python
import torch
from splitlearn_comm import GRPCComputeClient

# Connect to server
client = GRPCComputeClient("localhost:50051")
client.connect()

# Perform computation
input_tensor = torch.randn(1, 10, 768)
output_tensor = client.compute(input_tensor)

print(f"Output shape: {output_tensor.shape}")

# Get statistics
stats = client.get_statistics()
print(f"Total requests: {stats['total_requests']}")
print(f"Avg network time: {stats['avg_network_time_ms']:.2f}ms")

client.close()
```

## Core Concepts

### ComputeFunction

The `ComputeFunction` abstraction allows you to use any computation logic:

```python
from splitlearn_comm.core import ComputeFunction

class MyComputeFunction(ComputeFunction):
    def __init__(self, model):
        self.model = model

    def compute(self, input_tensor):
        # Your custom logic here
        with torch.no_grad():
            return self.model(input_tensor)

    def get_info(self):
        return {"model_name": "MyModel", "device": "cuda"}
```

### Retry Strategy

Built-in retry mechanisms for reliability:

```python
from splitlearn_comm import GRPCComputeClient, ExponentialBackoff

# Custom retry strategy
retry = ExponentialBackoff(
    max_retries=5,
    initial_delay=1.0,
    max_delay=30.0
)

client = GRPCComputeClient(
    "localhost:50051",
    retry_strategy=retry
)
```

## Advanced Usage

### Compressed Communication

For bandwidth-constrained scenarios:

```python
from splitlearn_comm.core import CompressedTensorCodec

codec = CompressedTensorCodec(compression_level=6)

server = GRPCComputeServer(
    compute_fn=compute_fn,
    codec=codec,
    port=50051
)
```

### Context Manager Support

```python
# Server
with GRPCComputeServer(compute_fn, port=50051) as server:
    # Server automatically starts and stops
    pass

# Client
with GRPCComputeClient("localhost:50051") as client:
    output = client.compute(input_tensor)
```

### Health Checks and Service Info

```python
# Health check
is_healthy = client.health_check()

# Get service information
info = client.get_service_info()
print(f"Service: {info['service_name']}")
print(f"Version: {info['version']}")
print(f"Device: {info['device']}")
print(f"Total requests: {info['total_requests']}")
```

## API Documentation

See [docs/api.md](docs/api.md) for complete API reference.

## Examples

- [examples/simple_server.py](examples/simple_server.py) - Basic server
- [examples/simple_client.py](examples/simple_client.py) - Basic client
- [examples/custom_service.py](examples/custom_service.py) - Custom compute function

## Performance

splitlearn-comm uses optimized binary serialization for tensors:

- **Bytes format**: 4x faster than protobuf's `repeated float`
- **Zero-copy**: Minimal overhead for numpy/torch conversion
- **Compression**: Optional zlib compression for WAN scenarios

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- gRPC >= 1.50.0
- NumPy >= 1.24.0

## Development

```bash
# Clone repository
git clone https://github.com/yourusername/splitlearn-comm.git
cd splitlearn-comm

# Install in development mode
pip install -e ".[dev]"

# Compile protobuf
bash scripts/compile_proto.sh

# Run tests
pytest tests/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Minimal smoke test (imports only)

```bash
export PYTHONPATH=/Users/lhy/Desktop/Git/SL/SplitLearnComm/src:${PYTHONPATH:-}
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 - <<'PY'
from splitlearn_comm import GRPCComputeClient, GRPCComputeServer
from splitlearn_comm.core import ComputeFunction
print("imports ok")
PY
```

## 简易 API 概览

- `GRPCComputeServer(compute_fn, host="0.0.0.0", port=50051, codec=None, retry_strategy=None)`: 启动 gRPC 计算服务，`compute_fn` 实现推理逻辑。
- `GRPCComputeClient(target, retry_strategy=None, codec=None)`: 连接远端服务，`client.compute(tensor)` 进行远程推理。
- `ComputeFunction`: 抽象基类，需实现 `compute(input_tensor)` 和 `get_info()`；可用 `ModelComputeFunction(model, device)` 封装 torch 模型。
- 其他常用工具：`ExponentialBackoff`（重试策略）、`CompressedTensorCodec`（压缩传输）。

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use splitlearn-comm in your research, please cite:

```bibtex
@software{splitlearn_comm2024,
  title = {splitlearn-comm: A gRPC Communication Library for Distributed Deep Learning},
  author = {SplitLearn Contributors},
  year = {2024},
  url = {https://github.com/yourusername/splitlearn-comm}
}
```
