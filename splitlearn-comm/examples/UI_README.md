# Gradio UI Examples

This directory contains examples for using the built-in Gradio UI features in splitlearn-comm.

## Installation

To use the UI features, install splitlearn-comm with UI dependencies:

```bash
pip install splitlearn-comm[ui]
```

Or install all optional dependencies:

```bash
pip install splitlearn-comm[all]
```

## Examples

### 1. Client UI (client_ui_example.py)

Interactive web interface for text generation using Split Learning.

**Features:**
- Real-time text generation with streaming output
- Parameter controls (max_length, temperature, top_k)
- Generation statistics (tokens/s, latency)
- Stop generation button
- Example prompts

**Usage:**
```python
from splitlearn_comm import GRPCComputeClient

client = GRPCComputeClient("localhost:50051")
client.connect()

client.launch_ui(
    bottom_model=bottom_model,
    top_model=top_model,
    tokenizer=tokenizer,
    theme="default",
    share=False,
    server_port=7860
)
```

**Run the example:**
```bash
python examples/client_ui_example.py
```

### 2. Server Monitoring UI (server_ui_example.py)

Real-time monitoring dashboard for server metrics and analytics.

**Features:**
- Real-time server statistics
- Request history with timestamps
- Compute time trends (graph)
- Success/failure rate tracking
- Auto-refresh every 2 seconds

**Usage:**
```python
from splitlearn_comm import GRPCComputeServer

server = GRPCComputeServer(compute_fn, port=50051)
server.start()

server.launch_monitoring_ui(
    theme="default",
    refresh_interval=2,
    share=False,
    server_port=7861,
    blocking=False  # Run in background
)

server.wait_for_termination()
```

**Run the example:**
```bash
python examples/server_ui_example.py
```

## Configuration Options

### Client UI Options

- `bottom_model`: Local bottom model (runs on client)
- `top_model`: Local top model (runs on client)
- `tokenizer`: Tokenizer for encoding/decoding text
- `theme`: UI theme ("default", "dark", "light")
- `share`: Create public Gradio link (default: False)
- `server_port`: Port for Gradio UI (default: 7860)

### Server Monitoring UI Options

- `theme`: UI theme ("default", "dark", "light")
- `refresh_interval`: Dashboard refresh rate in seconds (default: 2)
- `share`: Create public Gradio link (default: False)
- `server_port`: Port for monitoring UI (default: 7861)
- `blocking`: Run in main thread (True) or background (False)

## Themes

Three built-in themes are available:

1. **default**: Soft blue theme (recommended)
2. **dark**: Dark mode theme
3. **light**: Light mode theme

Example:
```python
client.launch_ui(..., theme="dark")
server.launch_monitoring_ui(..., theme="dark")
```

## Advanced Usage

### Non-Blocking Server Monitoring

Run the monitoring UI in a background thread while the server continues processing requests:

```python
server = GRPCComputeServer(compute_fn, port=50051)
server.start()

# Launch monitoring UI in background
server.launch_monitoring_ui(
    blocking=False,  # Key parameter
    server_port=7861
)

# Server continues running
server.wait_for_termination()
```

### Environment Variables

Client UI respects these environment variables:

- `GRADIO_SHARE`: Set to "true" to enable public links
- `GRADIO_PORT`: Custom port (default: 7860)

Example:
```bash
GRADIO_SHARE=false GRADIO_PORT=8080 python your_script.py
```

## Troubleshooting

### Import Error: "No module named 'gradio'"

Install UI dependencies:
```bash
pip install splitlearn-comm[ui]
```

### Port Already in Use

Change the server_port parameter:
```python
client.launch_ui(..., server_port=7870)
server.launch_monitoring_ui(..., server_port=7871)
```

### Connection Error

Ensure the server is running before launching the client UI:
```bash
# Terminal 1: Start server
python server_ui_example.py

# Terminal 2: Start client
python client_ui_example.py
```

## Performance Notes

- Client UI: Minimal overhead, only affects UI rendering
- Server Monitoring UI: ~1-2% CPU overhead for metrics collection
- Auto-refresh: Configurable via `refresh_interval` parameter
- Request history: Limited to last 100 requests (configurable in servicer)

## Security

**⚠️ Important:**
- Set `share=False` for production use
- Only enable `share=True` for temporary demos
- Use firewall rules to restrict access to UI ports
- Public Gradio links expire after 72 hours

## Next Steps

1. Modify the examples to use your own models
2. Customize UI themes and parameters
3. Integrate into your existing Split Learning pipeline
4. Monitor server performance in real-time
5. Share feedback and contribute improvements!

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/splitlearn-comm/issues
- Documentation: https://splitlearn-comm.readthedocs.io (coming soon)
