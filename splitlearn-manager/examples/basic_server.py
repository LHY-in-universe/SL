#!/usr/bin/env python3
"""
Basic Managed Server Example

Demonstrates how to:
1. Create a managed server
2. Load models dynamically
3. Serve requests with automatic routing
"""

import logging
import torch.nn as nn

from splitlearn_manager import ManagedServer, ModelConfig, ServerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def create_simple_model():
    """Create a simple PyTorch model for demonstration."""
    return nn.Sequential(
        nn.Linear(768, 1024),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(1024, 768)
    )


def main():
    print("=" * 70)
    print("Basic Managed Server Example")
    print("=" * 70)

    # 1. Create server configuration
    print("\n[1] Creating server configuration...")
    server_config = ServerConfig(
        host="0.0.0.0",
        port=50051,
        max_workers=10,
        max_models=3,
        enable_monitoring=True,
        metrics_port=8000,
        log_level="INFO"
    )

    # 2. Create and start managed server
    print("\n[2] Creating managed server...")
    server = ManagedServer(config=server_config)

    # 3. Load a model
    print("\n[3] Loading model...")

    # Create and save a simple model
    import torch
    model = create_simple_model()
    model_path = "/tmp/demo_model.pt"
    torch.save(model, model_path)

    # Create model configuration
    model_config = ModelConfig(
        model_id="demo_model_1",
        model_path=model_path,
        model_type="pytorch",
        device="cpu",
        batch_size=32,
        warmup=True,
        config={"input_shape": (1, 10, 768)}
    )

    # Load the model
    server.load_model(model_config)
    print(f"✓ Model {model_config.model_id} loaded successfully")

    # 4. Start server
    print("\n[4] Starting server...")
    server.start()

    print(f"\n✓ Server running on {server_config.host}:{server_config.port}")
    print(f"✓ Metrics available at http://localhost:{server_config.metrics_port}")
    print("\nPress Ctrl+C to stop...")

    # 5. Wait for termination
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n\n[5] Shutting down...")
        server.stop()
        print("✓ Server stopped")


if __name__ == "__main__":
    main()
