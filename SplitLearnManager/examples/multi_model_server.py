#!/usr/bin/env python3
"""
Multi-Model Server Example

Demonstrates:
1. Loading multiple models
2. Dynamic model loading/unloading
3. Monitoring and health checks
"""

import logging
import time
import torch
import torch.nn as nn

from splitlearn_manager import ManagedServer, ModelConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def create_model(hidden_size: int):
    """Create a model with specific hidden size."""
    return nn.Sequential(
        nn.Linear(768, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 768)
    )


def main():
    print("=" * 70)
    print("Multi-Model Server Example")
    print("=" * 70)

    # Create server
    server = ManagedServer()

    # Create and load multiple models
    model_sizes = [512, 1024, 2048]

    print("\n[1] Creating and loading models...")
    for i, size in enumerate(model_sizes):
        model_id = f"model_{size}"

        # Create and save model
        model = create_model(size)
        model_path = f"/tmp/{model_id}.pt"
        torch.save(model, model_path)

        # Create config
        config = ModelConfig(
            model_id=model_id,
            model_path=model_path,
            model_type="pytorch",
            device="cpu",
            config={"input_shape": (1, 10, 768)}
        )

        # Load model
        server.load_model(config)
        print(f"  ✓ Loaded {model_id}")

    # Start server
    print("\n[2] Starting server...")
    server.start()

    print("\n✓ Server started with models:")
    for model_info in server.model_manager.list_models():
        print(f"  - {model_info['model_id']}: {model_info['status']}")

    # Demonstrate dynamic model management
    print("\n[3] Demonstrating dynamic management...")
    time.sleep(2)

    # Check status
    status = server.get_status()
    print(f"\nCurrent status:")
    print(f"  Models loaded: {status['statistics']['total_models']}")
    print(f"  Max models: {status['statistics']['max_models']}")

    # Unload one model
    print(f"\n[4] Unloading model_512...")
    server.unload_model("model_512")

    # Load a new model
    print(f"\n[5] Loading new model...")
    new_model = create_model(4096)
    new_path = "/tmp/model_4096.pt"
    torch.save(new_model, new_path)

    new_config = ModelConfig(
        model_id="model_4096",
        model_path=new_path,
        model_type="pytorch",
        device="cpu"
    )
    server.load_model(new_config)

    # Show final state
    print("\n✓ Final model state:")
    for model_info in server.model_manager.list_models():
        print(f"  - {model_info['model_id']}: {model_info['status']}")

    print("\nPress Ctrl+C to stop...")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        server.stop()


if __name__ == "__main__":
    main()
