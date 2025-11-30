"""
Example: Using the Server Monitoring UI

This example demonstrates how to use the built-in Gradio UI for server-side
monitoring and analytics.
"""

import torch

from splitlearn_comm import GRPCComputeServer
from splitlearn_comm.core import ModelComputeFunction

# This example assumes you have:
# 1. A trunk model saved locally
# 2. Installed UI dependencies: pip install splitlearn-comm[ui]


def main():
    # Path to your trunk model
    trunk_model_path = "path/to/trunk_model.pt"

    print("Loading trunk model...")
    trunk_model = torch.load(trunk_model_path, map_location='cpu', weights_only=False)
    trunk_model.eval()
    print("✓ Trunk model loaded")

    # Create compute function
    compute_fn = ModelComputeFunction(
        model=trunk_model,
        device="cpu"  # Use "cuda" if available
    )

    # Create and start server
    print("\n🚀 Starting gRPC server...")
    server = GRPCComputeServer(
        compute_fn=compute_fn,
        host="0.0.0.0",
        port=50051
    )
    server.start()
    print("✓ Server started on 0.0.0.0:50051")

    # Launch monitoring UI in a separate thread
    print("\n📊 Launching monitoring UI...")
    print("   The UI will open in your browser automatically")
    print("   Monitoring URL: http://127.0.0.1:7861")
    print("   Server URL: 0.0.0.0:50051")
    print("\nPress Ctrl+C to stop")

    try:
        # Launch monitoring UI (blocking=False to run in background)
        server.launch_monitoring_ui(
            theme="default",       # Options: "default", "dark", "light"
            refresh_interval=2,    # Update every 2 seconds
            share=False,           # Set to True to create a public link
            server_port=7861,
            blocking=False         # Run in background thread
        )

        # Keep server running
        server.wait_for_termination()

    except KeyboardInterrupt:
        print("\n\n✓ Shutting down...")
    finally:
        server.stop()
        print("✓ Server stopped")


if __name__ == "__main__":
    main()
