"""
Quickstart Server Example

This example demonstrates the simplest way to start a SplitLearn server.
It loads a model and serves it via gRPC.

Usage:
    python quickstart_server.py

Then run quickstart_client.py in another terminal to test the server.
"""

from splitlearn_manager.quickstart import ManagedServer

def main():
    print("=== SplitLearn Quickstart Server ===\n")

    print("Starting server with GPT-2 Trunk model...")
    print("  Model: gpt2")
    print("  Component: trunk")
    print("  Port: 50051")
    print("  Device: auto-detect (CUDA if available, else CPU)\n")

    print("Press Ctrl+C to stop the server\n")
    print("-" * 50)

    # Create and start server (blocking)
    server = ManagedServer(
        model_type="gpt2",
        component="trunk",
        port=50051
    )

    # This will block until Ctrl+C
    server.start()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
