"""
Minimal Split Learning Example

This example demonstrates a complete Split Learning setup with:
- Client: Bottom + Top models (lightweight)
- Server: Trunk model (heavyweight)

This requires running server and client in separate processes/machines.

Components:
    1. prepare_models(): Download and split the model (run once)
    2. run_server(): Start server with Trunk model
    3. run_client(): Run client with Bottom + Top models
"""

import torch
from splitlearn.quickstart import load_split_model
from splitlearn_comm.quickstart import Client, Server


def prepare_models():
    """
    Step 1: Prepare split models (run this once)

    Downloads the model and splits it into Bottom/Trunk/Top.
    Models are cached locally for future use.
    """
    print("=== Preparing Split Models ===\n")

    print("Downloading and splitting GPT-2 model...")
    print("Split points: Bottom(0-2), Trunk(2-10), Top(10-12)\n")

    bottom, trunk, top = load_split_model(
        model_type="gpt2",
        split_points=[2, 10],
        cache_dir="./models"  # Save to local cache
    )

    print("\n✓ Models prepared successfully!")
    print(f"  Bottom: {sum(p.numel() for p in bottom.parameters())/1e6:.2f}M parameters")
    print(f"  Trunk:  {sum(p.numel() for p in trunk.parameters())/1e6:.2f}M parameters")
    print(f"  Top:    {sum(p.numel() for p in top.parameters())/1e6:.2f}M parameters")

    return bottom, trunk, top


def run_server():
    """
    Step 2: Run server (Trunk model)

    Start a gRPC server that serves the Trunk model.
    This should run on a machine with high computational power (GPU recommended).

    Usage:
        python -c "from split_learning_minimal import run_server; run_server()"
    """
    print("=== Starting Server (Trunk Model) ===\n")

    # Load Trunk model
    _, trunk, _ = load_split_model(
        "gpt2",
        split_points=[2, 10],
        cache_dir="./models"
    )

    # Start server
    print("Server starting on port 50051...")
    print("Press Ctrl+C to stop\n")

    server = Server(
        model=trunk,
        port=50051,
        device="cuda"  # Change to "cpu" if no GPU
    )

    server.start()
    server.wait_for_termination()


def run_client():
    """
    Step 3: Run client (Bottom + Top models)

    Connects to the server and performs split learning inference.
    This can run on a resource-limited device (e.g., laptop, mobile).

    Usage:
        python -c "from split_learning_minimal import run_client; run_client()"
    """
    print("=== Running Client (Bottom + Top Models) ===\n")

    # Load Bottom and Top models
    print("Loading Bottom and Top models...")
    bottom, _, top = load_split_model(
        "gpt2",
        split_points=[2, 10],
        cache_dir="./models"
    )
    bottom.eval()
    top.eval()

    # Connect to server
    print("Connecting to server at localhost:50051...")
    client = Client("localhost:50051")
    print("✓ Connected\n")

    # Prepare input
    print("Preparing input (random tokens for demonstration)...")
    input_ids = torch.randint(0, 50257, (1, 10))  # (batch=1, seq_len=10)
    print(f"Input shape: {input_ids.shape}\n")

    # === Split Learning Inference ===
    print("=== Split Learning Inference ===\n")

    # Step 1: Client-side - Bottom model
    print("Step 1: Running Bottom model on client...")
    with torch.no_grad():
        bottom_output = bottom(input_ids)
    print(f"  Bottom output shape: {bottom_output.shape}")
    print(f"  Data size: {bottom_output.numel() * 4 / 1024:.2f} KB\n")

    # Step 2: Send to server - Trunk model
    print("Step 2: Sending to server for Trunk computation...")
    with torch.no_grad():
        trunk_output = client.compute(bottom_output)
    print(f"  Trunk output shape: {trunk_output.shape}")
    print(f"  Data size: {trunk_output.numel() * 4 / 1024:.2f} KB\n")

    # Step 3: Client-side - Top model
    print("Step 3: Running Top model on client...")
    with torch.no_grad():
        final_output = top(trunk_output)
    print(f"  Final output shape: {final_output.shape}\n")

    # Get predictions
    logits = final_output
    predicted_ids = torch.argmax(logits, dim=-1)
    print(f"Predicted token IDs: {predicted_ids[0].tolist()}\n")

    # Close connection
    client.close()
    print("✓ Client completed successfully!")


def main():
    """
    Main entry point - demonstrates the workflow
    """
    import sys

    print("=== Split Learning Minimal Example ===\n")
    print("This example demonstrates a complete Split Learning setup.\n")
    print("Choose an option:")
    print("  1. Prepare models (download and split)")
    print("  2. Run server (Trunk model)")
    print("  3. Run client (Bottom + Top models)")
    print("  4. Prepare and show info only\n")

    try:
        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            prepare_models()
            print("\nNext steps:")
            print("  - Run server: python -c 'from split_learning_minimal import run_server; run_server()'")
            print("  - Run client: python -c 'from split_learning_minimal import run_client; run_client()'")

        elif choice == "2":
            run_server()

        elif choice == "3":
            run_client()

        elif choice == "4":
            bottom, trunk, top = prepare_models()
            print("\nModels are ready!")
            print("\nTo run:")
            print("  Terminal 1: python -c 'from split_learning_minimal import run_server; run_server()'")
            print("  Terminal 2: python -c 'from split_learning_minimal import run_client; run_client()'")

        else:
            print("Invalid choice. Please run again.")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
