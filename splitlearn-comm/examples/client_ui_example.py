"""
Example: Using the Client UI for interactive text generation

This example demonstrates how to use the built-in Gradio UI for client-side
text generation with Split Learning.
"""

import os
import torch
from transformers import AutoTokenizer

from splitlearn_comm import GRPCComputeClient

# This example assumes you have:
# 1. A running server (see server_example.py)
# 2. Bottom and Top models saved locally
# 3. Installed UI dependencies: pip install splitlearn-comm[ui]


def main():
    # Paths to your local models
    bottom_model_path = "path/to/bottom_model.pt"
    top_model_path = "path/to/top_model.pt"

    # Check if models exist
    if not os.path.exists(bottom_model_path) or not os.path.exists(top_model_path):
        print("❌ Model files not found!")
        print(f"   Bottom: {bottom_model_path}")
        print(f"   Top: {top_model_path}")
        print("\nPlease update the paths in this script.")
        return

    print("Loading models...")
    bottom_model = torch.load(bottom_model_path, map_location='cpu', weights_only=False)
    top_model = torch.load(top_model_path, map_location='cpu', weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    print("✓ Models loaded")

    # Connect to server
    print("\nConnecting to server...")
    client = GRPCComputeClient("localhost:50051", timeout=10.0)

    if not client.connect():
        print("❌ Failed to connect to server!")
        print("   Make sure the server is running on localhost:50051")
        return

    print("✓ Connected to server")

    # Launch the UI
    print("\n🚀 Launching Gradio UI...")
    print("   The UI will open in your browser automatically")
    print("   Access URL: http://127.0.0.1:7860")
    print("\nPress Ctrl+C to stop")

    try:
        client.launch_ui(
            bottom_model=bottom_model,
            top_model=top_model,
            tokenizer=tokenizer,
            theme="default",  # Options: "default", "dark", "light"
            share=False,      # Set to True to create a public link
            server_port=7860
        )
    except KeyboardInterrupt:
        print("\n\n✓ UI stopped")
    finally:
        client.close()
        print("✓ Connection closed")


if __name__ == "__main__":
    main()
