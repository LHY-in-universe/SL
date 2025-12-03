"""
Quickstart Client Example

This example demonstrates the simplest way to use the SplitLearn client.
It connects to a server and sends a computation request.

Usage:
    1. Start the server first (see quickstart_server.py)
    2. Run this script: python quickstart_client.py
"""

import torch
from splitlearn_comm.quickstart import Client

def main():
    print("=== SplitLearn Quickstart Client ===\n")

    # Step 1: Create client (automatically connects)
    print("1. Connecting to server at localhost:50051...")
    client = Client("localhost:50051")
    print("   ✓ Connected\n")

    # Step 2: Prepare input tensor
    print("2. Preparing input tensor...")
    input_tensor = torch.randn(1, 10, 768)  # (batch, seq_len, hidden_size)
    print(f"   Input shape: {input_tensor.shape}")
    print(f"   Input size: {input_tensor.numel() * 4 / 1024:.2f} KB\n")

    # Step 3: Send computation request
    print("3. Sending computation request to server...")
    output_tensor = client.compute(input_tensor)
    print(f"   ✓ Received response")
    print(f"   Output shape: {output_tensor.shape}")
    print(f"   Output size: {output_tensor.numel() * 4 / 1024:.2f} KB\n")

    # Step 4: Close connection
    print("4. Closing connection...")
    client.close()
    print("   ✓ Connection closed\n")

    print("=== Client completed successfully! ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nClient interrupted by user")
    except ConnectionRefusedError:
        print("\nError: Could not connect to server.")
        print("Please make sure the server is running (quickstart_server.py)")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
