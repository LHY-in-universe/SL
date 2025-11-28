#!/usr/bin/env python3
"""
Client Example for Managed Server

Demonstrates how to connect to a managed server and perform inference.
"""

import logging
import torch

from splitlearn_comm import GRPCComputeClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    print("=" * 70)
    print("Managed Server Client Example")
    print("=" * 70)

    # Connect to managed server
    print("\n[1] Connecting to managed server...")
    client = GRPCComputeClient("localhost:50051")

    if not client.connect():
        print("✗ Failed to connect to server")
        print("  Make sure managed server is running (python examples/basic_server.py)")
        return

    print("✓ Connected to server")

    # Get service information
    print("\n[2] Getting service information...")
    info = client.get_service_info()
    print(f"  Service: {info['service_name']}")
    print(f"  Models loaded: {info.get('custom_info', {}).get('num_models', 0)}")

    # Perform inference requests
    print("\n[3] Performing inference requests...")
    num_requests = 10

    for i in range(num_requests):
        # Create random input
        input_tensor = torch.randn(1, 10, 768)

        # Compute
        output_tensor = client.compute(input_tensor)

        print(f"  Request {i+1}: {input_tensor.shape} → {output_tensor.shape}")

    # Get statistics
    print("\n[4] Client statistics:")
    stats = client.get_statistics()
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Avg network time: {stats['avg_network_time_ms']:.2f}ms")
    print(f"  Avg compute time: {stats['avg_compute_time_ms']:.2f}ms")
    print(f"  Avg total time: {stats['avg_total_time_ms']:.2f}ms")

    # Close connection
    print("\n[5] Closing connection...")
    client.close()
    print("✓ Done!")


if __name__ == "__main__":
    main()
