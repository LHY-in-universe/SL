#!/usr/bin/env python3
"""
Basic Client Monitoring Example

This example demonstrates how to add monitoring to a Split Learning client.
"""
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from splitlearn_monitor import ClientMonitor


def simulate_bottom_model(data):
    """Simulate bottom model processing"""
    time.sleep(0.05)  # Simulate computation
    return data * 2


def simulate_trunk_call(data):
    """Simulate remote trunk server call"""
    time.sleep(0.1)  # Simulate network + server computation
    return data + 1


def simulate_top_model(data):
    """Simulate top model processing"""
    time.sleep(0.03)  # Simulate computation
    return data ** 2


def main():
    print("=" * 80)
    print("Split Learning Client Monitoring Example")
    print("=" * 80)
    print()

    # Create monitor
    monitor = ClientMonitor(
        session_name="example_client",
        sampling_interval=0.1,
        enable_gpu=True
    )

    # Start monitoring
    print("Starting monitoring...")
    monitor.start()

    # Simulate inference loop
    print("Running inference loop (10 steps)...")
    for i in range(10):
        # Track bottom model
        with monitor.track_phase("bottom_model"):
            data = simulate_bottom_model(i)

        # Track trunk remote call
        with monitor.track_phase("trunk_remote"):
            data = simulate_trunk_call(data)

        # Track top model
        with monitor.track_phase("top_model"):
            result = simulate_top_model(data)

        if i % 3 == 0:
            print(f"  Step {i+1}/10 completed, result: {result}")

    print("Inference complete!")

    # Stop monitoring
    print("\nStopping monitoring...")
    monitor.stop()

    # Print summary
    monitor.print_summary()

    # Save reports
    print("Generating reports...")
    html_report = monitor.save_report(format="html")
    json_report = monitor.save_report(format="json")

    print(f"\nReports generated:")
    print(f"  HTML: {html_report}")
    print(f"  JSON: {json_report}")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
