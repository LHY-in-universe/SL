#!/usr/bin/env python3
"""
Basic Server Monitoring Example

This example demonstrates how to add monitoring to a Split Learning server.
"""
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from splitlearn_monitor import ServerMonitor


def simulate_request_processing():
    """Simulate processing a compute request"""
    time.sleep(0.05)  # Simulate model computation
    return "result"


def main():
    print("=" * 80)
    print("Split Learning Server Monitoring Example")
    print("=" * 80)
    print()

    # Create monitor
    monitor = ServerMonitor(
        server_name="trunk_server",
        sampling_interval=0.1,
        enable_gpu=True
    )

    # Start monitoring
    print("Starting server monitoring...")
    monitor.start()

    # Simulate handling requests
    print("Simulating incoming requests (20 requests)...")
    for i in range(20):
        # Track request processing
        with monitor.track_request("compute"):
            result = simulate_request_processing()

        if i % 5 == 0:
            print(f"  Processed {i+1}/20 requests")
            monitor.print_status()

    print("\nRequest processing complete!")

    # Stop monitoring
    print("Stopping monitoring...")
    monitor.stop()

    # Get statistics
    stats = monitor.get_current_stats()
    print("\nFinal Statistics:")
    print(f"  Total requests: {stats.get('phase_stats', {}).get('request_compute', {}).get('count', 0)}")

    # Save report
    print("\nGenerating report...")
    report_path = monitor.save_report(format="html")
    print(f"Report saved: {report_path}")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
