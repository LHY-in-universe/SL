#!/usr/bin/env python3
"""
Quick Monitor Demo

Demonstrates the quick_monitor context manager for simple monitoring tasks.
"""
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from splitlearn_monitor.integrations.client_monitor import quick_monitor


def simulate_workload():
    """Simulate some workload"""
    time.sleep(0.1)


def main():
    print("=" * 80)
    print("Quick Monitor Demo")
    print("=" * 80)
    print()

    # Use quick_monitor context manager
    # Automatically starts monitoring, tracks code, and saves report
    with quick_monitor("quick_demo", auto_save=True) as monitor:
        print("Running monitored code...")

        # Track different phases
        for i in range(5):
            with monitor.track_phase("phase_a"):
                time.sleep(0.05)

            with monitor.track_phase("phase_b"):
                time.sleep(0.08)

            with monitor.track_phase("phase_c"):
                time.sleep(0.03)

            print(f"  Iteration {i+1}/5 completed")

    # Report is automatically saved when exiting context

    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
