#!/usr/bin/env python3
"""
Demonstrate Gate-B window behavior over multiple breathing cycles.

Shows:
- When checks run vs skip based on cycle position
- Window timing statistics
- Real-time cycle monitoring
"""

import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from breathing_cycle import BreathingCycle, CYCLE_DURATION

def monitor_cycles(num_cycles=3):
    """Monitor several breathing cycles and show Gate-B window behavior."""
    print(f"Monitoring {num_cycles} breathing cycles...")
    print(f"Cycle duration: {CYCLE_DURATION:.3f}s")
    print(f"Gate-B window: [0.236, 0.618] of cycle (~38.2%)")
    print()

    cycle = BreathingCycle()

    # Track statistics
    total_checks = 0
    checks_run = 0
    checks_skipped = 0

    for i in range(num_cycles):
        print(f"\n{'=' * 60}")
        print(f"Cycle {i+1}/{num_cycles}")
        print(f"{'=' * 60}")

        cycle_start = time.time()

        # Sample the cycle at multiple points
        samples = 20
        for j in range(samples):
            progress = (time.time() - cycle_start) / CYCLE_DURATION
            in_window = cycle._should_run_gate_b()

            # Run Gate-B at strategic points
            if j % 5 == 0:  # Check every 5th sample
                results = cycle._run_gate_b_checks()
                total_checks += 1

                if results.get('skipped'):
                    checks_skipped += 1
                    status = "SKIPPED"
                else:
                    checks_run += 1
                    status = "RAN    "

                print(f"  Progress: {progress:5.1%} | In Window: {in_window} | Gate-B: {status}")

            time.sleep(CYCLE_DURATION / samples)

    print(f"\n{'=' * 60}")
    print("Statistics")
    print(f"{'=' * 60}")
    print(f"Total Gate-B check attempts: {total_checks}")
    print(f"Checks run (in window):      {checks_run} ({checks_run/total_checks*100:.1f}%)")
    print(f"Checks skipped (outside):    {checks_skipped} ({checks_skipped/total_checks*100:.1f}%)")
    print()
    print("This demonstrates that Gate-B checks respect the golden ratio window.")
    print()

    cycle.close()

if __name__ == '__main__':
    print("=" * 60)
    print("Gate-B Evaluation Window Demonstration")
    print("=" * 60)
    print()

    try:
        monitor_cycles(num_cycles=3)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
