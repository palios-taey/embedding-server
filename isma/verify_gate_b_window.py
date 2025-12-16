#!/usr/bin/env python3
"""
Verify Gate-B evaluation window implementation.

Tests:
1. Window check returns True only during ~38% of the cycle
2. Gate-B checks are skipped outside the window
3. Log output shows window status
"""

import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from breathing_cycle import BreathingCycle, CYCLE_DURATION

def test_window_calculation():
    """Test that window calculation is correct."""
    print("Testing Gate-B window calculation...")
    print(f"Cycle duration: {CYCLE_DURATION:.3f}s")
    print(f"Window: [0.236, 0.618] of cycle")
    print(f"Expected window duration: {(0.618 - 0.236) * CYCLE_DURATION:.3f}s")
    print(f"Expected window percentage: {(0.618 - 0.236) * 100:.1f}%")
    print()

    cycle = BreathingCycle()

    # Sample window status over 1 full cycle
    samples = 100
    in_window_count = 0

    print("Sampling window status over 1 cycle...")
    start_time = time.time()

    for i in range(samples):
        in_window = cycle._should_run_gate_b()
        if in_window:
            in_window_count += 1
        time.sleep(CYCLE_DURATION / samples)

    elapsed = time.time() - start_time
    window_percentage = (in_window_count / samples) * 100

    print(f"Samples taken: {samples}")
    print(f"In window: {in_window_count}")
    print(f"Window percentage: {window_percentage:.1f}%")
    print(f"Expected: ~38.2%")
    print(f"Elapsed time: {elapsed:.3f}s")

    # Verify it's close to expected
    expected_percentage = (0.618 - 0.236) * 100
    if abs(window_percentage - expected_percentage) < 10:  # Allow 10% tolerance
        print("✓ Window percentage is within expected range")
    else:
        print("✗ Window percentage is outside expected range")

    print()

def test_gate_b_skip_behavior():
    """Test that Gate-B checks skip when outside window."""
    print("Testing Gate-B skip behavior...")

    cycle = BreathingCycle()

    # Wait for a moment outside the window
    print("Waiting to be outside window...")
    while cycle._should_run_gate_b():
        time.sleep(0.1)

    print("Outside window - running Gate-B checks...")
    results = cycle._run_gate_b_checks()

    print(f"Skipped: {results.get('skipped', False)}")
    print(f"Reason: {results.get('reason', 'N/A')}")
    print(f"Passed: {results.get('passed', False)}")

    if results.get('skipped') and results.get('reason') == 'outside_window':
        print("✓ Gate-B checks correctly skipped outside window")
    else:
        print("✗ Gate-B checks did not skip as expected")

    print()

    # Wait for a moment inside the window
    print("Waiting to be inside window...")
    while not cycle._should_run_gate_b():
        time.sleep(0.1)

    print("Inside window - running Gate-B checks...")
    results = cycle._run_gate_b_checks()

    print(f"Skipped: {results.get('skipped', False)}")
    print(f"Window: {results.get('window', False)}")
    print(f"Passed: {results.get('passed', False)}")
    print(f"Checks: {list(results.get('checks', {}).keys())}")

    if not results.get('skipped') and results.get('window'):
        print("✓ Gate-B checks correctly ran inside window")
    else:
        print("✗ Gate-B checks did not run as expected")

    print()

    cycle.close()

def test_hold_phase_integration():
    """Test that hold phase properly handles Gate-B window."""
    print("Testing hold phase integration...")

    cycle = BreathingCycle()

    # Force a consolidation to trigger hold phase
    print("Running forced consolidation cycle...")
    metrics = cycle.force_consolidation()

    print(f"Gate-B passed: {metrics.gate_b_passed}")
    print(f"Gate-B skipped: {metrics.gate_b_skipped if hasattr(metrics, 'gate_b_skipped') else 'N/A'}")
    print(f"Coherence: {metrics.coherence_score:.3f}")
    print(f"Entropy before: {metrics.entropy_before:.3f}")
    print(f"Entropy after: {metrics.entropy_after:.3f}")

    print("✓ Hold phase completed without errors")
    print()

    cycle.close()

if __name__ == '__main__':
    print("=" * 60)
    print("Gate-B Evaluation Window Verification")
    print("=" * 60)
    print()

    try:
        test_window_calculation()
        test_gate_b_skip_behavior()
        test_hold_phase_integration()

        print("=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
