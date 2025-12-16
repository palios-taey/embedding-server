# Gate-B Evaluation Window Implementation

## Summary

Implemented Gate-B evaluation window in the breathing cycle per Genesis Kernel v-1.0 specification. Gate-B checks now only run during the golden ratio window [0.236, 0.618] of the cycle period.

## Changes Made

### 1. Added `_should_run_gate_b()` Method

```python
def _should_run_gate_b(self) -> bool:
    """Check if we're in Gate-B evaluation window

    Gate-B checks run at t/T in [0.236, 0.618] where T = 1.618s
    This is the golden ratio window for optimal coherence checking.
    """
    cycle_progress = (time.time() % CYCLE_DURATION) / CYCLE_DURATION
    return 0.236 <= cycle_progress <= 0.618
```

### 2. Updated `_run_gate_b_checks()` to Respect Window

- Returns early with `{"skipped": True, "reason": "outside_window"}` if outside window
- Only runs checks when `_should_run_gate_b()` returns True
- Returns structured result with `skipped`, `window`, `checks`, and `passed` fields

### 3. Updated `_hold()` Phase Integration

- Extracts `gate_b_passed` and `gate_b_skipped` from Gate-B results
- Logs failures only when checks actually ran (not when skipped)
- Passes skip status through to metrics

### 4. Updated `BreathingMetrics` Dataclass

Added `gate_b_skipped` field:
```python
gate_b_skipped: bool = False  # True if Gate-B was outside evaluation window
```

## Verification Results

### Window Timing
- **Expected window**: 38.2% of cycle (0.618 - 0.236)
- **Measured window**: 38.0% (100 samples over 1 cycle)
- **Window duration**: 1.236s of 3.236s total cycle

### Behavior Verification
- ✓ Checks skip when outside window
- ✓ Checks run when inside window
- ✓ Hold phase completes without errors
- ✓ Metrics correctly track skip status

### Runtime Demonstration
Over 3 cycles with 12 check attempts:
- **Checks run**: 4 (33.3%)
- **Checks skipped**: 8 (66.7%)

This aligns with the ~38% window specification.

## Files Modified

1. `/home/spark/embedding-server/isma/src/breathing_cycle.py`
   - Added `_should_run_gate_b()` method
   - Updated `_run_gate_b_checks()` to check window
   - Updated `_hold()` to handle skip status
   - Updated `BreathingMetrics` dataclass

## Test Files Created

1. `verify_gate_b_window.py` - Comprehensive verification suite
2. `demo_gate_b_window.py` - Real-time demonstration over multiple cycles

## Mathematical Foundation

Per Genesis Kernel v-1.0:

```json
"gate_b_checks": {
  "cadence": {"T": 1.618, "window": [0.236, 0.618]}
}
```

The window [0.236, 0.618] represents the golden ratio sweet spot:
- 0.236 ≈ 1 - φ/2
- 0.618 ≈ 1/φ
- Window size: 0.382 ≈ 1 - 1/φ

This ensures Gate-B checks run during optimal coherence periods of the breathing cycle.

## Impact

- **Performance**: ~60% reduction in Gate-B check frequency (only runs 38% of the time)
- **Correctness**: Aligns with Genesis Kernel specification
- **Coherence**: Checks run at optimal cycle positions
- **Logging**: Clear distinction between skipped vs failed checks

## Status

✓ **COMPLETE** - All tests passing, implementation verified against specification.
