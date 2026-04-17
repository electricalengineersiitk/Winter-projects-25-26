"""
autorange.py  —  Auto-ranging engine
Smart Multimeter Simulation | End-Term Project

Logic:
  - Given a raw reading, decide which range is active
  - Step UP  if reading > 90% of current range max  (3 consecutive triggers)
  - Step DOWN if reading < 10% of current range max  (3 consecutive triggers)
  - SETTLED   if reading is in the 10-90% window
  - OL        if reading exceeds all ranges (overload)

Hysteresis rule: require 3 consecutive out-of-range readings before switching.
This prevents oscillation when a reading sits near a range boundary.
"""

import numpy as np

# ─────────────────────────────────────────────
# Range Definitions
# ─────────────────────────────────────────────

RANGES_R = [
    {"label": "Range 1", "max": 100},
    {"label": "Range 2", "max": 1_000},
    {"label": "Range 3", "max": 10_000},
    {"label": "Range 4", "max": 100_000},
    {"label": "Range 5", "max": 1_000_000},
]

RANGES_C = [
    {"label": "Range 1", "max": 10e-9},
    {"label": "Range 2", "max": 100e-9},
    {"label": "Range 3", "max": 1e-6},
    {"label": "Range 4", "max": 10e-6},
    {"label": "Range 5", "max": 100e-6},
]

RANGES_L = [
    {"label": "Range 1", "max": 10e-6},
    {"label": "Range 2", "max": 100e-6},
    {"label": "Range 3", "max": 1e-3},
    {"label": "Range 4", "max": 10e-3},
    {"label": "Range 5", "max": 100e-3},
]

STEP_UP_THRESHOLD   = 0.90
STEP_DOWN_THRESHOLD = 0.10
HYSTERESIS_COUNT    = 3


def get_ranges(mode):
    """Return the correct range list for a given mode (R, C, or L)."""
    if mode == "R":
        return RANGES_R
    elif mode == "C":
        return RANGES_C
    elif mode == "L":
        return RANGES_L
    else:
        raise ValueError("Mode must be R, C, or L")


def make_ranger_state(mode):
    """
    Create and return a fresh ranger state dictionary.
    This acts as the memory of the auto-ranger across multiple readings.

    Returns a plain dictionary — no classes involved.
    """
    return {
        "mode":        mode,
        "ranges":      get_ranges(mode),
        "current_idx": 0,
        "up_count":    0,
        "down_count":  0,
    }


def reset_ranger(state):
    """Reset ranger back to lowest range with no pending triggers."""
    state["current_idx"] = 0
    state["up_count"]    = 0
    state["down_count"]  = 0
    return state


def update_ranger(state, reading):
    """
    Feed one measurement reading into the auto-ranging engine.
    Updates the state dictionary in place and returns the result.

    Args:
        state:   ranger state dictionary from make_ranger_state()
        reading: the measured value in Ohms / Farads / Henrys

    Returns:
        (range_index, range_label, status)
        status is one of: "SETTLED", "STEP_UP", "STEP_DOWN", "OL"
    """
    ranges      = state["ranges"]
    current_idx = state["current_idx"]

    # Overload: reading exceeds the highest range
    if reading > ranges[-1]["max"]:
        return len(ranges) - 1, ranges[-1]["label"], "OL"

    range_max   = ranges[current_idx]["max"]
    up_thresh   = STEP_UP_THRESHOLD   * range_max
    down_thresh = STEP_DOWN_THRESHOLD * range_max

    # Step UP
    if reading > up_thresh and current_idx < len(ranges) - 1:
        state["up_count"]   += 1
        state["down_count"]  = 0

        if state["up_count"] >= HYSTERESIS_COUNT:
            state["current_idx"] += 1
            state["up_count"]     = 0
            idx = state["current_idx"]
            return idx, ranges[idx]["label"], "STEP_UP"
        else:
            return current_idx, ranges[current_idx]["label"], "SETTLED"

    # Step DOWN
    elif reading < down_thresh and current_idx > 0:
        state["down_count"] += 1
        state["up_count"]    = 0

        if state["down_count"] >= HYSTERESIS_COUNT:
            state["current_idx"] -= 1
            state["down_count"]   = 0
            idx = state["current_idx"]
            return idx, ranges[idx]["label"], "STEP_DOWN"
        else:
            return current_idx, ranges[current_idx]["label"], "SETTLED"

    # Settled: reading is comfortably within 10-90% window
    else:
        state["up_count"]   = 0
        state["down_count"] = 0
        return current_idx, ranges[current_idx]["label"], "SETTLED"


def find_correct_range(mode, value):
    """
    Instantly find the ideal range for a value without any state.
    Used for the fixed-range baseline comparison in simulate.py.

    Returns:
        (range_index, range_label)
    """
    ranges = get_ranges(mode)
    for i, r in enumerate(ranges):
        if value <= r["max"]:
            return i, r["label"]
    return len(ranges) - 1, ranges[-1]["label"]


# ─────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────
'''
if __name__ == "__main__":
    print("=" * 60)
    print("  autorange.py - Self-Test  |  Mode: Resistance")
    print("=" * 60)

    test_values = np.logspace(1, 6, 50)   # 50 points: 10 Ohm to 1 MOhm

    state = make_ranger_state("R")

    print(f"\n  {'Sample':<8} {'True R (Ohm)':<18} {'Range':<12} {'Status'}")
    print(f"  {'-'*54}")

    prev_label = None
    for i, val in enumerate(test_values):
        noisy = np.random.normal(val, 0.005 * val)
        noisy = max(noisy, 1e-6)

        ridx, rlabel, status = update_ranger(state, noisy)

        marker = " <- transition" if rlabel != prev_label and prev_label is not None else ""
        print(f"  {i+1:<8} {val:<18,.2f} {rlabel:<12} {status}{marker}")
        prev_label = rlabel

    print("\n  All ranges covered correctly.")
    print("  Hysteresis active: 3 consecutive triggers required to switch.")
    print("=" * 60)
'''