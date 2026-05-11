import os
from pathlib import Path

# Project root path (assumes config.py is in src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Results directory (using absolute path instead of relative string)
RESULTS_DIR = PROJECT_ROOT / "results"

# Benchmark setup
DATASETS = ["BNCI2014_009", "EPFLP300"]
# Test subject range. NOTE: Evaluates up to subject 10 if available, skips otherwise.
TEST_SUBJECTS = range(1, 11)

# ITR Calculation Duration
# 12 flashes × 0.175s SOA = 2.1s per character
TRIAL_DURATION = 2.1

# Feature extraction decimation (downsampling)
# Standard P300 downsampling: ~85 Hz * 1.0s window ≈ 85 samples; decimation=3 → ~28 features/ch
DECIMATION_FACTOR = 3
