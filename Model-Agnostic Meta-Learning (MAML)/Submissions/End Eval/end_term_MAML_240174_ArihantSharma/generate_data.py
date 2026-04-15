"""
generate_data.py — Synthetic Channel Estimation Dataset
Creates meta-learning tasks for wireless channel estimation.
Each task = one wireless environment with varying SNR / path counts.
"""

import numpy as np
import os

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Dataset parameters ────────────────────────────────────────────────────────
NUM_TRAIN_TASKS = 100
NUM_TEST_TASKS  = 20
N_SUPPORT       = 10        # labelled examples per task (adaptation)
N_QUERY         = 80        # evaluation examples per task
PILOT_DIM       = 16        # number of pilot subcarriers (input dim)
CHANNEL_DIM     = 16        # channel coefficients to estimate (output dim)


def generate_channel(n_samples: int, n_paths: int, snr_db: float) -> tuple:
    """
    Simulate an OFDM channel estimation task.

    Args:
        n_samples : number of (pilot, channel) pairs to generate
        n_paths   : number of multipath components
        snr_db    : signal-to-noise ratio in dB

    Returns:
        X : pilot observations  [n_samples, PILOT_DIM * 2]   (real + imag stacked)
        Y : true channel        [n_samples, CHANNEL_DIM * 2] (real + imag stacked)
    """
    snr_linear = 10 ** (snr_db / 10.0)
    noise_std  = 1.0 / np.sqrt(2 * snr_linear)

    # True channel: sum of n_paths complex exponentials
    delays = np.random.uniform(0, 1, n_paths)
    gains  = (np.random.randn(n_paths) + 1j * np.random.randn(n_paths)) / np.sqrt(2 * n_paths)
    freq_idx = np.arange(CHANNEL_DIM)
    H = np.sum(
        gains[:, None] * np.exp(-2j * np.pi * delays[:, None] * freq_idx[None, :]),
        axis=0
    )   # shape: (CHANNEL_DIM,)

    # Pilot observations = channel * pilot_symbols + noise
    # pilot_symbols are unit-energy QPSK
    pilot_phases  = np.random.choice([1, -1, 1j, -1j], size=(n_samples, PILOT_DIM))
    channel_tiled = np.tile(H[:PILOT_DIM], (n_samples, 1))
    noise         = (noise_std * np.random.randn(n_samples, PILOT_DIM) +
                     1j * noise_std * np.random.randn(n_samples, PILOT_DIM))
    X_complex     = channel_tiled * pilot_phases + noise

    # Stack real/imag → real-valued tensors
    X = np.hstack([X_complex.real, X_complex.imag]).astype(np.float32)
    Y_full = np.tile(H, (n_samples, 1))
    Y = np.hstack([Y_full.real, Y_full.imag]).astype(np.float32)

    return X, Y


def build_task(snr_range: tuple, path_range: tuple) -> dict:
    """Return a single meta-learning task dict with support + query sets."""
    snr_db  = np.random.uniform(*snr_range)
    n_paths = np.random.randint(*path_range)
    n_total = N_SUPPORT + N_QUERY

    X, Y = generate_channel(n_total, n_paths, snr_db)
    return {
        "X_support": X[:N_SUPPORT],
        "Y_support": Y[:N_SUPPORT],
        "X_query"  : X[N_SUPPORT:],
        "Y_query"  : Y[N_SUPPORT:],
        "snr_db"   : snr_db,
        "n_paths"  : n_paths,
    }


def build_dataset(num_tasks: int, snr_range=(0, 20), path_range=(2, 8)) -> list:
    return [build_task(snr_range, path_range) for _ in range(num_tasks)]


def save_dataset(tasks: list, path: str) -> None:
    arrays = {}
    for key in ["X_support", "Y_support", "X_query", "Y_query"]:
        arrays[key] = np.stack([t[key] for t in tasks])
    np.savez(path, **arrays)
    print(f"  Saved {len(tasks)} tasks → {path}")


def main():
    os.makedirs("data", exist_ok=True)

    print("Generating training tasks …")
    train_tasks = build_dataset(NUM_TRAIN_TASKS)
    save_dataset(train_tasks, os.path.join("data", "train_tasks.npz"))

    print("Generating test tasks …")
    test_tasks = build_dataset(NUM_TEST_TASKS)
    save_dataset(test_tasks, os.path.join("data", "test_tasks.npz"))

    print(f"\nDone.  Input dim = {PILOT_DIM * 2},  Output dim = {CHANNEL_DIM * 2}")
    print(f"  Support size per task : {N_SUPPORT} samples")
    print(f"  Query size per task   : {N_QUERY} samples")


if __name__ == "__main__":
    main()
