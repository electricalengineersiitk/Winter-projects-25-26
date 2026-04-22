"""
generate_data.py  —  Synthetic OFDM Channel Estimation Dataset

Creates wireless channel estimation tasks for meta-learning.
Each task = one wireless environment with a unique channel and SNR.

Setup
-----
  OFDM system with 32 subcarriers, 8 evenly-spaced pilot subcarriers.
  Channel has 2-6 multipath components with exponential power-delay profile.
  Input :  noisy received pilots   (real + imag  →  16 dims)
  Output:  full channel response   (real + imag  →  64 dims)

Generates 100 training tasks and 20 test tasks.
Each task has a support set (up to 20 shots) and a query set (50 samples).
"""

import os
import numpy as np

# ───────────────── Configuration ─────────────────
N_SUB       = 32        # OFDM subcarriers
N_PILOT     = 8         # pilot subcarriers
PILOT_IDX   = np.arange(0, N_SUB, N_SUB // N_PILOT)

N_TRAIN_TASKS = 100
N_TEST_TASKS  = 20
N_SUPPORT     = 20      # support samples per task (use 5/10/20-shot subsets)
N_QUERY       = 50      # query samples per task

SEED     = 42
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ───────────────── Helpers ───────────────────────
def generate_channel(n_sub, rng):
    """Random frequency-selective channel H[k] with L multipath components."""
    n_paths = rng.integers(2, 7)
    delays  = rng.choice(n_sub, size=n_paths, replace=False).astype(float)
    tau_rms = rng.uniform(1.0, 5.0)
    powers  = np.exp(-delays / tau_rms)
    powers /= powers.sum()
    gains   = np.sqrt(powers) * (
        rng.standard_normal(n_paths) + 1j * rng.standard_normal(n_paths)
    ) / np.sqrt(2)

    k = np.arange(n_sub)
    H = np.zeros(n_sub, dtype=complex)
    for l in range(n_paths):
        H += gains[l] * np.exp(-1j * 2 * np.pi * k * delays[l] / n_sub)
    return H


def generate_samples(H, pilot_idx, n_samples, snr_db, rng):
    """Noisy pilot observations (X) and ground-truth full channel (Y)."""
    H_pilot       = H[pilot_idx]
    signal_power  = np.mean(np.abs(H_pilot) ** 2)
    noise_power   = signal_power / (10 ** (snr_db / 10))

    X_all, Y_all = [], []
    y_out = np.concatenate([H.real, H.imag]).astype(np.float32)

    for _ in range(n_samples):
        noise = np.sqrt(noise_power / 2) * (
            rng.standard_normal(len(pilot_idx))
            + 1j * rng.standard_normal(len(pilot_idx))
        )
        y_rx  = H_pilot + noise
        x_in  = np.concatenate([y_rx.real, y_rx.imag]).astype(np.float32)
        X_all.append(x_in)
        Y_all.append(y_out)

    return np.array(X_all), np.array(Y_all)


def generate_task(rng):
    """One meta-learning task: support set + query set."""
    H      = generate_channel(N_SUB, rng)
    snr_db = rng.uniform(5.0, 25.0)
    X_s, Y_s = generate_samples(H, PILOT_IDX, N_SUPPORT, snr_db, rng)
    X_q, Y_q = generate_samples(H, PILOT_IDX, N_QUERY,   snr_db, rng)
    return X_s, Y_s, X_q, Y_q


# ───────────────── Main ─────────────────────────
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    input_dim = 2 * N_PILOT

    for split, n_tasks in [("train", N_TRAIN_TASKS), ("test", N_TEST_TASKS)]:
        xs_all, ys_all, xq_all, yq_all = [], [], [], []
        for _ in range(n_tasks):
            xs, ys, xq, yq = generate_task(rng)
            xs_all.append(xs); ys_all.append(ys)
            xq_all.append(xq); yq_all.append(yq)

        x_support = np.array(xs_all)
        y_support = np.array(ys_all)
        x_query = np.array(xq_all)
        y_query = np.array(yq_all)
        path = os.path.join(SAVE_DIR, f"{split}_tasks.npz")
        np.savez(path,
                 X_support=x_support,
                 Y_support=y_support,
                 X_query=x_query,
                 Y_query=y_query)
        assert x_support.shape == (n_tasks, N_SUPPORT, input_dim), (
            f"{split} X_support has shape {x_support.shape}, expected "
            f"({n_tasks}, {N_SUPPORT}, {input_dim})."
        )
        print(f"[OK] {n_tasks} {split} tasks  ->  {path}")
        print(f"     support: X {xs_all[0].shape}  Y {ys_all[0].shape}")
        print(f"     query  : X {xq_all[0].shape}  Y {yq_all[0].shape}")

    print("\nDone.  Data saved in:", SAVE_DIR)


if __name__ == "__main__":
    main()
