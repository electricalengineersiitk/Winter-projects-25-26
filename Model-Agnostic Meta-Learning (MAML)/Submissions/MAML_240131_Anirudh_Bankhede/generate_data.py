from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "channel_estimation_tasks.npz"
CONFIG_PATH = DATA_DIR / "dataset_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic few-shot channel estimation tasks."
    )
    parser.add_argument("--train-tasks", type=int, default=100)
    parser.add_argument("--test-tasks", type=int, default=20)
    parser.add_argument(
        "--support-size",
        type=int,
        default=20,
        help="Support pool per task. Evaluation uses 5/10/20-shot subsets from this pool.",
    )
    parser.add_argument("--query-size", type=int, default=100)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--channel-dim", type=int, default=12)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_family(
    rng: np.random.Generator,
    input_dim: int,
    channel_dim: int,
    latent_dim: int,
) -> dict[str, np.ndarray]:
    scale = 1.0 / np.sqrt(channel_dim)
    return {
        "base_linear": rng.normal(
            0.0, 0.55 * scale, size=(input_dim, channel_dim)
        ).astype(np.float32),
        "linear_basis": rng.normal(
            0.0, 0.16 * scale, size=(latent_dim, input_dim, channel_dim)
        ).astype(np.float32),
        "nonlinear_basis": rng.normal(
            0.0, 0.20 * scale, size=(input_dim, channel_dim)
        ).astype(np.float32),
        "obs_bias_basis": rng.normal(
            0.0, 0.08, size=(latent_dim, input_dim)
        ).astype(np.float32),
        "channel_bias_basis": rng.normal(
            0.0, 0.10, size=(latent_dim, channel_dim)
        ).astype(np.float32),
    }


def sample_task(
    rng: np.random.Generator,
    family: dict[str, np.ndarray],
    support_size: int,
    query_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    input_dim, channel_dim = family["base_linear"].shape
    total_samples = support_size + query_size

    latent = rng.uniform(-1.0, 1.0, size=family["linear_basis"].shape[0]).astype(
        np.float32
    )
    snr_db = float(rng.uniform(6.0, 18.0))
    num_paths = int(rng.integers(2, 7))
    env_gain = float(rng.uniform(0.85, 1.25))
    phase_shift = float(rng.uniform(-np.pi, np.pi))

    linear_map = family["base_linear"] + np.tensordot(
        latent, family["linear_basis"], axes=(0, 0)
    )
    channel_bias = 0.18 * (latent @ family["channel_bias_basis"])
    obs_bias = 0.12 * (latent @ family["obs_bias_basis"])

    path_profile = np.exp(-np.linspace(0.0, 2.2, channel_dim, dtype=np.float32))
    active_dims = min(channel_dim, num_paths * 2)
    active_idx = rng.choice(channel_dim, size=active_dims, replace=False)
    support_mask = np.full(channel_dim, 0.18, dtype=np.float32)
    support_mask[active_idx] = 1.0
    channel_scale = env_gain * (0.10 + path_profile * support_mask)

    channels = (
        rng.normal(size=(total_samples, channel_dim)).astype(np.float32) * channel_scale
        + channel_bias
    )

    clean_obs = channels @ linear_map.T
    clean_obs += 0.08 * np.sin(channels @ family["nonlinear_basis"].T + phase_shift)
    clean_obs += obs_bias

    signal_power = float(np.mean(clean_obs**2))
    noise_std = np.sqrt(signal_power / (10.0 ** (snr_db / 10.0)))
    noisy_obs = clean_obs + noise_std * rng.normal(size=clean_obs.shape).astype(np.float32)

    x_support = noisy_obs[:support_size].astype(np.float32)
    y_support = channels[:support_size].astype(np.float32)
    x_query = noisy_obs[support_size:].astype(np.float32)
    y_query = channels[support_size:].astype(np.float32)
    return x_support, y_support, x_query, y_query, snr_db, num_paths


def stack_tasks(
    num_tasks: int,
    rng: np.random.Generator,
    family: dict[str, np.ndarray],
    support_size: int,
    query_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_support_list = []
    y_support_list = []
    x_query_list = []
    y_query_list = []
    snr_list = []
    num_paths_list = []

    for _ in range(num_tasks):
        task = sample_task(rng, family, support_size, query_size)
        x_support, y_support, x_query, y_query, snr_db, num_paths = task
        x_support_list.append(x_support)
        y_support_list.append(y_support)
        x_query_list.append(x_query)
        y_query_list.append(y_query)
        snr_list.append(snr_db)
        num_paths_list.append(num_paths)

    return (
        np.stack(x_support_list),
        np.stack(y_support_list),
        np.stack(x_query_list),
        np.stack(y_query_list),
        np.asarray(snr_list, dtype=np.float32),
        np.asarray(num_paths_list, dtype=np.int32),
    )


def compute_stats(
    train_x_support: np.ndarray,
    train_y_support: np.ndarray,
    train_x_query: np.ndarray,
    train_y_query: np.ndarray,
) -> dict[str, np.ndarray]:
    x_all = np.concatenate(
        [
            train_x_support.reshape(-1, train_x_support.shape[-1]),
            train_x_query.reshape(-1, train_x_query.shape[-1]),
        ],
        axis=0,
    )
    y_all = np.concatenate(
        [
            train_y_support.reshape(-1, train_y_support.shape[-1]),
            train_y_query.reshape(-1, train_y_query.shape[-1]),
        ],
        axis=0,
    )
    return {
        "x_mean": x_all.mean(axis=0).astype(np.float32),
        "x_std": (x_all.std(axis=0) + 1e-6).astype(np.float32),
        "y_mean": y_all.mean(axis=0).astype(np.float32),
        "y_std": (y_all.std(axis=0) + 1e-6).astype(np.float32),
    }


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    family = build_family(rng, args.input_dim, args.channel_dim, args.latent_dim)

    (
        train_x_support,
        train_y_support,
        train_x_query,
        train_y_query,
        train_snr_db,
        train_num_paths,
    ) = stack_tasks(args.train_tasks, rng, family, args.support_size, args.query_size)

    (
        test_x_support,
        test_y_support,
        test_x_query,
        test_y_query,
        test_snr_db,
        test_num_paths,
    ) = stack_tasks(args.test_tasks, rng, family, args.support_size, args.query_size)

    stats = compute_stats(train_x_support, train_y_support, train_x_query, train_y_query)

    np.savez_compressed(
        DATA_PATH,
        train_x_support=train_x_support,
        train_y_support=train_y_support,
        train_x_query=train_x_query,
        train_y_query=train_y_query,
        test_x_support=test_x_support,
        test_y_support=test_y_support,
        test_x_query=test_x_query,
        test_y_query=test_y_query,
        train_snr_db=train_snr_db,
        test_snr_db=test_snr_db,
        train_num_paths=train_num_paths,
        test_num_paths=test_num_paths,
        x_mean=stats["x_mean"],
        x_std=stats["x_std"],
        y_mean=stats["y_mean"],
        y_std=stats["y_std"],
    )

    config = {
        "seed": args.seed,
        "train_tasks": args.train_tasks,
        "test_tasks": args.test_tasks,
        "support_size": args.support_size,
        "query_size": args.query_size,
        "input_dim": args.input_dim,
        "channel_dim": args.channel_dim,
        "latent_dim": args.latent_dim,
        "dataset_path": str(DATA_PATH.name),
        "description": (
            "Synthetic channel estimation tasks with task-specific pilot/channel mappings, "
            "variable SNR, path counts, and observation noise."
        ),
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Saved dataset to {DATA_PATH}")
    print(
        "Train tasks:",
        train_x_support.shape[0],
        "| Test tasks:",
        test_x_support.shape[0],
        "| Support pool:",
        train_x_support.shape[1],
        "| Query size:",
        train_x_query.shape[1],
    )
    print(
        f"SNR range: {train_snr_db.min():.1f} dB to {train_snr_db.max():.1f} dB "
        f"(train), {test_snr_db.min():.1f} dB to {test_snr_db.max():.1f} dB (test)"
    )


if __name__ == "__main__":
    main()
