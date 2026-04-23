from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "channel_estimation_tasks.npz"
REPTILE_MODEL_PATH = BASE_DIR / "results" / "reptile_model.npz"
MAML_MODEL_PATH = BASE_DIR / "results" / "maml_model.npz"
RESULTS_DIR = BASE_DIR / "results"
COMPARISON_PATH = RESULTS_DIR / "comparison_metrics.json"
COMPARISON_PLOT_PATH = RESULTS_DIR / "plot_comparison.png"

PARAM_KEYS = ("W1", "b1", "W2", "b2", "W3", "b3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MAML, Reptile, and a scratch baseline.")
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--adapt-steps", type=int, default=5)
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--baseline-steps", type=int, default=200)
    parser.add_argument("--baseline-lr", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def load_dataset() -> dict[str, np.ndarray]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset at {DATA_PATH}. Run `python generate_data.py` first."
        )
    with np.load(DATA_PATH) as data:
        output: dict[str, np.ndarray] = {}
        for key in data.files:
            value = data[key]
            output[key] = value.astype(np.float32) if value.dtype.kind == "f" else value
        return output


def load_model(model_path: Path) -> tuple[dict[str, np.ndarray], int]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing trained model at {model_path}. Run `python train.py` first."
        )
    with np.load(model_path) as checkpoint:
        params = {key: checkpoint[key].astype(np.float32) for key in PARAM_KEYS}
        hidden_dim = int(checkpoint["hidden_dim"][0])
    return params, hidden_dim


def normalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((data - mean) / std).astype(np.float32)


def denormalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (data * std + mean).astype(np.float32)


def init_params(
    rng: np.random.Generator, input_dim: int, hidden_dim: int, output_dim: int
) -> dict[str, np.ndarray]:
    return {
        "W1": rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim)).astype(np.float32),
        "b1": np.zeros(hidden_dim, dtype=np.float32),
        "W2": rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, hidden_dim)).astype(np.float32),
        "b2": np.zeros(hidden_dim, dtype=np.float32),
        "W3": rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, output_dim)).astype(np.float32),
        "b3": np.zeros(output_dim, dtype=np.float32),
    }


def clone_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in params.items()}


def forward(
    params: dict[str, np.ndarray], x: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    z1 = x @ params["W1"] + params["b1"]
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = np.maximum(z2, 0.0)
    out = a2 @ params["W3"] + params["b3"]
    cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
    return out, cache


def mse_loss(pred: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    diff = pred - target
    loss = float(np.mean(diff**2))
    grad = (2.0 / diff.size) * diff
    return loss, grad.astype(np.float32)


def backward(
    params: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    grad_out: np.ndarray,
) -> dict[str, np.ndarray]:
    grads: dict[str, np.ndarray] = {}
    grads["W3"] = cache["a2"].T @ grad_out
    grads["b3"] = grad_out.sum(axis=0)

    grad_a2 = grad_out @ params["W3"].T
    grad_z2 = grad_a2 * (cache["z2"] > 0.0)
    grads["W2"] = cache["a1"].T @ grad_z2
    grads["b2"] = grad_z2.sum(axis=0)

    grad_a1 = grad_z2 @ params["W2"].T
    grad_z1 = grad_a1 * (cache["z1"] > 0.0)
    grads["W1"] = cache["x"].T @ grad_z1
    grads["b1"] = grad_z1.sum(axis=0)
    return {key: value.astype(np.float32) for key, value in grads.items()}


class Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float) -> None:
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0
        self.m = {key: np.zeros_like(value) for key, value in params.items()}
        self.v = {key: np.zeros_like(value) for key, value in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for key in params:
            grad = grads[key]
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (grad * grad)

            m_hat = self.m[key] / (1.0 - self.beta1**self.t)
            v_hat = self.v[key] / (1.0 - self.beta2**self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def adapt_params(
    init_params_dict: dict[str, np.ndarray],
    x_support: np.ndarray,
    y_support: np.ndarray,
    steps: int,
    lr: float,
) -> dict[str, np.ndarray]:
    params = clone_params(init_params_dict)
    optimizer = Adam(params, lr)
    for _ in range(steps):
        pred, cache = forward(params, x_support)
        _, grad_out = mse_loss(pred, y_support)
        grads = backward(params, cache, grad_out)
        optimizer.step(params, grads)
    return params


def nmse_linear(pred: np.ndarray, target: np.ndarray) -> float:
    numerator = float(np.mean(np.sum((pred - target) ** 2, axis=1)))
    denominator = float(np.mean(np.sum(target**2, axis=1))) + 1e-8
    return numerator / denominator


def evaluate_method(
    init_params_dict: dict[str, np.ndarray],
    x_support_pool: np.ndarray,
    y_support_pool: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
    shots: int,
    steps: int,
    lr: float,
    rng: np.random.Generator,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> float:
    support_pool = x_support_pool.shape[0]
    support_idx = rng.choice(support_pool, size=shots, replace=False)
    x_support = normalize(x_support_pool[support_idx], x_mean, x_std)
    y_support = normalize(y_support_pool[support_idx], y_mean, y_std)
    x_query_norm = normalize(x_query, x_mean, x_std)

    adapted = adapt_params(init_params_dict, x_support, y_support, steps, lr)
    pred_norm, _ = forward(adapted, x_query_norm)
    pred = denormalize(pred_norm, y_mean, y_std)
    return nmse_linear(pred, y_query)


def plot_comparison(
    shots: list[int],
    reptile_db: list[float],
    maml_db: list[float],
    baseline_db: list[float],
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(shots, reptile_db, marker="o", linewidth=2.0, color="#005f73", label="Reptile")
    plt.plot(shots, maml_db, marker="^", linewidth=2.0, color="#6a4c93", label="MAML")
    plt.plot(shots, baseline_db, marker="s", linewidth=2.0, color="#bb3e03", label="Baseline")
    plt.title("Channel Estimation: MAML vs Reptile vs Baseline")
    plt.xlabel("Support Shots")
    plt.ylabel("Average Query NMSE (dB)")
    plt.xticks(shots)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(COMPARISON_PLOT_PATH, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    reptile_params, reptile_hidden_dim = load_model(REPTILE_MODEL_PATH)
    maml_params, maml_hidden_dim = load_model(MAML_MODEL_PATH)
    if reptile_hidden_dim != maml_hidden_dim:
        raise ValueError(
            f"Model hidden dimensions do not match: Reptile={reptile_hidden_dim}, MAML={maml_hidden_dim}."
        )
    if reptile_hidden_dim != args.hidden_dim:
        print(
            f"Warning: checkpoint hidden_dim={reptile_hidden_dim}, overriding requested hidden_dim={args.hidden_dim}."
        )
        args.hidden_dim = reptile_hidden_dim

    input_dim = dataset["test_x_support"].shape[-1]
    output_dim = dataset["test_y_support"].shape[-1]
    support_pool = dataset["test_x_support"].shape[1]
    shots_grid = [shot for shot in (5, 10, 20) if shot <= support_pool]

    reptile_metrics: dict[str, float] = {}
    maml_metrics: dict[str, float] = {}
    baseline_metrics: dict[str, float] = {}

    for shots in shots_grid:
        reptile_errors = []
        maml_errors = []
        baseline_errors = []

        for task_idx in range(dataset["test_x_support"].shape[0]):
            support_seed = args.seed + 101 * task_idx + shots
            baseline_init_rng = np.random.default_rng(args.seed + 1001 * task_idx + shots)

            reptile_nmse = evaluate_method(
                reptile_params,
                dataset["test_x_support"][task_idx],
                dataset["test_y_support"][task_idx],
                dataset["test_x_query"][task_idx],
                dataset["test_y_query"][task_idx],
                shots=shots,
                steps=args.adapt_steps,
                lr=args.inner_lr,
                rng=np.random.default_rng(support_seed),
                x_mean=dataset["x_mean"],
                x_std=dataset["x_std"],
                y_mean=dataset["y_mean"],
                y_std=dataset["y_std"],
            )
            reptile_errors.append(reptile_nmse)

            maml_nmse = evaluate_method(
                maml_params,
                dataset["test_x_support"][task_idx],
                dataset["test_y_support"][task_idx],
                dataset["test_x_query"][task_idx],
                dataset["test_y_query"][task_idx],
                shots=shots,
                steps=args.adapt_steps,
                lr=args.inner_lr,
                rng=np.random.default_rng(support_seed),
                x_mean=dataset["x_mean"],
                x_std=dataset["x_std"],
                y_mean=dataset["y_mean"],
                y_std=dataset["y_std"],
            )
            maml_errors.append(maml_nmse)

            baseline_params = init_params(
                baseline_init_rng, input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=output_dim
            )
            baseline_nmse = evaluate_method(
                baseline_params,
                dataset["test_x_support"][task_idx],
                dataset["test_y_support"][task_idx],
                dataset["test_x_query"][task_idx],
                dataset["test_y_query"][task_idx],
                shots=shots,
                steps=args.baseline_steps,
                lr=args.baseline_lr,
                rng=np.random.default_rng(support_seed),
                x_mean=dataset["x_mean"],
                x_std=dataset["x_std"],
                y_mean=dataset["y_mean"],
                y_std=dataset["y_std"],
            )
            baseline_errors.append(baseline_nmse)

        reptile_avg = float(np.mean(reptile_errors))
        maml_avg = float(np.mean(maml_errors))
        baseline_avg = float(np.mean(baseline_errors))
        reptile_metrics[str(shots)] = float(10.0 * np.log10(reptile_avg + 1e-12))
        maml_metrics[str(shots)] = float(10.0 * np.log10(maml_avg + 1e-12))
        baseline_metrics[str(shots)] = float(10.0 * np.log10(baseline_avg + 1e-12))

    plot_comparison(
        shots_grid,
        [reptile_metrics[str(shot)] for shot in shots_grid],
        [maml_metrics[str(shot)] for shot in shots_grid],
        [baseline_metrics[str(shot)] for shot in shots_grid],
    )

    metrics_payload = {
        "shots": shots_grid,
        "reptile_nmse_db": reptile_metrics,
        "maml_nmse_db": maml_metrics,
        "baseline_nmse_db": baseline_metrics,
        "adapt_steps": args.adapt_steps,
        "baseline_steps": args.baseline_steps,
        "inner_lr": args.inner_lr,
        "baseline_lr": args.baseline_lr,
        "default_report_shots": args.shots,
    }
    COMPARISON_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    report_shots = str(args.shots)
    if report_shots not in reptile_metrics:
        raise ValueError(f"Requested --shots={args.shots} is not available for support pool {support_pool}.")

    print("Average query NMSE over 20 unseen test tasks")
    print(f"Reptile ({args.shots}-shot, {args.adapt_steps} adaptation steps): {reptile_metrics[report_shots]:.2f} dB")
    print(f"MAML ({args.shots}-shot, {args.adapt_steps} adaptation steps): {maml_metrics[report_shots]:.2f} dB")
    print(f"Baseline ({args.shots}-shot, {args.baseline_steps} training steps): {baseline_metrics[report_shots]:.2f} dB")
    print("")
    print("Support-shot comparison")
    for shots in shots_grid:
        print(
            f"{shots:>2}-shot | Reptile: {reptile_metrics[str(shots)]:>7.2f} dB | "
            f"MAML: {maml_metrics[str(shots)]:>7.2f} dB | "
            f"Baseline: {baseline_metrics[str(shots)]:>7.2f} dB"
        )

    print(f"\nSaved comparison plot to {COMPARISON_PLOT_PATH}")


if __name__ == "__main__":
    main()
