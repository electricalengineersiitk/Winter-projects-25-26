"""
test.py  —  Evaluate MAML, Reptile, and a scratch baseline on unseen tasks

Meta-learned models are adapted for a small number of gradient steps.
The baseline is trained from scratch on each new task and reported through 200
optimization steps, which matches the project rubric while still letting us
compare the early-stage learning curves.
"""

import os
import copy
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "projectbyclaude-mpl"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM = 16, 64, 64
ADAPT_LR = 0.01
N_SHOTS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
REPORT_STEP_COUNTS = [1, 3, 5, 10, 20, 50, 200]
BASELINE_STEPS = 200
FEW_SHOT_STEPS = [1, 3, 5, 10]


def set_seed(seed):
    """Set PyTorch/NumPy seeds so evaluation is reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ChannelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        )

    def forward(self, x):
        return self.net(x)


def compute_nmse_db(pred, target):
    mse = ((pred - target) ** 2).sum(-1).mean()
    norm = (target ** 2).sum(-1).mean()
    return (10 * torch.log10(mse / norm + 1e-12)).item()


def adapt_and_eval(init_model, x_s, y_s, x_q, y_q, n_steps):
    """Adapt a cloned model for n_steps, then report query NMSE."""
    model = copy.deepcopy(init_model)
    opt = torch.optim.Adam(model.parameters(), lr=ADAPT_LR)
    for _ in range(n_steps):
        opt.zero_grad()
        F.mse_loss(model(x_s), y_s).backward()
        opt.step()
    with torch.no_grad():
        return compute_nmse_db(model(x_q), y_q)


def train_baseline_from_scratch(x_s, y_s, x_q, y_q, task_seed, report_steps):
    """Train a fresh model and record NMSE checkpoints along the trajectory."""
    set_seed(task_seed)
    model = ChannelNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=ADAPT_LR)

    scores = {}
    max_steps = max(report_steps)
    for step in range(1, max_steps + 1):
        opt.zero_grad()
        F.mse_loss(model(x_s), y_s).backward()
        opt.step()
        if step in report_steps:
            with torch.no_grad():
                scores[step] = compute_nmse_db(model(x_q), y_q)

    return scores


def mean_results(results, method, steps):
    """Return mean NMSE values for a method at the requested steps."""
    return {step: float(np.mean(results[method][step])) for step in steps}


def print_few_shot_table(results):
    """Print a markdown-friendly few-shot comparison table."""
    baseline_means = mean_results(results, "Baseline", FEW_SHOT_STEPS)
    maml_means = mean_results(results, "MAML", FEW_SHOT_STEPS)

    print("### Few-Shot Advantage")
    print("| **Steps** | **Baseline NMSE** | **MAML NMSE** | **MAML gain over Baseline (dB)** |")
    print("|---|---|---|---|")
    for step in FEW_SHOT_STEPS:
        gain = baseline_means[step] - maml_means[step]
        print(
            f"| {step} | {baseline_means[step]:.1f} dB | {maml_means[step]:.1f} dB | "
            f"{gain:.1f} dB |"
        )


def annotate_gap(ax, step, baseline_value, maml_value):
    """Annotate the few-shot gap between MAML and the scratch baseline."""
    gain = baseline_value - maml_value
    ax.annotate(
        f"MAML gain: {gain:.1f} dB",
        xy=(step, maml_value),
        xytext=(step + 1.5, baseline_value + 1.5),
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.0),
        color="#1f77b4",
        fontsize=8,
        ha="left",
    )


def main():
    set_seed(SEED)
    print("Loading test data...")
    d = np.load(os.path.join(DATA_DIR, "test_tasks.npz"))
    Xs = torch.tensor(d["X_support"], dtype=torch.float32)
    Ys = torch.tensor(d["Y_support"], dtype=torch.float32)
    Xq = torch.tensor(d["X_query"], dtype=torch.float32)
    Yq = torch.tensor(d["Y_query"], dtype=torch.float32)
    n_tasks = Xs.shape[0]
    print(f"  {n_tasks} test tasks.  {N_SHOTS}-shot.  Adam lr = {ADAPT_LR}\n")

    maml_model = ChannelNet().to(DEVICE)
    maml_model.load_state_dict(
        torch.load(
            os.path.join(CKPT_DIR, "maml_model.pt"),
            map_location=DEVICE,
            weights_only=True,
        )
    )

    reptile_model = ChannelNet().to(DEVICE)
    reptile_model.load_state_dict(
        torch.load(
            os.path.join(CKPT_DIR, "reptile_model.pt"),
            map_location=DEVICE,
            weights_only=True,
        )
    )

    model_map = [
        ("Baseline", None),
        ("MAML", maml_model),
        ("Reptile", reptile_model),
    ]
    results = {name: {s: [] for s in REPORT_STEP_COUNTS} for name, _ in model_map}

    print(f"--- Baseline trained from scratch for {BASELINE_STEPS} Adam steps ---")
    for t in range(n_tasks):
        x_s = Xs[t, :N_SHOTS].to(DEVICE)
        y_s = Ys[t, :N_SHOTS].to(DEVICE)
        x_q = Xq[t].to(DEVICE)
        y_q = Yq[t].to(DEVICE)
        task_scores = train_baseline_from_scratch(
            x_s,
            y_s,
            x_q,
            y_q,
            task_seed=SEED + t,
            report_steps=REPORT_STEP_COUNTS,
        )
        for step, score in task_scores.items():
            results["Baseline"][step].append(score)
    for step in REPORT_STEP_COUNTS:
        print(f"  step {step:>3d}:  NMSE = {np.mean(results['Baseline'][step]):.2f} dB")
    print()

    for n_steps in REPORT_STEP_COUNTS:
        print(f"--- Meta-learned models after {n_steps} adaptation steps ---")
        for t in range(n_tasks):
            x_s = Xs[t, :N_SHOTS].to(DEVICE)
            y_s = Ys[t, :N_SHOTS].to(DEVICE)
            x_q = Xq[t].to(DEVICE)
            y_q = Yq[t].to(DEVICE)
            for name, model in model_map[1:]:
                results[name][n_steps].append(
                    adapt_and_eval(model, x_s, y_s, x_q, y_q, n_steps)
                )
        for name, _ in model_map[1:]:
            print(f"  {name:>10s}:  NMSE = {np.mean(results[name][n_steps]):.2f} dB")
        print()

    print("=" * 90)
    print(f"{'Method':>10s}", end="")
    for step in REPORT_STEP_COUNTS:
        print(f" | {step:>4d}-step", end="")
    print()
    print("-" * 90)
    for name, _ in model_map:
        print(f"{name:>10s}", end="")
        for step in REPORT_STEP_COUNTS:
            print(f" | {np.mean(results[name][step]):>8.2f}", end="")
        print("  dB")
    print("=" * 90)
    print("  NMSE in dB — lower is better.\n")

    print("README results:")
    print("| **Method** | **5-step NMSE** | **10-step NMSE** | **20-step NMSE** | **200-step NMSE** |")
    print("|---|---|---|---|---|")
    for name, _ in model_map:
        values = [np.mean(results[name][step]) for step in [5, 10, 20, 200]]
        print(
            f"| {name} | {values[0]:.1f} dB | {values[1]:.1f} dB | "
            f"{values[2]:.1f} dB | {values[3]:.1f} dB |"
        )
    print()
    print_few_shot_table(results)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    plot_styles = list(zip(
        model_map,
        ["s", "o", "^"],
        ["#d62728", "#1f77b4", "#2ca02c"],
    ))
    left_steps = [step for step in REPORT_STEP_COUNTS if step <= 20]

    for ax, steps, title in [
        (axes[0], left_steps, "Few-Shot Regime (1–20 steps)"),
        (axes[1], REPORT_STEP_COUNTS, "Full Adaptation Curve"),
    ]:
        for (name, _), marker, color in plot_styles:
            means = [np.mean(results[name][step]) for step in steps]
            ax.plot(
                steps,
                means,
                marker=marker,
                label=name,
                linewidth=2,
                markersize=7,
                color=color,
            )
        ax.set_xlabel("Number of Adaptation / Training Steps (Adam)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[0].set_xlim(1, 20)
    axes[1].set_xlim(1, 200)

    baseline_means = mean_results(results, "Baseline", [5, 10])
    maml_means = mean_results(results, "MAML", [5, 10])
    annotate_gap(axes[0], 5, baseline_means[5], maml_means[5])
    annotate_gap(axes[0], 10, baseline_means[10], maml_means[10])

    fig.supylabel("NMSE (dB) [lower is better]")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(RESULTS_DIR, "plot_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Comparison plot -> {path}")


if __name__ == "__main__":
    main()
