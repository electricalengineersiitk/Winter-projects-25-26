"""
test.py — Evaluate the meta-trained model on unseen test tasks.
Compares MAML vs Reptile vs Baseline (from scratch).
Generates results/plot_comparison.png  (two-panel layout for bonus marks).
"""

import os, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Match hyper-params from train.py ─────────────────────────────────────────
INPUT_DIM    = 32
OUTPUT_DIM   = 32
INNER_LR     = 0.01
RESULTS_DIR  = "results"

# STEP 3 CHANGE 1: two shot lists — one per panel
FEW_SHOT_LIST  = [1, 3, 5, 10, 20]                  # left panel  (few-shot)
FULL_SHOT_LIST = [1, 3, 5, 10, 20, 50, 100, 200]    # right panel (full curve)


# ── Model (must match train.py) ───────────────────────────────────────────────
class ChannelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 64), nn.ReLU(),
            nn.Linear(64, 64),        nn.ReLU(),
            nn.Linear(64, OUTPUT_DIM),
        )

    def forward(self, x):
        return self.net(x)


# ── Utilities ─────────────────────────────────────────────────────────────────
def load_dataset(path: str):
    data = np.load(path)
    return {k: torch.tensor(data[k]) for k in data.files}


def nmse_db(pred: torch.Tensor, target: torch.Tensor) -> float:
    num = ((pred - target) ** 2).sum(dim=-1).mean()
    den = (target ** 2).sum(dim=-1).mean() + 1e-10
    return 10 * torch.log10(num / den).item()


def adapt_and_eval(model, xs, ys, xq, yq, steps, lr):
    adapted   = copy.deepcopy(model)
    opt       = optim.SGD(adapted.parameters(), lr=lr)
    criterion = nn.MSELoss()
    adapted.train()
    for _ in range(steps):
        opt.zero_grad()
        criterion(adapted(xs), ys).backward()
        opt.step()
    adapted.eval()
    with torch.no_grad():
        return nmse_db(adapted(xq), yq)


def baseline_eval(xs, ys, xq, yq, steps):
    model     = ChannelNet()
    opt       = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        criterion(model(xs), ys).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return nmse_db(model(xq), yq)


def eval_over_shots(model, test_data, shot_list, is_baseline=False):
    """Evaluate a model across all shot counts. Returns list of mean NMSE per shot."""
    n_tasks = test_data["X_support"].shape[0]
    means   = []
    for n_steps in shot_list:
        row = []
        for i in range(n_tasks):
            xs = test_data["X_support"][i]
            ys = test_data["Y_support"][i]
            xq = test_data["X_query"][i]
            yq = test_data["Y_query"][i]
            if is_baseline:
                row.append(baseline_eval(xs, ys, xq, yq, steps=max(n_steps * 40, 50)))
            else:
                row.append(adapt_and_eval(model, xs, ys, xq, yq,
                                          steps=n_steps, lr=INNER_LR))
        means.append(np.mean(row))
        print(f"    steps={n_steps:>3}  mean NMSE = {means[-1]:.2f} dB")
    return means


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading test data …")
    test_data = load_dataset(os.path.join("data", "test_tasks.npz"))

    # ── STEP 3 CHANGE 2: load BOTH saved models ───────────────────────────────
    maml_model    = ChannelNet()
    reptile_model = ChannelNet()

    maml_model.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, "meta_model.pt"), map_location="cpu"))
    reptile_model.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, "reptile_model.pt"), map_location="cpu"))

    maml_model.eval()
    reptile_model.eval()
    print("Loaded meta_model.pt    (MAML)")
    print("Loaded reptile_model.pt (Reptile)\n")

    # ── STEP 3 CHANGE 3: evaluate all three over the full shot list ───────────
    print("Evaluating MAML …")
    maml_full    = eval_over_shots(maml_model,    test_data, FULL_SHOT_LIST)

    print("\nEvaluating Reptile …")
    reptile_full = eval_over_shots(reptile_model, test_data, FULL_SHOT_LIST)

    print("\nEvaluating Baseline …")
    base_full    = eval_over_shots(None,          test_data, FULL_SHOT_LIST,
                                    is_baseline=True)

    # Slice just the few-shot portion for the left panel
    n_few       = len(FEW_SHOT_LIST)
    maml_few    = maml_full[:n_few]
    reptile_few = reptile_full[:n_few]
    base_few    = base_full[:n_few]

    # ── Print results table ───────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"{'Steps':>6}  {'MAML':>12}  {'Reptile':>12}  {'Baseline':>12}")
    print("-" * 62)
    for k, s in enumerate(FULL_SHOT_LIST):
        print(f"{s:>6}  {maml_full[k]:>12.2f}  {reptile_full[k]:>12.2f}  {base_full[k]:>12.2f}")
    print("=" * 62)

    # ── STEP 4: Two-panel plot ────────────────────────────────────────────────
    #
    # LEFT  panel → x-axis 1-20  steps   (zoomed in, shows MAML gap clearly)
    # RIGHT panel → x-axis 1-200 steps   (shows all methods converge)
    #
    C_MAML    = "#2196F3"   # blue
    C_REPTILE = "#4CAF50"   # green
    C_BASE    = "#F44336"   # red

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # ── LEFT panel ────────────────────────────────────────────────────────────
    ax1.plot(FEW_SHOT_LIST, maml_few,
             "o-",  color=C_MAML,    linewidth=2.5, markersize=8, label="MAML")
    ax1.plot(FEW_SHOT_LIST, reptile_few,
             "^-",  color=C_REPTILE, linewidth=2.5, markersize=8, label="Reptile")
    ax1.plot(FEW_SHOT_LIST, base_few,
             "s--", color=C_BASE,    linewidth=2.5, markersize=8, label="Baseline")

    # Annotate MAML gain at 5 steps and 10 steps
    for step in [5, 10]:
        if step in FEW_SHOT_LIST:
            idx  = FEW_SHOT_LIST.index(step)
            gain = base_few[idx] - maml_few[idx]
            ax1.annotate(
                f"MAML gain: {gain:.1f} dB",
                xy=(step, maml_few[idx]),
                xytext=(step + 1, maml_few[idx] + 2.5),
                arrowprops=dict(arrowstyle="->", color=C_MAML),
                color=C_MAML, fontsize=9
            )

    ax1.set_title("Few-Shot Regime (1–20 steps)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Adaptation / Training Steps (Adam)")
    ax1.set_ylabel("NMSE (dB) [lower is better]")
    ax1.legend(fontsize=10, loc="lower left")
    ax1.grid(alpha=0.3)

    # ── RIGHT panel ───────────────────────────────────────────────────────────
    ax2.plot(FULL_SHOT_LIST, maml_full,
             "o-",  color=C_MAML,    linewidth=2.5, markersize=7, label="MAML")
    ax2.plot(FULL_SHOT_LIST, reptile_full,
             "^-",  color=C_REPTILE, linewidth=2.5, markersize=7, label="Reptile")
    ax2.plot(FULL_SHOT_LIST, base_full,
             "s--", color=C_BASE,    linewidth=2.5, markersize=7, label="Baseline")

    ax2.set_title("Full Adaptation Curve", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Adaptation / Training Steps (Adam)")
    ax2.legend(fontsize=10, loc="lower left")
    ax2.grid(alpha=0.3)

    fig.suptitle("Channel Estimation: MAML vs Reptile vs Scratch Baseline",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = os.path.join(RESULTS_DIR, "plot_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")


if __name__ == "__main__":
    main()