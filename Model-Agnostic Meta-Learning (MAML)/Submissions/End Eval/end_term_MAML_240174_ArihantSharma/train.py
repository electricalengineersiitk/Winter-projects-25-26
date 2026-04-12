"""
train.py — Meta-train a channel estimator using MAML (+ optional Reptile).
Saves model weights and a training-loss plot to results/.
"""

import os, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
INPUT_DIM       = 32        # PILOT_DIM * 2  (real + imag)
OUTPUT_DIM      = 32        # CHANNEL_DIM * 2

META_LR         = 1e-3      # outer (meta) learning rate
INNER_LR        = 0.01      # inner (adaptation) learning rate
INNER_STEPS     = 5         # gradient steps during adaptation
META_ITERATIONS = 500       # total meta-training iterations
TASK_BATCH      = 8         # tasks per meta-update

USE_REPTILE     = False   # set True to use Reptile instead of MAML
REPTILE_EPSILON = 0.1       # Reptile step size

RESULTS_DIR     = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Neural network ────────────────────────────────────────────────────────────
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


# ── Utility ───────────────────────────────────────────────────────────────────
def load_dataset(path: str):
    data = np.load(path)
    return {k: torch.tensor(data[k]) for k in data.files}


def task_iter(dataset: dict, batch_size: int):
    """Yield random mini-batches of tasks."""
    n = dataset["X_support"].shape[0]
    idx = torch.randperm(n)[:batch_size]
    return {k: v[idx] for k, v in dataset.items()}


def inner_adapt(model: nn.Module, xs: torch.Tensor, ys: torch.Tensor,
                steps: int, lr: float) -> nn.Module:
    """Gradient-descent adaptation on the support set. Returns adapted clone."""
    adapted = copy.deepcopy(model)
    opt = optim.SGD(adapted.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in range(steps):
        opt.zero_grad()
        criterion(adapted(xs), ys).backward()
        opt.step()
    return adapted


def nmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    num = ((pred - target) ** 2).sum(dim=-1).mean()
    den = (target ** 2).sum(dim=-1).mean()
    return 10 * torch.log10(num / den).item()


# ── MAML outer loop ───────────────────────────────────────────────────────────
def maml_step(model: nn.Module, meta_opt: optim.Optimizer,
              batch: dict) -> float:
    criterion = nn.MSELoss()
    meta_opt.zero_grad()
    outer_loss = torch.tensor(0.0)

    n_tasks = batch["X_support"].shape[0]
    for i in range(n_tasks):
        xs = batch["X_support"][i]
        ys = batch["Y_support"][i]
        xq = batch["X_query"][i]
        yq = batch["Y_query"][i]

        adapted = inner_adapt(model, xs, ys, INNER_STEPS, INNER_LR)

        # Query loss on adapted model — accumulate grads w.r.t. original θ
        # (MAML first-order approximation for simplicity & stability)
        q_pred = adapted(xq)
        outer_loss = outer_loss + criterion(q_pred, yq)

    outer_loss = outer_loss / n_tasks
    outer_loss.backward()
    meta_opt.step()
    return outer_loss.item()


# ── Reptile outer loop ────────────────────────────────────────────────────────
def reptile_step(model: nn.Module, batch: dict) -> float:
    criterion = nn.MSELoss()
    total_loss = 0.0
    n_tasks = batch["X_support"].shape[0]

    theta = {n: p.data.clone() for n, p in model.named_parameters()}

    for i in range(n_tasks):
        xs = batch["X_support"][i]
        ys = batch["Y_support"][i]
        xq = batch["X_query"][i]
        yq = batch["Y_query"][i]

        adapted = inner_adapt(model, xs, ys, INNER_STEPS, INNER_LR)
        with torch.no_grad():
            total_loss += criterion(adapted(xq), yq).item()

        # Nudge θ toward θ'
        with torch.no_grad():
            for name, param in model.named_parameters():
                theta_prime = dict(adapted.named_parameters())[name].data
                param.data += (REPTILE_EPSILON / n_tasks) * (theta_prime - param.data)

    return total_loss / n_tasks


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    print("Loading data …")
    train_data = load_dataset(os.path.join("data", "train_tasks.npz"))

    model    = ChannelNet()
    meta_opt = optim.Adam(model.parameters(), lr=META_LR)

    algo = "Reptile" if USE_REPTILE else "MAML"
    print(f"Meta-training with {algo} for {META_ITERATIONS} iterations …\n")

    losses = []
    for it in range(1, META_ITERATIONS + 1):
        batch = task_iter(train_data, TASK_BATCH)

        if USE_REPTILE:
            loss = reptile_step(model, batch)
        else:
            loss = maml_step(model, meta_opt, batch)

        losses.append(loss)
        if it % 10 == 0:
            print(f"  Iter {it:4d}/{META_ITERATIONS}   loss = {loss:.4f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "meta_model.pt"))
    print(f"\nModel saved → {RESULTS_DIR}/meta_model.pt")

    # ── Plot 1: Training loss curve ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color="#2196F3", linewidth=1.5, label=f"{algo} loss")

    # Smoothed trend
    window = 20
    smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
    ax.plot(range(window - 1, len(losses)), smoothed,
            color="#F44336", linewidth=2, label="Smoothed")

    ax.set_title(f"{algo} Meta-Training Loss", fontsize=14, fontweight="bold")
    ax.set_xlabel("Meta-Training Iteration")
    ax.set_ylabel("Outer Loss (MSE)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "plot_loss.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Loss plot saved → {plot_path}")


if __name__ == "__main__":
    train()
