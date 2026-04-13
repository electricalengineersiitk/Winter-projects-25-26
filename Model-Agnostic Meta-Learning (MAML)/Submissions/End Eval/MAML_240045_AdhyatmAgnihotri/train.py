"""
train.py  —  Meta-train MAML and Reptile for Channel Estimation

Implements:
  1. FOMAML  (First-Order MAML) with Adam inner loop
  2. Reptile  with Adam inner loop

Both use Adam for inner and outer optimization as recommended.
Network: 3 layers, 64 neurons each.
Saves trained models to checkpoints/ and training loss curve to results/.
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

# ───────────────── Configuration ─────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
CKPT_DIR     = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")

INPUT_DIM    = 16       # 2 * N_pilot
OUTPUT_DIM   = 64       # 2 * N_sub
HIDDEN_DIM   = 64       # 64 neurons per layer (as per rubric)

# Meta-learning hyper-parameters
META_LR       = 1e-3    # outer learning rate (Adam)
INNER_LR      = 0.01    # inner learning rate (Adam)
INNER_STEPS   = 5       # gradient steps during adaptation
N_SHOTS       = 10      # support samples used per task
TASK_BATCH    = 5       # tasks per meta-update
META_ITERS    = 800     # total meta-training iterations

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42


def set_seed(seed):
    """Set NumPy and PyTorch seeds for reproducible training runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ───────────────── Model ─────────────────────────
class ChannelNet(nn.Module):
    """3-layer MLP, 64 neurons each, for channel estimation."""
    def __init__(self, in_dim=INPUT_DIM, hid=HIDDEN_DIM, out_dim=OUTPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ───────────────── Utilities ─────────────────────
def load_tasks(split="train"):
    d = np.load(os.path.join(DATA_DIR, f"{split}_tasks.npz"))
    return (
        torch.tensor(d["X_support"], dtype=torch.float32),
        torch.tensor(d["Y_support"], dtype=torch.float32),
        torch.tensor(d["X_query"],   dtype=torch.float32),
        torch.tensor(d["Y_query"],   dtype=torch.float32),
    )


def inner_adapt(model, x_s, y_s, inner_lr, inner_steps):
    """Adapt a deep-copy of model on support set with Adam.
    Returns the adapted model (with grad-enabled parameters)."""
    adapted = copy.deepcopy(model)
    opt = torch.optim.Adam(adapted.parameters(), lr=inner_lr)
    for _ in range(inner_steps):
        opt.zero_grad()
        loss = F.mse_loss(adapted(x_s), y_s)
        loss.backward()
        opt.step()
    return adapted


# ───────────────── FOMAML Training ───────────────
def train_maml(Xs, Ys, Xq, Yq, seed=SEED):
    """First-order MAML with Adam inner and outer loops."""
    print("\n" + "=" * 50)
    print("  Training FOMAML  (Adam inner + Adam outer)")
    print("=" * 50)

    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = ChannelNet().to(DEVICE)
    meta_opt = torch.optim.Adam(model.parameters(), lr=META_LR)
    n_tasks  = Xs.shape[0]
    losses   = []

    for it in range(1, META_ITERS + 1):
        meta_opt.zero_grad()
        batch_loss = 0.0
        idxs = rng.choice(n_tasks, TASK_BATCH, replace=False)

        for idx in idxs:
            x_s = Xs[idx, :N_SHOTS].to(DEVICE)
            y_s = Ys[idx, :N_SHOTS].to(DEVICE)
            x_q = Xq[idx].to(DEVICE)
            y_q = Yq[idx].to(DEVICE)

            # Inner loop: adapt with Adam on support set
            adapted = inner_adapt(model, x_s, y_s, INNER_LR, INNER_STEPS)

            # Query loss on adapted model
            q_pred = adapted(x_q)
            q_loss = F.mse_loss(q_pred, y_q)
            q_loss.backward()

            # FOMAML: use adapted model's gradients as surrogate for meta-gradient
            with torch.no_grad():
                for p_orig, p_adapt in zip(model.parameters(), adapted.parameters()):
                    if p_adapt.grad is not None:
                        if p_orig.grad is None:
                            p_orig.grad = p_adapt.grad.clone() / TASK_BATCH
                        else:
                            p_orig.grad += p_adapt.grad / TASK_BATCH

            batch_loss += q_loss.item()

        meta_opt.step()
        avg_loss = batch_loss / TASK_BATCH
        losses.append(avg_loss)
        if it % 10 == 0:
            print(f"  MAML iter {it:4d}/{META_ITERS}  loss = {avg_loss:.6f}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "maml_model.pt"))
    print("[OK] MAML model saved.")
    return losses


# ───────────────── Reptile Training ──────────────
def train_reptile(Xs, Ys, Xq, Yq, seed=SEED + 1):
    """Reptile meta-learning with Adam inner loop."""
    print("\n" + "=" * 50)
    print("  Training Reptile  (Adam inner)")
    print("=" * 50)

    set_seed(seed)
    rng = np.random.default_rng(seed)
    model    = ChannelNet().to(DEVICE)
    n_tasks  = Xs.shape[0]
    losses   = []

    for it in range(1, META_ITERS + 1):
        old_state = {k: v.clone() for k, v in model.state_dict().items()}
        batch_loss = 0.0
        idxs = rng.choice(n_tasks, TASK_BATCH, replace=False)

        weight_diffs = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}

        for idx in idxs:
            x_s = Xs[idx, :N_SHOTS].to(DEVICE)
            y_s = Ys[idx, :N_SHOTS].to(DEVICE)
            x_q = Xq[idx].to(DEVICE)
            y_q = Yq[idx].to(DEVICE)

            # Reset to meta-params
            model.load_state_dict(old_state)

            # Inner loop: adapt with Adam
            inner_opt = torch.optim.Adam(model.parameters(), lr=INNER_LR)
            for _ in range(INNER_STEPS):
                inner_opt.zero_grad()
                F.mse_loss(model(x_s), y_s).backward()
                inner_opt.step()

            # Track query loss for logging
            with torch.no_grad():
                q_loss = F.mse_loss(model(x_q), y_q)
                batch_loss += q_loss.item()

            # Accumulate (theta_adapted - theta_old)
            for k, v in model.state_dict().items():
                weight_diffs[k] += (v - old_state[k])

        # Reptile update: theta <- theta + epsilon * mean(theta' - theta)
        eps = META_LR * (1 - it / META_ITERS)   # linearly decaying step
        new_state = {k: old_state[k] + eps * weight_diffs[k] / TASK_BATCH
                     for k in old_state}
        model.load_state_dict(new_state)

        avg_loss = batch_loss / TASK_BATCH
        losses.append(avg_loss)
        if it % 10 == 0:
            print(f"  Reptile iter {it:4d}/{META_ITERS}  loss = {avg_loss:.6f}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "reptile_model.pt"))
    print("[OK] Reptile model saved.")
    return losses


# ──────────────── Plot training loss ─────────────
def plot_losses(maml_losses, reptile_losses):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def rolling_stats(vals, window):
        vals = np.asarray(vals, dtype=np.float32)
        means, stds = [], []
        for start in range(len(vals) - window + 1):
            chunk = vals[start:start + window]
            means.append(chunk.mean())
            stds.append(chunk.std())
        x = np.arange(window, len(vals) + 1)
        return x, np.array(means), np.array(stds)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_specs = [
        ("MAML", maml_losses, 20, "#1f77b4"),
        ("Reptile", reptile_losses, 40, "#2ca02c"),
    ]

    for name, losses, window, color in plot_specs:
        x, smoothed, std = rolling_stats(losses, window)
        ax.plot(x, smoothed, label=name, color=color, alpha=0.9, linewidth=1.7)
        ax.fill_between(
            x,
            smoothed - std,
            smoothed + std,
            color=color,
            alpha=0.15,
        )
        final_value = smoothed[-1]
        ax.axhline(final_value, color=color, linestyle="--", linewidth=1.0, alpha=0.8)
        ax.text(
            x[-1],
            final_value,
            f"{name} final",
            color=color,
            fontsize=8,
            ha="right",
            va="bottom",
        )

    ax.set_xlabel("Meta-training Iteration")
    ax.set_ylabel("Query Loss (MSE)")
    ax.set_title("Meta-Training Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(RESULTS_DIR, "plot_loss.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Loss curve saved -> {path}")


# ───────────────── Main ──────────────────────────
def main():
    print("Loading training data...")
    Xs, Ys, Xq, Yq = load_tasks("train")
    print(f"  {Xs.shape[0]} tasks loaded.  Device: {DEVICE}  Seed: {SEED}")

    maml_losses    = train_maml(Xs, Ys, Xq, Yq)
    reptile_losses = train_reptile(Xs, Ys, Xq, Yq)
    plot_losses(maml_losses, reptile_losses)

    print("\nTraining complete.  Run  python test.py  to evaluate.")


if __name__ == "__main__":
    main()
