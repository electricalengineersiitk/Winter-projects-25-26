import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --- Model & Metrics ---
class ChannelEstimator(nn.Module):
    """3-Layer MLP for Channel Estimation."""
    def __init__(self, input_size=16, hidden_size=64, output_size=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x): return self.net(x)

def get_nmse_db(mse, target):
    """Calculates Normalized Mean Square Error in decibels."""
    power = torch.mean(target ** 2).item()
    return 10 * np.log10((mse / (power + 1e-10)) + 1e-10)

# --- Plotting Module ---
class Visualizer:
    @staticmethod
    def plot_loss(meta_losses, path):
        plt.figure(figsize=(7, 4))
        plt.plot(meta_losses, alpha=0.2, color='gray', label='Raw')
        # Apply 20-period moving average
        smooth = np.convolve(meta_losses, np.ones(20)/20, mode='valid')
        plt.plot(smooth, color='tab:blue', label='Smoothed')
        plt.title("Meta-Training Convergence")
        plt.ylabel("MSE"); plt.legend(); plt.grid(True)
        plt.savefig(path); plt.close()

    @staticmethod
    def plot_comparison(maml_hist, base_hist, path):
        plt.figure(figsize=(7, 4))
        plt.plot(maml_hist, 'ro-', label="MAML (Meta-Learned)")
        plt.plot(base_hist, 'bs--', label="Baseline (Transfer)")
        plt.title("Adaptation Speed: MAML vs Baseline")
        plt.xlabel("Gradient Steps"); plt.ylabel("NMSE (dB)")
        plt.legend(); plt.grid(True)
        plt.savefig(path); plt.close()

# --- Evaluation Core ---
def run_adaptation_test(model, tasks, steps, lr):
    """Simulates few-shot learning for a given model."""
    criterion = nn.MSELoss()
    history = np.zeros(steps + 1)

    for task in tasks:
        # Data preparation
        xs, ys = torch.tensor(task["X_support"]), torch.tensor(task["Y_support"])
        xq, yq = torch.tensor(task["X_query"]), torch.tensor(task["Y_query"])
        
        # Clone model to prevent leaking updates between tasks
        temp_model = copy.deepcopy(model)
        optimizer = optim.Adam(temp_model.parameters(), lr=lr)

        for step in range(steps + 1):
            # Record performance on query set
            with torch.no_grad():
                loss_q = criterion(temp_model(xq), yq).item()
                history[step] += get_nmse_db(loss_q, yq)

            # Update on support set (skip after last step)
            if step < steps:
                optimizer.zero_grad()
                criterion(temp_model(xs), ys).backward()
                optimizer.step()

    return history / len(tasks)

# --- Execution ---
def main():
    # Setup paths
    root = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(root, "results")
    os.makedirs(res_dir, exist_ok=True)

    # 1. Plot Training History
    if os.path.exists("meta_losses.npy"):
        Visualizer.plot_loss(np.load("meta_losses.npy"), f"{res_dir}/loss.png")

    # 2. Load Evaluation Data & Models
    data = np.load("channel_data.npz", allow_pickle=True)
    maml = ChannelEstimator(); maml.load_state_dict(torch.load("maml_model.pth"))
    base = ChannelEstimator(); base.load_state_dict(torch.load("baseline_model.pth"))

    # 3. Compare Models
    print("🚀 Evaluating Meta-Adaptation...")
    maml_perf = run_adaptation_test(maml, data["test"], steps=20, lr=0.01)
    base_perf = run_adaptation_test(base, data["test"], steps=20, lr=0.01)

    # 4. Finalize
    Visualizer.plot_comparison(maml_perf, base_perf, f"{res_dir}/comparison.png")
    print(f"✅ Success! Results saved to {res_dir}")

if __name__ == "__main__":
    main()