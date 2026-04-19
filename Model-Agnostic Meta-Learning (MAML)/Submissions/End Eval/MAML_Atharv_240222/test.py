import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy


class ChannelEstimator(nn.Module):
    def __init__(self, input_size=16, hidden_dim=64, output_size=16):
        super(ChannelEstimator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, x):
        return self.model(x)


def compute_nmse_db(mse, true_y):
    power = torch.mean(true_y ** 2).item()
    nmse = mse / (power + 1e-10)
    return 10 * np.log10(nmse + 1e-10)


def smooth_curve(data, window=25):
    kernel = np.exp(-0.5 * (np.linspace(-2, 2, window) ** 2))
    kernel /= np.sum(kernel)
    return np.convolve(data, kernel, mode='valid')

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "channel_data.npz")
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("Generating Plot 1...")

    meta_losses = np.load(os.path.join(current_dir, "meta_losses.npy"))

    plt.figure(figsize=(9, 5))

    plt.plot(meta_losses, color='lightcoral', alpha=0.4, label="Raw Loss")

    smooth = smooth_curve(meta_losses)
    plt.plot(smooth, color='darkred', linewidth=2.5, label="Smoothed")

    plt.fill_between(range(len(smooth)), smooth, alpha=0.15, color='red')

    plt.yscale("log")

    plt.title("Meta-Learning Convergence")
    plt.xlabel("Iterations")
    plt.ylabel("Loss (log scale)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.savefig(os.path.join(results_dir, "plot_loss.png"))
    plt.close()

    print("Evaluating models...")

    data = np.load(data_path, allow_pickle=True)
    test_tasks = data['test']

    maml_model = ChannelEstimator()
    baseline_model = ChannelEstimator()

    maml_model.load_state_dict(torch.load(os.path.join(current_dir, "maml_model.pth")))
    baseline_model.load_state_dict(torch.load(os.path.join(current_dir, "baseline_model.pth")))

    num_steps = 20
    maml_nmse = np.zeros(num_steps + 1)
    base_nmse = np.zeros(num_steps + 1)

    criterion = nn.MSELoss()
    inner_lr = 0.01

    for task in test_tasks:
        x_s = torch.tensor(task['X_support'], dtype=torch.float32)
        y_s = torch.tensor(task['Y_support'], dtype=torch.float32)
        x_q = torch.tensor(task['X_query'], dtype=torch.float32)
        y_q = torch.tensor(task['Y_query'], dtype=torch.float32)

        temp_maml = copy.deepcopy(maml_model)
        temp_base = copy.deepcopy(baseline_model)

        opt_maml = optim.Adam(temp_maml.parameters(), lr=inner_lr)
        opt_base = optim.Adam(temp_base.parameters(), lr=inner_lr)

        with torch.no_grad():
            maml_nmse[0] += compute_nmse_db(criterion(temp_maml(x_q), y_q).item(), y_q)
            base_nmse[0] += compute_nmse_db(criterion(temp_base(x_q), y_q).item(), y_q)

        for step in range(1, num_steps + 1):
            # MAML
            opt_maml.zero_grad()
            loss = criterion(temp_maml(x_s), y_s)
            loss.backward()
            opt_maml.step()

            # Baseline
            opt_base.zero_grad()
            loss = criterion(temp_base(x_s), y_s)
            loss.backward()
            opt_base.step()

            with torch.no_grad():
                maml_nmse[step] += compute_nmse_db(criterion(temp_maml(x_q), y_q).item(), y_q)
                base_nmse[step] += compute_nmse_db(criterion(temp_base(x_q), y_q).item(), y_q)

    maml_nmse /= len(test_tasks)
    base_nmse /= len(test_tasks)

    # Plot
    plt.figure(figsize=(9, 5))
    steps = np.arange(num_steps + 1)

    plt.plot(steps, maml_nmse, marker='D', color='purple', linewidth=2.5, label="Meta Model")
    plt.plot(steps, base_nmse, marker='x', color='orange', linestyle='-.', linewidth=2, label="Baseline")

    plt.fill_between(steps, maml_nmse, base_nmse,
                     where=(maml_nmse < base_nmse),
                     alpha=0.1, color='green')

    plt.title("Adaptation Performance Comparison")
    plt.xlabel("Steps")
    plt.ylabel("NMSE (dB)")
    plt.grid(True, linestyle=':')
    plt.legend()

    plt.savefig(os.path.join(results_dir, "plot_comparison.png"))

    print("Plots saved successfully!")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()