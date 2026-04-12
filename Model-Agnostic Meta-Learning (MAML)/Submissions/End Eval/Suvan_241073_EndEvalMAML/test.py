# test.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

# 1. Define the Neural Network exactly as it is in train.py
class ChannelEstimator(nn.Module):
    def __init__(self, input_size=16, hidden_size=64, output_size=16):
        super(ChannelEstimator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

def compute_nmse_db(mse, true_y):
    """
    Converts MSE to Normalized Mean Squared Error (NMSE) in decibels (dB).
    Lower error = better for NMSE.
    """
    power = torch.mean(true_y ** 2).item()
    if power == 0:
        return 0
    nmse_linear = mse / power
    return 10 * np.log10(nmse_linear + 1e-10)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "channel_data.npz")
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("="*40)
    print("GENERATING PLOT 1: TRAINING LOSS")
    print("="*40)
    
    try:
        # Load the losses saved by train.py
        meta_losses = np.load(os.path.join(current_dir, "meta_losses.npy"))
        plt.figure(figsize=(8, 5))
        
        plt.plot(meta_losses, label="Meta-Training Loss", color='green', alpha=0.3)
        smoothed = np.convolve(meta_losses, np.ones(20)/20, mode='valid')
        plt.plot(smoothed, color='darkgreen', linewidth=2, label="Smoothed Loss")
        
        plt.title("Plot 1: MAML/Reptile Training Loss Curve")
        plt.xlabel("Meta-Training Iteration")
        plt.ylabel("Loss Value (MSE)")
        plt.legend()
        plt.grid(True)
        
        loss_plot_path = os.path.join(results_dir, "plot_loss.png")
        plt.savefig(loss_plot_path)
        print(f"Success! Plot 1 saved to {loss_plot_path}")
        plt.close() # Close figure to prevent overlap
    except FileNotFoundError:
        print("Error: Could not find meta_losses.npy. Run train.py first!")

    print("\n" + "="*40)
    print("EVALUATING MODELS & GENERATING PLOT 2")
    print("="*40)

    data = np.load(data_path, allow_pickle=True)
    test_tasks = data['test']
    num_test_tasks = len(test_tasks)

    maml_model = ChannelEstimator()
    baseline_model = ChannelEstimator()
    
    # Load weights
    maml_model.load_state_dict(torch.load(os.path.join(current_dir, "maml_model.pth"), weights_only=True))
    baseline_model.load_state_dict(torch.load(os.path.join(current_dir, "baseline_model.pth"), weights_only=True))

    num_steps = 20
    maml_nmse_history = np.zeros(num_steps + 1)
    base_nmse_history = np.zeros(num_steps + 1)
    
    criterion = nn.MSELoss()
    inner_lr = 0.01

    print(f"Running adaptation on {num_test_tasks} test tasks...")
    
    for i, task in enumerate(test_tasks):
        x_s = torch.tensor(task['X_support'], dtype=torch.float32)
        y_s = torch.tensor(task['Y_support'], dtype=torch.float32)
        x_q = torch.tensor(task['X_query'], dtype=torch.float32)
        y_q = torch.tensor(task['Y_query'], dtype=torch.float32)

        temp_maml = copy.deepcopy(maml_model)
        temp_base = copy.deepcopy(baseline_model)
        
        opt_maml = optim.Adam(temp_maml.parameters(), lr=inner_lr)
        opt_base = optim.Adam(temp_base.parameters(), lr=inner_lr)
        
        # Step 0 (Pre-adaptation)
        with torch.no_grad():
            maml_nmse_history[0] += compute_nmse_db(criterion(temp_maml(x_q), y_q).item(), y_q)
            base_nmse_history[0] += compute_nmse_db(criterion(temp_base(x_q), y_q).item(), y_q)
            
        # Adapt for 1 to 5 steps
        for step in range(1, num_steps + 1):
            opt_maml.zero_grad()
            loss_maml = criterion(temp_maml(x_s), y_s)
            loss_maml.backward()
            opt_maml.step()
            
            opt_base.zero_grad()
            loss_base = criterion(temp_base(x_s), y_s)
            loss_base.backward()
            opt_base.step()
            
            with torch.no_grad():
                maml_nmse_history[step] += compute_nmse_db(criterion(temp_maml(x_q), y_q).item(), y_q)
                base_nmse_history[step] += compute_nmse_db(criterion(temp_base(x_q), y_q).item(), y_q)
                    
    # Average the NMSE across all test tasks
    maml_nmse_history /= num_test_tasks
    base_nmse_history /= num_test_tasks

    print("\n" + "="*40)
    print(f"RESULTS ACROSS {num_steps} ADAPTATION STEPS")
    print("="*40)

    # Loop through all recorded steps and print when it's a multiple of 5
    for step in range(num_steps + 1):
        if step > 0 and step % 5 == 0:
            print(f"--- {step}-Step Adaptation ---")
            print(f"Baseline model: {base_nmse_history[step]:.2f} dB")
            print(f"MAML model:     {maml_nmse_history[step]:.2f} dB\n")

    # Plot 2 — MAML vs Baseline Comparison
    plt.figure(figsize=(8, 5))
    steps = np.arange(num_steps + 1)
    
    plt.plot(steps, maml_nmse_history, marker='o', color='red', linewidth=2, label="MAML (Ours)")
    plt.plot(steps, base_nmse_history, marker='s', color='blue', linewidth=2, linestyle='--', label="Baseline")
    
    plt.title("Plot 2: MAML vs Baseline Comparison")
    plt.xlabel("Number of Adaptation Steps")
    plt.ylabel("Test Error (NMSE in dB)")
    plt.xticks(steps)
    plt.legend()
    plt.grid(True)
    
    comp_plot_path = os.path.join(results_dir, "plot_comparison.png")
    plt.savefig(comp_plot_path)
    print(f"Success! Plot 2 saved to {comp_plot_path}")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
