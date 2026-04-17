import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =========================================================
# 1. Model Architecture
# =========================================================
class ChannelEstimator(nn.Module):
    """
    Standard 3-layer MLP for Channel Estimation.
    Input/Output sizes typically represent Real/Imag parts of the signal.
    """
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

# =========================================================
# 2. Training Logic (Reptile Algorithm)
# =========================================================
def train_meta_model():
    # --- Configuration & Paths ---
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(curr_dir, "channel_data.npz")
    
    # Hyperparameters
    INNER_LR = 0.01 
    OUTER_LR = 0.001
    INNER_STEPS = 5
    EPOCHS = 1500

    # --- Data Loading ---
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please ensure dataset is present.")
        return

    data = np.load(data_path, allow_pickle=True)
    train_tasks = data['train']

    # --- Model Initialization ---
    meta_model = ChannelEstimator()
    baseline_model = ChannelEstimator()
    
    # Temporary model for inner-loop updates (avoids creating new objects inside the loop)
    temp_model = ChannelEstimator() 

    # Optimizers
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=OUTER_LR)
    base_optimizer = optim.Adam(baseline_model.parameters(), lr=OUTER_LR)
    criterion = nn.MSELoss()

    meta_losses = []

    print(f"Starting Training: {EPOCHS} iterations...")

    for epoch in range(EPOCHS):
        # 1. Sample Task
        task = np.random.choice(train_tasks)
        x_s = torch.tensor(task['X_support'], dtype=torch.float32)
        y_s = torch.tensor(task['Y_support'], dtype=torch.float32)
        x_q = torch.tensor(task['X_query'], dtype=torch.float32)
        y_q = torch.tensor(task['Y_query'], dtype=torch.float32)

        # ---------------------------------------------------------
        # META-LEARNING PHASE (REPTILE)
        # ---------------------------------------------------------
        # Sync temp_model with the current meta_model weights
        temp_model.load_state_dict(meta_model.state_dict())
        inner_optimizer = optim.Adam(temp_model.parameters(), lr=INNER_LR)

        # Inner Loop: Adapt to the specific task
        for _ in range(INNER_STEPS):
            inner_optimizer.zero_grad()
            loss = criterion(temp_model(x_s), y_s)
            loss.backward()
            inner_optimizer.step()

        # Outer Loop: Nudge meta_model weights toward adapted weights
        meta_optimizer.zero_grad()
        temp_params = temp_model.state_dict()
        for name, param in meta_model.named_parameters():
            # Manual gradient calculation: (Theta - Theta_adapted)
            # When meta_optimizer.step() runs, it subtracts this 'grad',
            # resulting in: Theta = Theta + lr * (Theta_adapted - Theta)
            param.grad = param.data - temp_params[name].data
        meta_optimizer.step()

        # Record Query Loss for tracking
        with torch.no_grad():
            q_loss = criterion(temp_model(x_q), y_q)
            meta_losses.append(q_loss.item())

        # ---------------------------------------------------------
        # BASELINE PHASE (Standard Supervised Learning)
        # ---------------------------------------------------------
        base_optimizer.zero_grad()
        # Baseline trains on all available data for this task jointly
        b_inputs = torch.cat([x_s, x_q])
        b_targets = torch.cat([y_s, y_q])
        b_loss = criterion(baseline_model(b_inputs), b_targets)
        b_loss.backward()
        base_optimizer.step()

        # Logging
        if (epoch + 1) % 100 == 0:
            print(f"Iter {epoch+1:4d} | Meta-Q-Loss: {q_loss.item():.5f} | Base-Loss: {b_loss.item():.5f}")

    # --- Save Artifacts ---
    print("\nSaving results...")
    np.save(os.path.join(curr_dir, "meta_losses.npy"), np.array(meta_losses))
    torch.save(meta_model.state_dict(), os.path.join(curr_dir, "maml_model.pth"))
    torch.save(baseline_model.state_dict(), os.path.join(curr_dir, "baseline_model.pth"))
    print("All files (models and loss history) saved successfully.")

# =========================================================
# 3. Entry Point
# =========================================================
if __name__ == "__main__":
    # For reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    train_meta_model()