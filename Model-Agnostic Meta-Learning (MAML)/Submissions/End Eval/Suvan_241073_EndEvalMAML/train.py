import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

# 1. Define the Neural Network: 3 layers, 64 neurons each
class ChannelEstimator(nn.Module):
    def __init__(self, input_size=16, hidden_size=64, output_size=16):
        super(ChannelEstimator, self).__init__()
        # Small network to prevent overfitting on few-shot tasks
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

def main():
    # Keep paths relative so it works on any machine
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "channel_data.npz")
    
    print("Loading dataset...")
    data = np.load(data_path, allow_pickle=True)
    train_tasks = data['train']

    # Initialize the Meta-Learning Model (Using Reptile as allowed)
    meta_model = ChannelEstimator()
    
    # Initialize Baseline Model (Train from scratch/jointly on all tasks)
    baseline_model = ChannelEstimator()
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)

    # Hyperparameters per the project tips
    inner_lr = 0.01 
    outer_lr = 0.001
    num_inner_steps = 5
    num_epochs = 1500  # Meta-training iterations

    # Adam optimizer for the outer loop
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=outer_lr)
    criterion = nn.MSELoss()

    meta_losses = []

    print("Starting Meta-Training (Reptile Algorithm)...")
    for epoch in range(num_epochs):
        # Sample a random training task
        task = np.random.choice(train_tasks)
        x_s = torch.tensor(task['X_support'], dtype=torch.float32)
        y_s = torch.tensor(task['Y_support'], dtype=torch.float32)
        x_q = torch.tensor(task['X_query'], dtype=torch.float32)
        y_q = torch.tensor(task['Y_query'], dtype=torch.float32)

        # ---------------------------------------------------------
        # META-LEARNING (REPTILE)
        # ---------------------------------------------------------
        theta = copy.deepcopy(meta_model.state_dict())
        
        temp_model = ChannelEstimator()
        temp_model.load_state_dict(theta)
        temp_optimizer = optim.Adam(temp_model.parameters(), lr=inner_lr)

        # Step 2: Do a few gradient steps on the support set -> get theta'
        for _ in range(num_inner_steps):
            temp_optimizer.zero_grad()
            preds = temp_model(x_s)
            loss = criterion(preds, y_s)
            loss.backward()
            temp_optimizer.step()

        # Step 3: Nudge theta toward theta'
        meta_optimizer.zero_grad()
        temp_state = temp_model.state_dict()
        for name, param in meta_model.named_parameters():
            param.grad = param.data - temp_state[name].data
        meta_optimizer.step()

        # Evaluate on query set for saving loss history
        with torch.no_grad():
            q_loss = criterion(temp_model(x_q), y_q)
            meta_losses.append(q_loss.item())

        # ---------------------------------------------------------
        # BASELINE TRAINING
        # ---------------------------------------------------------
        baseline_optimizer.zero_grad()
        b_preds = baseline_model(torch.cat([x_s, x_q]))
        b_loss = criterion(b_preds, torch.cat([y_s, y_q]))
        b_loss.backward()
        baseline_optimizer.step()

        # Print loss every 10 iterations
        if (epoch + 1) % 10 == 0:
            print(f"Iteration {epoch+1}/{num_epochs} | Meta-Loss (Query): {q_loss.item():.4f} | Baseline Loss: {b_loss.item():.4f}")

    print("\nTraining complete! Saving models and loss data...")
    
    # Save the loss data for test.py to plot
    np.save(os.path.join(current_dir, "meta_losses.npy"), np.array(meta_losses))
    
    # Save both models for test.py
    torch.save(meta_model.state_dict(), os.path.join(current_dir, "maml_model.pth"))
    torch.save(baseline_model.state_dict(), os.path.join(current_dir, "baseline_model.pth"))
    print("Files successfully saved!")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
