import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
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



def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "channel_data.npz")

    print("Loading dataset...")
    data = np.load(data_path, allow_pickle=True)
    train_tasks = data['train']

    # Models
    meta_model = ChannelEstimator()
    baseline_model = ChannelEstimator()

    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.001)

    criterion = nn.MSELoss()

    # Hyperparameters
    inner_lr = 0.01
    num_epochs = 1500

    meta_losses = []

    print("\nStarting Meta-Training...\n")

    for epoch in range(num_epochs):

        task = np.random.choice(train_tasks)

        x_s = torch.tensor(task['X_support'], dtype=torch.float32)
        y_s = torch.tensor(task['Y_support'], dtype=torch.float32)
        x_q = torch.tensor(task['X_query'], dtype=torch.float32)
        y_q = torch.tensor(task['Y_query'], dtype=torch.float32)

      
        num_inner_steps = np.random.choice([4, 5, 6])

        theta = copy.deepcopy(meta_model.state_dict())

        temp_model = ChannelEstimator()
        temp_model.load_state_dict(theta)

        temp_optimizer = optim.Adam(temp_model.parameters(), lr=inner_lr)

        for _ in range(num_inner_steps):
            temp_optimizer.zero_grad()
            loss = criterion(temp_model(x_s), y_s)
            loss.backward()
            temp_optimizer.step()

        meta_optimizer.zero_grad()
        temp_state = temp_model.state_dict()

        for name, param in meta_model.named_parameters():
            param.grad = param.data - temp_state[name].data

        meta_optimizer.step()

     
        with torch.no_grad():
            q_loss = criterion(temp_model(x_q), y_q)
            meta_losses.append(q_loss.item())

        
        baseline_optimizer.zero_grad()
        preds = baseline_model(torch.cat([x_s, x_q]))
        b_loss = criterion(preds, torch.cat([y_s, y_q]))
        b_loss.backward()
        baseline_optimizer.step()

       
        if (epoch + 1) % 20 == 0:
            print(f"[{epoch+1:04d}] Meta Loss: {q_loss.item():.5f} | Baseline: {b_loss.item():.5f}")

    print("\nTraining complete. Saving outputs...\n")

    np.save(os.path.join(current_dir, "meta_losses.npy"), np.array(meta_losses))
    torch.save(meta_model.state_dict(), os.path.join(current_dir, "maml_model.pth"))
    torch.save(baseline_model.state_dict(), os.path.join(current_dir, "baseline_model.pth"))

    print("Saved successfully!")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()