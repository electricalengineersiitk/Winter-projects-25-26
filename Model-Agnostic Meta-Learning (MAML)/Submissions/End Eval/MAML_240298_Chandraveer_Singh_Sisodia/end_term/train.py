import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt

class WirelessNet(nn.Module):
    def __init__(self):
        super(WirelessNet, self).__init__()
        # Requirement: 3 layers, 64 neurons each
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

def train():
    maml_model = WirelessNet()
    reptile_model = WirelessNet()

    # Requirement: Adam for both inner/outer loops
    maml_opt = optim.Adam(maml_model.parameters(), lr=0.001)
    rep_opt = optim.Adam(reptile_model.parameters(), lr=0.001)

    train_tasks = np.load(os.path.join("end_term", "data", "train_tasks.npy"), allow_pickle=True)
    m_loss_hist, r_loss_hist = [], []

    for i in range(2500): # Increased iterations for better convergence
        # Meta-batching to stabilize gradients
        m_grads = [torch.zeros_like(p) for p in maml_model.parameters()]
        r_grads = [torch.zeros_like(p) for p in reptile_model.parameters()]

        for _ in range(5): # Process 5 tasks per meta-step
            task = np.random.choice(train_tasks)
            x_s, y_s = torch.tensor(task['x_s']).float(), torch.tensor(task['y_s']).float()
            x_q, y_q = torch.tensor(task['x_q']).float(), torch.tensor(task['y_q']).float()

            # --- MAML UPDATE (First Order) ---
            temp_m = WirelessNet(); temp_m.load_state_dict(maml_model.state_dict())
            # Use 0.001 inner LR as suggested for stability
            t_opt = optim.Adam(temp_m.parameters(), lr=0.001)
            for _ in range(5):
                nn.MSELoss()(temp_m(x_s), y_s).backward(); t_opt.step(); t_opt.zero_grad()

            # Compute meta-gradient on query set
            loss_q = nn.MSELoss()(temp_m(x_q), y_q)
            loss_q.backward()

            # FIXED: Correctly iterate through parameters to avoid Shape Errors
            for j, p_temp in enumerate(temp_m.parameters()):
                if p_temp.grad is not None:
                    m_grads[j] += p_temp.grad / 5

            # --- REPTILE UPDATE ---
            w_pre = [p.clone() for p in reptile_model.parameters()]
            r_opt_inner = optim.Adam(reptile_model.parameters(), lr=0.001)
            for _ in range(5):
                nn.MSELoss()(reptile_model(x_s), y_s).backward(); r_opt_inner.step(); r_opt_inner.zero_grad()
            for j, p_post in enumerate(reptile_model.parameters()):
                r_grads[j] += (w_pre[j] - p_post) / 5

        # Apply Meta-updates to the original models
        for j, p in enumerate(maml_model.parameters()): p.grad = m_grads[j]
        for j, p in enumerate(reptile_model.parameters()): p.grad = r_grads[j]
        maml_opt.step(); rep_opt.step(); maml_opt.zero_grad(); rep_opt.zero_grad()

        # Requirement: Print loss every 10 iterations
        if i % 10 == 0:
            m_loss = nn.MSELoss()(maml_model(x_s), y_s).item()
            r_loss = nn.MSELoss()(reptile_model(x_s), y_s).item()
            m_loss_hist.append(m_loss); r_loss_hist.append(r_loss)
            if i % 100 == 0:
                print(f"Iter {i:4} | MAML Loss: {m_loss:.4f} | Reptile Loss: {r_loss:.4f}")

    # Plot 1: Training Loss Curve
    plt.figure()
    plt.plot(m_loss_hist, label="MAML")
    plt.plot(r_loss_hist, label="Reptile")
    plt.title("Training Loss: MAML vs Reptile"); plt.legend(); plt.xlabel("Iteration (x10)")
    plt.savefig(os.path.join("end_term", "results", "plot_loss.png"))

    torch.save(maml_model.state_dict(), os.path.join("end_term", "maml_model.pth"))
    torch.save(reptile_model.state_dict(), os.path.join("end_term", "reptile_model.pth"))

if __name__ == "__main__":
    train()
