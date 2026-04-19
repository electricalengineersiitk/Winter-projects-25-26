import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from train import WirelessNet

def test_model(path, x_s, y_s, x_q, y_q):
    model = WirelessNet()
    model.load_state_dict(torch.load(path))
    opt = optim.Adam(model.parameters(), lr=0.001)
    for _ in range(5):
        nn.MSELoss()(model(x_s), y_s).backward(); opt.step(); opt.zero_grad()
    return nn.MSELoss()(model(x_q), y_q).item()

def evaluate():
    tasks = np.load(os.path.join("end_term", "data", "test_tasks.npy"), allow_pickle=True)
    m_err, r_err, b_err = [], [], []

    for task in tasks:
        x_s, y_s = torch.tensor(task['x_s']).float(), torch.tensor(task['y_s']).float()
        x_q, y_q = torch.tensor(task['x_q']).float(), torch.tensor(task['y_q']).float()

        m_err.append(test_model(os.path.join("end_term", "maml_model.pth"), x_s, y_s, x_q, y_q))
        r_err.append(test_model(os.path.join("end_term", "reptile_model.pth"), x_s, y_s, x_q, y_q))

        b_model = WirelessNet(); b_opt = optim.Adam(b_model.parameters(), lr=0.01)
        for _ in range(200):
            nn.MSELoss()(b_model(x_s), y_s).backward(); b_opt.step(); b_opt.zero_grad()
        b_err.append(nn.MSELoss()(b_model(x_q), y_q).item())

    # CLEAN OUTPUT FORMAT
    print("\n" + "="*45)
    print(f"{'METHOD':<25} | {'AVG MSE ERROR':<15}")
    print("-" * 45)
    print(f"{'Baseline (From Scratch)':<25} | {np.mean(b_err):.6f}")
    print(f"{'Reptile Model':<25} | {np.mean(r_err):.6f}")
    print(f"{'MAML Model (Bonus)':<25} | {np.mean(m_err):.6f}")
    print("="*45 + "\n")

    plt.bar(['MAML', 'Reptile', 'Baseline'], [np.mean(m_err), np.mean(r_err), np.mean(b_err)], color=['blue', 'green', 'red'])
    plt.title("Bonus: MAML vs Reptile vs Baseline Comparison"); plt.ylabel("MSE Error")
    plt.savefig(os.path.join("end_term", "results", "plot_comparison.png"))

if __name__ == "__main__": evaluate()
