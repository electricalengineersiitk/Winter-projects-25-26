import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os

os.makedirs("results", exist_ok=True)

class MetaNet(nn.Module):
    def __init__(self, input_dim=10):
        super(MetaNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), 
            nn.Tanh(),
            nn.Linear(64, 64), 
            nn.Tanh(),
            nn.Linear(64, input_dim)
        )
    def forward(self, x): return self.net(x)
data = np.load("data/dataset.npz", allow_pickle=True)
test_tasks = data["test_tasks"][:20] 
meta_model = MetaNet()
meta_model.load_state_dict(torch.load("best_meta_model.pth"))
loss_fn = nn.MSELoss()

m_errs = []
b_errs = []

print(" Evaluating MAML vs Baseline on 20 tasks...")

for task in test_tasks:
    Xs, Ys, Xq, Yq = [torch.tensor(t, dtype=torch.float32) for t in task]

    m_test = copy.deepcopy(meta_model)
    m_opt = optim.SGD(m_test.parameters(), lr=0.01) 
    for _ in range(5):
        l = loss_fn(m_test(Xs), Ys)
        m_opt.zero_grad()
        l.backward()
        m_opt.step()
    
    m_errs.append(loss_fn(m_test(Xq), Yq).item())

    b_test = MetaNet() 
    b_opt = optim.Adam(b_test.parameters(), lr=0.01)
    for _ in range(200): 
        lb = loss_fn(b_test(Xs), Ys)
        b_opt.zero_grad()
        lb.backward()
        b_opt.step()
    
    b_errs.append(loss_fn(b_test(Xq), Yq).item())

np.savez("results/test_data.npz", m_errs=np.array(m_errs), b_errs=np.array(b_errs))

print(f"\n Evaluation Results:")
print(f" MAML Average Error: {np.mean(m_errs):.6f}")
print(f" Baseline Average Error: {np.mean(b_errs):.6f}")
print(f" Data saved for plotting in results/test_data.npz")