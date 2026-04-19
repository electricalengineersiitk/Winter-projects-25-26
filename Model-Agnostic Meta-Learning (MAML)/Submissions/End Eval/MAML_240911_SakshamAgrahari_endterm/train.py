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
train_tasks = data["train_tasks"]
model = MetaNet()
loss_fn = nn.MSELoss()

outer_step_size = 0.5 
epochs = 1000
history_loss = [] 

print(" Training Best Meta-Model...")

for epoch in range(epochs):
    batch_idx = np.random.randint(0, len(train_tasks), size=4)
    meta_grad = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    
    epoch_loss = 0 

    for idx in batch_idx:
        Xs, Ys, Xq, Yq = [torch.tensor(t, dtype=torch.float32) for t in train_tasks[idx]]
        
        fast_model = copy.deepcopy(model)
        inner_opt = optim.SGD(fast_model.parameters(), lr=0.01)

        for _ in range(5): 
            loss = loss_fn(fast_model(Xs), Ys)
            inner_opt.zero_grad()
            loss.backward()
            inner_opt.step()

        for name, param in model.named_parameters():
            fast_param = dict(fast_model.named_parameters())[name]
            meta_grad[name] += (param.data - fast_param.data)
        
        epoch_loss += loss.item()

    for name, param in model.named_parameters():
        param.data -= (outer_step_size / 4) * meta_grad[name]

    history_loss.append(epoch_loss / 4)

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Avg Loss: {history_loss[-1]:.6f}")

torch.save(model.state_dict(), "best_meta_model.pth")
print(" Training Complete!")

np.save("results/train_loss.npy", np.array(history_loss))
print(" Loss history saved in results/train_loss.npy")