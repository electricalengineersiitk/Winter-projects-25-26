print("TRAINING STARTED")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x, params=None):
        if params is None:
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)
        else:
            x = torch.relu(torch.nn.functional.linear(x, params[0], params[1]))
            x = torch.relu(torch.nn.functional.linear(x, params[2], params[3]))
            return torch.nn.functional.linear(x, params[4], params[5])


def clone_params(model):
    return [p.clone() for p in model.parameters()]


train_tasks = np.load("data/train.npy", allow_pickle=True)

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

inner_lr = 0.005
inner_steps = 5
meta_iters = 150

loss_fn = nn.MSELoss()
losses = []

for it in range(meta_iters):
    meta_loss = 0

    for task in train_tasks[:10]:
        X_s, Y_s, X_q, Y_q = task

        X_s = torch.tensor(X_s, dtype=torch.float32).to(device)
        Y_s = torch.tensor(Y_s, dtype=torch.float32).to(device)
        X_q = torch.tensor(X_q, dtype=torch.float32).to(device)
        Y_q = torch.tensor(Y_q, dtype=torch.float32).to(device)

        fast_weights = clone_params(model)

        for _ in range(inner_steps):
            preds = model(X_s, fast_weights)
            loss = loss_fn(preds, Y_s)
            grads = torch.autograd.grad(loss, fast_weights, create_graph=True)
            fast_weights = [w - inner_lr * g for w, g in zip(fast_weights, grads)]

        preds_q = model(X_q, fast_weights)
        meta_loss += loss_fn(preds_q, Y_q)

    meta_loss /= 10

    optimizer.zero_grad()
    meta_loss.backward()
    optimizer.step()

    losses.append(meta_loss.item())

    if it % 10 == 0:
        print(f"Iteration {it}, Loss: {meta_loss.item():.4f}")


os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/maml.pth")

os.makedirs("results", exist_ok=True)
plt.figure(figsize=(6,4))
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid()
plt.savefig("results/plot_loss.png")

print("TRAINING COMPLETE")