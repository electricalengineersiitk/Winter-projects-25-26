import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "tasks.npz")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

data = np.load(DATA_PATH)
train_Xs = data["train_Xs"]
train_Ys = data["train_Ys"]
train_Xq = data["train_Xq"]
train_Yq = data["train_Yq"]

m = train_Xs.shape[-1]

model = nn.Sequential(
    nn.Linear(m, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
inner_lr = 0.01
inner_steps = 3
meta_iterations = 200
loss_history = []

for _ in range(meta_iterations):
    meta_loss = 0

    for i in range(len(train_Xs)):
        Xs = torch.tensor(train_Xs[i], dtype=torch.float32)
        Ys = torch.tensor(train_Ys[i], dtype=torch.float32)
        Xq = torch.tensor(train_Xq[i], dtype=torch.float32)
        Yq = torch.tensor(train_Yq[i], dtype=torch.float32)

        fast_weights = list(model.parameters())

        def forward_with_weights(x, weights):
            idx = 0
            x = torch.nn.functional.linear(x, weights[idx], weights[idx+1]); idx += 2
            x = torch.relu(x)
            x = torch.nn.functional.linear(x, weights[idx], weights[idx+1]); idx += 2
            x = torch.relu(x)
            x = torch.nn.functional.linear(x, weights[idx], weights[idx+1])
            return x

        for j in range(inner_steps):
            preds = forward_with_weights(Xs, fast_weights)
            loss = loss_fn(preds, Ys)
            grads = torch.autograd.grad(loss, fast_weights, create_graph=False)
            fast_weights = [
                w - inner_lr * g for w, g in zip(fast_weights, grads)
            ]

        preds_q = forward_with_weights(Xq, fast_weights)
        loss_q = loss_fn(preds_q, Yq)

        meta_loss += loss_q

    meta_loss /= len(train_Xs)
    loss_history.append(meta_loss.item())
    optimizer.zero_grad()
    meta_loss.backward()
    optimizer.step()

    if _ % 10 == 0:
        print(f"Iteration {_}, Loss: {meta_loss.item():.4f}")

MODEL_PATH = os.path.join(BASE_DIR, "model.pth")
torch.save(model.state_dict(), MODEL_PATH)

plt.figure()
plt.plot(loss_history)
plt.xlabel("Meta-Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid()
plt.savefig(os.path.join(RESULTS_DIR, "plot_loss.png"))
plt.close()

print("Training complete.")
