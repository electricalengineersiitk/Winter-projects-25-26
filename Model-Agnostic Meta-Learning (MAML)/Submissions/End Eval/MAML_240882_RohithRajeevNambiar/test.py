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

test_Xs = data["test_Xs"]
test_Ys = data["test_Ys"]
test_Xq = data["test_Xq"]
test_Yq = data["test_Yq"]
support = test_Xs.shape[1]
print(f"Testing with {support}-shot tasks")

m = test_Xs.shape[-1]

model = nn.Sequential(
    nn.Linear(m, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

model.load_state_dict(torch.load(os.path.join(BASE_DIR, "model.pth")))
model.eval()
loss_fn = nn.MSELoss()
adapt_steps = [1, 3, 5, 10]
maml_results = []
baseline_results = []

for steps in adapt_steps:
    maml_losses = []
    baseline_losses = []

    for i in range(len(test_Xs)):
        Xs = torch.tensor(test_Xs[i], dtype=torch.float32)
        Ys = torch.tensor(test_Ys[i], dtype=torch.float32)
        Xq = torch.tensor(test_Xq[i], dtype=torch.float32)
        Yq = torch.tensor(test_Yq[i], dtype=torch.float32)

        adapted = copy.deepcopy(model)
        adapted.train()

        for _ in range(steps):
            loss = loss_fn(adapted(Xs), Ys)
            grads = torch.autograd.grad(loss, adapted.parameters(), create_graph=False)
            with torch.no_grad():
                for p, g in zip(adapted.parameters(), grads):
                    p -= 0.01 * g

        maml_losses.append(loss_fn(adapted(Xq), Yq).item())
        baseline = nn.Sequential(
            nn.Linear(m, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        opt = optim.Adam(baseline.parameters(), lr=0.01)
        for _ in range(steps):
            loss = loss_fn(baseline(Xs), Ys)
            opt.zero_grad()
            loss.backward()
            opt.step()

        baseline_losses.append(loss_fn(baseline(Xq), Yq).item())

    maml_results.append(np.mean(maml_losses))
    baseline_results.append(np.mean(baseline_losses))

maml_db = [10 * np.log10(x) for x in maml_results]
baseline_db = [10 * np.log10(x) for x in baseline_results]

print("\nResults (NMSE in dB):")
print("Steps |   MAML (dB)   | Baseline (dB)")
print("--------------------------------------")
for i, steps in enumerate(adapt_steps):
    print(f"{steps:5d} | {maml_db[i]:12.2f} | {baseline_db[i]:14.2f}")

plt.figure()
plt.plot(adapt_steps, maml_db, marker='o', label="MAML")
plt.plot(adapt_steps, baseline_db, marker='o', label="Baseline")
plt.xlabel("Adaptation Steps")
plt.ylabel("NMSE (dB)")
plt.title("MAML vs Baseline (dB)")
plt.legend()
plt.grid()
plt.savefig(os.path.join(RESULTS_DIR, "plot_comparison_db.png"))
plt.close()

print("\nTesting complete.")
