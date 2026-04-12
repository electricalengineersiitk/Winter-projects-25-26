print("TESTING STARTED")

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


test_tasks = np.load("data/test.npy", allow_pickle=True)

model = Net().to(device)
model.load_state_dict(torch.load("models/maml.pth"))

loss_fn = nn.MSELoss()


def evaluate_maml(steps):
    total_error = 0

    for task in test_tasks:
        X_s, Y_s, X_q, Y_q = task

        X_s = torch.tensor(X_s, dtype=torch.float32).to(device)
        Y_s = torch.tensor(Y_s, dtype=torch.float32).to(device)
        X_q = torch.tensor(X_q, dtype=torch.float32).to(device)
        Y_q = torch.tensor(Y_q, dtype=torch.float32).to(device)

        temp_model = Net().to(device)
        temp_model.load_state_dict(model.state_dict())

        optimizer = torch.optim.Adam(temp_model.parameters(), lr=0.02)

        for _ in range(steps):
            loss = loss_fn(temp_model(X_s), Y_s)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_error += loss_fn(temp_model(X_q), Y_q).item()

    return total_error / len(test_tasks)


def evaluate_baseline():
    total_error = 0

    for task in test_tasks:
        X_s, Y_s, X_q, Y_q = task

        X_s = torch.tensor(X_s, dtype=torch.float32).to(device)
        Y_s = torch.tensor(Y_s, dtype=torch.float32).to(device)
        X_q = torch.tensor(X_q, dtype=torch.float32).to(device)
        Y_q = torch.tensor(Y_q, dtype=torch.float32).to(device)

        model_b = Net().to(device)
        optimizer = torch.optim.Adam(model_b.parameters(), lr=0.005)

        for _ in range(50):
            loss = loss_fn(model_b(X_s), Y_s)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_b.parameters(), 5.0)
            optimizer.step()

        total_error += loss_fn(model_b(X_q), Y_q).item()

    return total_error / len(test_tasks)


steps_list = [1, 3, 5]

maml_results = []
for s in steps_list:
    print(f"Evaluating MAML with {s} steps...")
    maml_results.append(evaluate_maml(s))

print("Evaluating Baseline...")
baseline_error = evaluate_baseline()

print("\nFINAL RESULTS:")
print("MAML:", maml_results)
print("Baseline:", baseline_error)

plt.figure(figsize=(6,4))
plt.plot(steps_list, maml_results, marker='o', linewidth=2, label="MAML")
plt.axhline(y=baseline_error, color='r', linestyle='--', linewidth=2, label="Baseline")
plt.xlabel("Adaptation Steps")
plt.ylabel("Localization Error")
plt.title("MAML vs Baseline")
plt.legend()
plt.grid()
plt.savefig("results/plot_comparison.png")

print("TESTING COMPLETE")