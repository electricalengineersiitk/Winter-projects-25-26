import numpy as np #Importing required libraries
import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import os

DATA_PATH = "tasks.npz" #Getting dataset and trained reptile model
MODEL_PATH = "reptile_model.pth"

INNER_LR = 0.01
ADAPT_STEPS_LIST = [1, 2, 3, 4, 5] #For knowing test performances after every tests

os.makedirs("results", exist_ok=True) #Creating folders for plots


# BASELINE MODEL

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )

    def forward(self, x): #Forward function
        return self.net(x).squeeze()

data = np.load(DATA_PATH, allow_pickle=True) #Loading data
test_tasks = data["test"]

loss_fn = nn.MSELoss()

# EVALUATE FUNCTION

def evaluate(model, tasks, steps):
    errors = []

    for task in tasks:
        Xs = torch.tensor(task["X_support"], dtype=torch.float32)
        Ys = torch.tensor(task["Y_support"], dtype=torch.float32)

        Xq = torch.tensor(task["X_query"], dtype=torch.float32)
        Yq = torch.tensor(task["Y_query"], dtype=torch.float32)

        temp_model = copy.deepcopy(model)

        for _ in range(steps):
            preds = temp_model(Xs)
            loss = loss_fn(preds, Ys)

            temp_model.zero_grad()
            loss.backward()

            for p in temp_model.parameters():
                p.data -= INNER_LR * p.grad

        with torch.no_grad():
            preds = temp_model(Xq)
            mse = ((preds - Yq) ** 2).mean().item() #Computing mse manually
            errors.append(mse)

    return np.mean(errors)


#Loading trained models
reptile_model = Net()
reptile_model.load_state_dict(torch.load(MODEL_PATH))

baseline_model = Net()

#Comparing after five adaptation steps
rep_error = evaluate(reptile_model, test_tasks, 5)
base_error = evaluate(baseline_model, test_tasks, 5)

print(f"Reptile Avg Error: {rep_error:.4f}")
print(f"Baseline Avg Error: {base_error:.4f}")


# PLOT 1: PLOT LOSS

loss_history = np.load("loss_history/loss.npy")

plt.figure()
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.savefig("results/plot_loss.png")
plt.close()


# PLOT 2: PLOT COMPARISON

rep_list, base_list = [], []

for s in ADAPT_STEPS_LIST:
    rep_list.append(evaluate(reptile_model, test_tasks, s))
    base_list.append(evaluate(baseline_model, test_tasks, s))

plt.figure()
plt.plot(ADAPT_STEPS_LIST, rep_list, marker='o', label="Reptile")
plt.plot(ADAPT_STEPS_LIST, base_list, marker='o', label="Baseline")

plt.xlabel("Adaptation Steps")
plt.ylabel("MSE Error")
plt.title("Reptile vs Baseline")
plt.legend()

plt.savefig("results/plot_comparison.png")
plt.close()

print("Plots saved!")