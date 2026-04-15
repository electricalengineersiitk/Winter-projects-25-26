import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

class WirelessNet(nn.Module):
    def __init__(self, input_size=64, hidden_size=64, output_size=64):
        super(WirelessNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def functional_forward(x, weights):
    x = F.linear(x, weights['layer1.weight'], weights['layer1.bias'])
    x = F.relu(x)
    x = F.linear(x, weights['layer2.weight'], weights['layer2.bias'])
    x = F.relu(x)
    x = F.linear(x, weights['layer3.weight'], weights['layer3.bias'])
    return x

def evaluate_meta_model(model_path):
    print(f"Adapting {model_path} on unseen test tasks...")
    data = np.load("wireless_dataset.npz")
    test_x_supp, test_y_supp = torch.tensor(data['test_x_support']), torch.tensor(data['test_y_support'])
    test_x_query, test_y_query = torch.tensor(data['test_x_query']), torch.tensor(data['test_y_query'])

    num_test_tasks = test_x_supp.shape[0]
    eval_steps, inner_lr = 20, 0.01

    meta_model = WirelessNet()
    meta_model.load_state_dict(torch.load(model_path, weights_only=True))
    
    errors = np.zeros(eval_steps + 1)

    for task_idx in range(num_test_tasks):
        x_supp, y_supp = test_x_supp[task_idx], test_y_supp[task_idx]
        x_query, y_query = test_x_query[task_idx], test_y_query[task_idx]

        fast_weights = {name: param for name, param in meta_model.named_parameters()}
        
        with torch.no_grad():
            errors[0] += F.mse_loss(functional_forward(x_query, fast_weights), y_query).item()

        for step in range(1, eval_steps + 1):
            loss = F.mse_loss(functional_forward(x_supp, fast_weights), y_supp)
            grads = torch.autograd.grad(loss, fast_weights.values())
            fast_weights = {name: param - inner_lr * grad for ((name, param), grad) in zip(fast_weights.items(), grads)}
            
            with torch.no_grad():
                errors[step] += F.mse_loss(functional_forward(x_query, fast_weights), y_query).item()

    errors /= num_test_tasks
    return errors

if __name__ == "__main__":
    maml_errors = evaluate_meta_model("meta_model.pth")
    reptile_errors = evaluate_meta_model("reptile_model.pth")
    baseline_errors = np.load("baseline_errors.npy")

    print("\n" + "="*65)
    print(" RESULTS FOR YOUR README.MD TABLE")
    print("="*65)
    print(f"Method                 | 5-shot Error | 20-shot Error")
    print(f"-----------------------------------------------------")
    print(f"Basic model (scratch)  | {10 * np.log10(baseline_errors[5]):7.2f} dB | {10 * np.log10(baseline_errors[20]):7.2f} dB")
    print(f"MAML                   | {10 * np.log10(maml_errors[5]):7.2f} dB | {10 * np.log10(maml_errors[20]):7.2f} dB")
    print(f"Reptile                | {10 * np.log10(reptile_errors[5]):7.2f} dB | {10 * np.log10(reptile_errors[20]):7.2f} dB")
    print("="*65 + "\n")

    os.makedirs("results", exist_ok=True)
    steps = range(21)

    plt.figure(figsize=(8, 5))
    plt.plot(steps, baseline_errors[:21], label="Baseline (Random Init)", color='red', linestyle='--')
    plt.plot(steps, maml_errors[:21], label="MAML", color='green', linewidth=2)
    plt.plot(steps, reptile_errors[:21], label="Reptile", color='purple', linewidth=2)
    
    plt.title("Plot 2: MAML vs Reptile vs Baseline Adaptation")
    plt.xlabel("Number of Adaptation Steps (Shots)")
    plt.ylabel("MSE Error on Query Set")
    plt.xticks(np.arange(0, 21, 5))
    plt.legend()
    plt.grid(True)
    
    plt.savefig("results/plot_comparison.png")
    print("Plot 2 successfully saved to results/plot_comparison.png.")
