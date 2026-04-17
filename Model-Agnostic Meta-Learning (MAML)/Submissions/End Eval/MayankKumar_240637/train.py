import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

# 1. MAML TRAINING
def train_maml():
    print("\n--- Starting MAML Meta-Training ---")
    data = np.load("wireless_dataset.npz")
    train_x_supp, train_y_supp = torch.tensor(data['train_x_support']), torch.tensor(data['train_y_support'])
    train_x_query, train_y_query = torch.tensor(data['train_x_query']), torch.tensor(data['train_y_query'])

    num_tasks = train_x_supp.shape[0]
    meta_model = WirelessNet()
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.001) 
    
    iterations, meta_batch_size, inner_steps, inner_lr = 500, 4, 5, 0.01      
    meta_losses = []

    for iteration in range(iterations):
        meta_optimizer.zero_grad()
        meta_batch_loss = 0.0
        task_indices = np.random.choice(num_tasks, meta_batch_size, replace=False)
        
        for task_idx in task_indices:
            x_supp, y_supp = train_x_supp[task_idx], train_y_supp[task_idx]
            x_query, y_query = train_x_query[task_idx], train_y_query[task_idx]
            fast_weights = {name: param for name, param in meta_model.named_parameters()}

            for _ in range(inner_steps):
                preds = functional_forward(x_supp, fast_weights)
                loss = F.mse_loss(preds, y_supp)
                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
                fast_weights = {name: param - inner_lr * grad for ((name, param), grad) in zip(fast_weights.items(), grads)}

            query_loss = F.mse_loss(functional_forward(x_query, fast_weights), y_query)
            meta_batch_loss += query_loss

        meta_batch_loss = meta_batch_loss / meta_batch_size
        meta_batch_loss.backward()  
        meta_optimizer.step()
        meta_losses.append(meta_batch_loss.item())

        if (iteration + 1) % 50 == 0:
            print(f"Iteration {iteration + 1}/{iterations} | MAML Loss: {meta_batch_loss.item():.4f}")

    torch.save(meta_model.state_dict(), "meta_model.pth")
    return meta_losses

# 2. REPTILE TRAINING
def train_reptile():
    print("\n--- Starting Reptile Meta-Training ---")
    data = np.load("wireless_dataset.npz")
    train_x_supp, train_y_supp = torch.tensor(data['train_x_support']), torch.tensor(data['train_y_support'])
    train_x_query, train_y_query = torch.tensor(data['train_x_query']), torch.tensor(data['train_y_query'])

    num_tasks = train_x_supp.shape[0]
    meta_model = WirelessNet()
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.001) 
    
    iterations, meta_batch_size, inner_steps, inner_lr = 500, 4, 5, 0.01      
    reptile_losses = []

    for iteration in range(iterations):
        meta_optimizer.zero_grad()
        task_indices = np.random.choice(num_tasks, meta_batch_size, replace=False)
        sum_adapted_weights = {name: torch.zeros_like(param) for name, param in meta_model.named_parameters()}
        meta_batch_loss = 0.0
        
        for task_idx in task_indices:
            x_supp, y_supp = train_x_supp[task_idx], train_y_supp[task_idx]
            x_query, y_query = train_x_query[task_idx], train_y_query[task_idx]

            task_model = WirelessNet()
            task_model.load_state_dict(meta_model.state_dict())
            task_optimizer = optim.SGD(task_model.parameters(), lr=inner_lr)

            for _ in range(inner_steps):
                loss = F.mse_loss(task_model(x_supp), y_supp)
                task_optimizer.zero_grad()
                loss.backward()
                task_optimizer.step()

            for name, param in task_model.named_parameters():
                sum_adapted_weights[name] += param.data
                
            with torch.no_grad():
                meta_batch_loss += F.mse_loss(task_model(x_query), y_query).item()

        # Reptile Outer Update
        for name, param in meta_model.named_parameters():
            avg_adapted_weight = sum_adapted_weights[name] / meta_batch_size
            param.grad = param.data - avg_adapted_weight # Treat difference as gradient for Adam
            
        meta_optimizer.step()
        
        meta_batch_loss /= meta_batch_size
        reptile_losses.append(meta_batch_loss)

        if (iteration + 1) % 50 == 0:
            print(f"Iteration {iteration + 1}/{iterations} | Reptile Loss: {meta_batch_loss:.4f}")

    torch.save(meta_model.state_dict(), "reptile_model.pth")
    return reptile_losses

# 3. BASELINE TRAINING
def train_baseline_on_test_tasks():
    print("\n--- Starting Baseline Training on Test Tasks ---")
    data = np.load("wireless_dataset.npz")
    test_x_supp, test_y_supp = torch.tensor(data['test_x_support']), torch.tensor(data['test_y_support'])
    test_x_query, test_y_query = torch.tensor(data['test_x_query']), torch.tensor(data['test_y_query'])

    num_test_tasks = test_x_supp.shape[0]
    total_steps, inner_lr = 200, 0.01
    baseline_errors = np.zeros(total_steps + 1)

    for task_idx in range(num_test_tasks):
        x_supp, y_supp = test_x_supp[task_idx], test_y_supp[task_idx]
        x_query, y_query = test_x_query[task_idx], test_y_query[task_idx]

        baseline_model = WirelessNet()
        optimizer = optim.Adam(baseline_model.parameters(), lr=inner_lr)
        
        with torch.no_grad():
            baseline_errors[0] += F.mse_loss(baseline_model(x_query), y_query).item()

        for step in range(1, total_steps + 1):
            optimizer.zero_grad()
            loss = F.mse_loss(baseline_model(x_supp), y_supp)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                baseline_errors[step] += F.mse_loss(baseline_model(x_query), y_query).item()

    baseline_errors /= num_test_tasks
    np.save("baseline_errors.npy", baseline_errors)
    print("Baseline errors saved to baseline_errors.npy.")

# 4. PLOTTING
def plot_training_loss(maml_losses, reptile_losses):
    m_smooth = [np.mean(maml_losses[max(0, i-10):i+1]) for i in range(len(maml_losses))]
    r_smooth = [np.mean(reptile_losses[max(0, i-10):i+1]) for i in range(len(reptile_losses))]
    
    os.makedirs("results", exist_ok=True) 
    plt.figure(figsize=(8, 5))
    plt.plot(m_smooth, label="MAML Query Loss", color='green')
    plt.plot(r_smooth, label="Reptile Query Loss", color='purple')
    plt.title("Plot 1: Meta-Training Loss Curves")
    plt.xlabel("Meta-Training Iteration")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plot_loss.png")
    print("\nPlot 1 saved to results/plot_loss.png.")

if __name__ == "__main__":
    maml_losses = train_maml()
    reptile_losses = train_reptile()
    train_baseline_on_test_tasks()
    plot_training_loss(maml_losses, reptile_losses)
