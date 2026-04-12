import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Main Network
class WirelessNet(nn.Module):
    def __init__(self):
        super(WirelessNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # Output: (x, y, z)
        )
        
    def forward(self, x):
        return self.net(x)

# 2. Data Loader
def load_tasks(filepath):
    data = np.load(filepath)
    # Convert arrays to float tensors
    x_sup = torch.tensor(data['x_support'], dtype=torch.float32)
    y_sup = torch.tensor(data['y_support'], dtype=torch.float32)
    x_que = torch.tensor(data['x_query'], dtype=torch.float32)
    y_que = torch.tensor(data['y_query'], dtype=torch.float32)
    return x_sup, y_sup, x_que, y_que

# 3. Reptile Meta-Learning
def train_reptile(x_sup, y_sup, outer_iters=500, inner_steps=5, inner_lr=0.01, outer_lr=0.001):
    print("\n--- Starting Reptile ---")
    model = WirelessNet()
    outer_optimizer = optim.Adam(model.parameters(), lr=outer_lr)
    loss_fn = nn.MSELoss()
    num_tasks = x_sup.shape[0]
    
    for iteration in range(outer_iters):
        outer_optimizer.zero_grad()
        
        # Sample random tasks for this batch
        batch_tasks = np.random.choice(num_tasks, size=10, replace=False)
        original_weights = [p.clone().detach() for p in model.parameters()]
        sum_adapted_weights = [torch.zeros_like(p) for p in model.parameters()]
        total_inner_loss = 0.0
        
        for task_idx in batch_tasks:
            # Adapt on task support set
            task_model = WirelessNet()
            task_model.load_state_dict(model.state_dict())
            inner_optimizer = optim.Adam(task_model.parameters(), lr=inner_lr)
            
            for _ in range(inner_steps):
                preds = task_model(x_sup[task_idx])
                loss = loss_fn(preds, y_sup[task_idx])
                inner_optimizer.zero_grad()
                loss.backward()
                inner_optimizer.step()
                
            total_inner_loss += loss.item()
            for i, p in enumerate(task_model.parameters()):
                sum_adapted_weights[i] += p.data
                
        # Nudge master model toward the task average
        for i, p in enumerate(model.parameters()):
            avg_adapted_weight = sum_adapted_weights[i] / len(batch_tasks)
            p.grad = (original_weights[i] - avg_adapted_weight)
        outer_optimizer.step()
        
        if (iteration + 1) % 10 == 0:
            print(f"Reptile {iteration + 1}/{outer_iters} | Loss: {total_inner_loss/len(batch_tasks):.4f}")
    return model

# 4. MAML Meta-Learning
def train_maml(x_sup, y_sup, x_que, y_que, outer_iters=500, inner_steps=5, inner_lr=0.01, outer_lr=0.001):
    print("\n--- Starting MAML ---")
    model = WirelessNet()
    outer_optimizer = optim.Adam(model.parameters(), lr=outer_lr)
    loss_fn = nn.MSELoss()
    num_tasks = x_sup.shape[0]
    
    for iteration in range(outer_iters):
        outer_optimizer.zero_grad()
        batch_tasks = np.random.choice(num_tasks, size=10, replace=False)
        total_meta_loss = 0.0
        
        for task_idx in batch_tasks:
            # Task adaptation loop
            task_model = WirelessNet()
            task_model.load_state_dict(model.state_dict())
            inner_optimizer = optim.Adam(task_model.parameters(), lr=inner_lr)
            
            for _ in range(inner_steps):
                preds = task_model(x_sup[task_idx])
                loss = loss_fn(preds, y_sup[task_idx])
                inner_optimizer.zero_grad()
                loss.backward()
                inner_optimizer.step()
            
            # Use query set to update meta-model
            query_preds = task_model(x_que[task_idx])
            query_loss = loss_fn(query_preds, y_que[task_idx])
            total_meta_loss += query_loss.item()
            
            task_model.zero_grad()
            query_loss.backward()
            
            # Distribute task gradients to master model
            for master_p, task_p in zip(model.parameters(), task_model.parameters()):
                if task_p.grad is not None:
                    if master_p.grad is None:
                        master_p.grad = task_p.grad.clone() / len(batch_tasks)
                    else:
                        master_p.grad += task_p.grad.clone() / len(batch_tasks)
        
        outer_optimizer.step()
        if (iteration + 1) % 10 == 0:
            print(f"MAML {iteration + 1}/{outer_iters} | Loss: {total_meta_loss/len(batch_tasks):.4f}")
    return model

# 5. Baseline (Random Initialization)
def train_baseline_on_task(x_support, y_support, x_query, y_query, train_steps=200, lr=0.01):
    """ Trains from scratch on one task. No shared prior knowledge. """
    model = WirelessNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    for _ in range(train_steps):
        preds = model(x_support)
        loss = loss_fn(preds, y_support)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        query_preds = model(x_query)
        query_loss = loss_fn(query_preds, y_query).item()
    return query_loss

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    
    print(f"Loading data from {data_dir}")
    x_sup, y_sup, x_que, y_que = load_tasks(os.path.join(data_dir, "train_data.npz"))
    os.makedirs(models_dir, exist_ok=True)
    
    # Verify baseline performance
    print("\n--- Verifying Baseline ---")
    for i in range(5):
        bl_loss = train_baseline_on_task(x_sup[i], y_sup[i], x_que[i], y_que[i])
        print(f"Task {i+1} Baseline Loss: {bl_loss:.4f}")
    
    # Train and save Reptile
    reptile_model = train_reptile(x_sup, y_sup, outer_iters=500, inner_steps=5, inner_lr=0.02, outer_lr=0.001)
    rep_path = os.path.join(models_dir, "reptile_master.pth")
    torch.save(reptile_model.state_dict(), rep_path)
    print(f"Reptile model saved: {rep_path}")
    
    # Train and save MAML
    maml_model = train_maml(x_sup, y_sup, x_que, y_que, outer_iters=500, inner_steps=5, inner_lr=0.02, outer_lr=0.001)
    maml_path = os.path.join(models_dir, "maml_master.pth")
    torch.save(maml_model.state_dict(), maml_path)
    print(f"MAML model saved: {maml_path}")
    
    print("\nSuccessfully trained and saved all models.")
