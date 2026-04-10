import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- Model Definition ---
class IndoorLocNet(nn.Module):
    def __init__(self):
        super(IndoorLocNet, self).__init__()
        # Standard 3-layer MLP. Taking 4 RSSI signals -> (x,y,z) coords
        self.backbone = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3) 
        )
        
    def forward(self, x):
        return self.backbone(x)

# --- Utils ---
def fetch_data_tensors(filepath):
    """Loads npz files and casts them straight to PyTorch tensors."""
    raw = np.load(filepath)
    return (
        torch.tensor(raw['x_support'], dtype=torch.float32),
        torch.tensor(raw['y_support'], dtype=torch.float32),
        torch.tensor(raw['x_query'], dtype=torch.float32),
        torch.tensor(raw['y_query'], dtype=torch.float32)
    )

# --- Meta Learning Algorithms ---

def run_reptile(x_train, y_train, meta_epochs=500, adapt_steps=5, fast_lr=0.01, meta_lr=0.001):
    print("\n[ Initiating Reptile Meta-Training ]")
    master_model = IndoorLocNet()
    meta_opt = optim.Adam(master_model.parameters(), lr=meta_lr)
    mse = nn.MSELoss()
    total_envs = x_train.shape[0]
    
    for epoch in range(meta_epochs):
        meta_opt.zero_grad()
        
        # Grab a random subset of rooms for this meta-batch
        env_batch = np.random.choice(total_envs, size=10, replace=False)
        
        # Keep track of where we started and accumulate the adapted weights
        weights_before = [p.clone().detach() for p in master_model.parameters()]
        accumulated_weights = [torch.zeros_like(p) for p in master_model.parameters()]
        running_loss = 0.0
        
        for env_idx in env_batch:
            # Create a temporary clone for task-specific adaptation
            temp_net = IndoorLocNet()
            temp_net.load_state_dict(master_model.state_dict())
            local_opt = optim.Adam(temp_net.parameters(), lr=fast_lr)
            
            # Fast adapt on the support set
            for _ in range(adapt_steps):
                preds = temp_net(x_train[env_idx])
                loss = mse(preds, y_train[env_idx])
                local_opt.zero_grad()
                loss.backward()
                local_opt.step()
                
            running_loss += loss.item()
            
            # Tally up the final weights after adaptation
            for i, p in enumerate(temp_net.parameters()):
                accumulated_weights[i] += p.data
                
        # Reptile update rule: Move master weights toward the average of the adapted weights
        for i, p in enumerate(master_model.parameters()):
            avg_adapted = accumulated_weights[i] / len(env_batch)
            p.grad = (weights_before[i] - avg_adapted)
            
        meta_opt.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Reptile Epoch {epoch + 1}/{meta_epochs} | Avg Task Loss: {running_loss/len(env_batch):.4f}")
            
    return master_model


def run_maml(x_train, y_train, x_test, y_test, meta_epochs=500, adapt_steps=5, fast_lr=0.01, meta_lr=0.001):
    print("\n[ Initiating FOMAML Meta-Training ]")
    master_model = IndoorLocNet()
    meta_opt = optim.Adam(master_model.parameters(), lr=meta_lr)
    mse = nn.MSELoss()
    total_envs = x_train.shape[0]
    
    for epoch in range(meta_epochs):
        meta_opt.zero_grad()
        env_batch = np.random.choice(total_envs, size=10, replace=False)
        running_meta_loss = 0.0
        
        for env_idx in env_batch:
            # Clone model for inner loop
            temp_net = IndoorLocNet()
            temp_net.load_state_dict(master_model.state_dict())
            local_opt = optim.Adam(temp_net.parameters(), lr=fast_lr)
            
            # Inner loop: adapt on support data
            for _ in range(adapt_steps):
                preds = temp_net(x_train[env_idx])
                loss = mse(preds, y_train[env_idx])
                local_opt.zero_grad()
                loss.backward()
                local_opt.step()
            
            # Outer loop: evaluate on query data and compute gradients
            query_preds = temp_net(x_test[env_idx])
            query_loss = mse(query_preds, y_test[env_idx])
            running_meta_loss += query_loss.item()
            
            temp_net.zero_grad()
            query_loss.backward()
            
            # First-Order MAML approximation: pass task gradients directly to master model
            for m_param, t_param in zip(master_model.parameters(), temp_net.parameters()):
                if t_param.grad is not None:
                    if m_param.grad is None:
                        m_param.grad = t_param.grad.clone() / len(env_batch)
                    else:
                        m_param.grad += t_param.grad.clone() / len(env_batch)
        
        meta_opt.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"MAML Epoch {epoch + 1}/{meta_epochs} | Meta-Loss: {running_meta_loss/len(env_batch):.4f}")
            
    return master_model


def train_from_scratch(x_sup, y_sup, x_que, y_que, iters=200, lr=0.01):
    """ Baseline comparison: Train a fresh network on a new room with zero prior knowledge. """
    net = IndoorLocNet()
    opt = optim.Adam(net.parameters(), lr=lr)
    mse = nn.MSELoss()
    
    for _ in range(iters):
        preds = net(x_sup)
        loss = mse(preds, y_sup)
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    with torch.no_grad():
        target_preds = net(x_que)
        target_loss = mse(target_preds, y_que).item()
        
    return target_loss


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "models")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading datasets from {data_dir}...")
    x_s, y_s, x_q, y_q = fetch_data_tensors(os.path.join(data_dir, "train_data.npz"))
    
    # Sanity check baseline
    print("\n--- Testing Scratch Baseline ---")
    for i in range(3):
        err = train_from_scratch(x_s[i], y_s[i], x_q[i], y_q[i])
        print(f"Room {i+1} Baseline Error (MSE): {err:.4f}")
    
    # Train & Export Reptile
    rep_net = run_reptile(x_s, y_s, meta_epochs=500, adapt_steps=5, fast_lr=0.02, meta_lr=0.001)
    rep_export_path = os.path.join(out_dir, "reptile_weights.pth")
    torch.save(rep_net.state_dict(), rep_export_path)
    
    # Train & Export MAML
    maml_net = run_maml(x_s, y_s, x_q, y_q, meta_epochs=500, adapt_steps=5, fast_lr=0.02, meta_lr=0.001)
    maml_export_path = os.path.join(out_dir, "maml_weights.pth")
    torch.save(maml_net.state_dict(), maml_export_path)
    
    print(f"\nTraining complete. Weights saved to {out_dir}/")