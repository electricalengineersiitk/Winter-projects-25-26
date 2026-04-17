"""
Meta-learning implementation for wireless channel estimation.
Evaluates rapid adaptation capability versus a standard train-from-scratch approach.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt


class ChannelEstModel(nn.Module):
    """
    Neural network tailored for channel estimation from pilot signals.
    """
    def __init__(self, in_features=4, out_features=1, hidden_nodes=64):
        super().__init__()
        # Explicit layer definitions instead of nn.Sequential
        self.fc1 = nn.Linear(in_features, hidden_nodes)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_nodes, 32)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(32, out_features)
    
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)
    
    def copy_model(self):
        """Generates a detached clone of the current network state."""
        return deepcopy(self)


class MetaLearner:
    """
    Model-Agnostic Meta-Learning (MAML) logic.
    Optimizes a global initialization that can be fine-tuned rapidly.
    """
    def __init__(self, network, dev='cpu', alpha=0.01, beta=0.001, steps=5):
        self.net = network.to(dev)
        self.dev = dev
        self.alpha = alpha  # Task-specific learning rate (inner loop)
        self.beta = beta    # Meta-learning rate (outer loop)
        self.steps = steps
        
        self.meta_opt = optim.Adam(self.net.parameters(), lr=self.beta)
        self.mse = nn.MSELoss()
    
    def fast_adapt(self, supp_x, supp_y):
        """Inner loop: adapts the global model to a specific task."""
        task_model = self.net.copy_model()
        task_opt = optim.SGD(task_model.parameters(), lr=self.alpha)
        
        for _ in range(self.steps):
            task_opt.zero_grad()
            preds = task_model(supp_x)
            loss = self.mse(preds, supp_y)
            loss.backward()
            task_opt.step()
            
        with torch.no_grad():
            final_loss = self.mse(task_model(supp_x), supp_y).item()
            
        return task_model, final_loss
    
    def meta_step(self, task_batch):
        """Outer loop: computes meta-gradients and updates global weights."""
        self.meta_opt.zero_grad()
        
        q_loss_total = 0.0
        s_loss_total = 0.0
        batch_sz = len(task_batch)
        
        for task in task_batch:
            # Move data to target device
            supp_x = torch.tensor(task['X_support'], dtype=torch.float32, device=self.dev)
            supp_y = torch.tensor(task['Y_support'], dtype=torch.float32, device=self.dev)
            query_x = torch.tensor(task['X_query'], dtype=torch.float32, device=self.dev)
            query_y = torch.tensor(task['Y_query'], dtype=torch.float32, device=self.dev)
            
            # 1. Fast adaptation on support set
            task_model, s_loss = self.fast_adapt(supp_x, supp_y)
            s_loss_total += s_loss
            
            # 2. Evaluate on query set
            q_preds = task_model(query_x)
            q_loss = self.mse(q_preds, query_y)
            
            # 3. Accumulate meta-gradients
            q_loss.backward()
            q_loss_total += q_loss.item()
        
        # Apply the accumulated meta-update
        self.meta_opt.step()
        
        return {
            'q_loss_avg': q_loss_total / batch_sz,
            's_loss_avg': s_loss_total / batch_sz
        }
    
    def run_eval(self, task_batch):
        """Evaluates model performance without updating meta-weights."""
        training_state = self.net.training
        self.net.eval()
        
        q_loss_total = 0.0
        s_loss_total = 0.0
        batch_sz = len(task_batch)
        
        for task in task_batch:
            supp_x = torch.tensor(task['X_support'], dtype=torch.float32, device=self.dev)
            supp_y = torch.tensor(task['Y_support'], dtype=torch.float32, device=self.dev)
            query_x = torch.tensor(task['X_query'], dtype=torch.float32, device=self.dev)
            query_y = torch.tensor(task['Y_query'], dtype=torch.float32, device=self.dev)
            
            # Adapt requires gradients, even during evaluation
            task_model, s_loss = self.fast_adapt(supp_x, supp_y)
            s_loss_total += s_loss
            
            with torch.no_grad():
                q_loss = self.mse(task_model(query_x), query_y).item()
                q_loss_total += q_loss
        
        self.net.train(training_state)
        
        return {
            'q_loss_avg': q_loss_total / batch_sz,
            's_loss_avg': s_loss_total / batch_sz
        }


class ScratchTrainer:
    """Trains individual models from scratch to serve as a performance baseline."""
    def __init__(self, in_features=4, out_features=1, dev='cpu', steps=200):
        self.in_features = in_features
        self.out_features = out_features
        self.dev = dev
        self.steps = steps
        self.mse = nn.MSELoss()
    
    def train_single(self, supp_x, supp_y, query_x, query_y):
        model = ChannelEstModel(self.in_features, self.out_features).to(self.dev)
        opt = optim.Adam(model.parameters(), lr=0.001)
        
        s_x = torch.tensor(supp_x, dtype=torch.float32, device=self.dev)
        s_y = torch.tensor(supp_y, dtype=torch.float32, device=self.dev)
        q_x = torch.tensor(query_x, dtype=torch.float32, device=self.dev)
        q_y = torch.tensor(query_y, dtype=torch.float32, device=self.dev)
        
        for _ in range(self.steps):
            opt.zero_grad()
            loss = self.mse(model(s_x), s_y)
            loss.backward()
            opt.step()
            
        model.eval()
        with torch.no_grad():
            s_err = self.mse(model(s_x), s_y).item()
            q_err = self.mse(model(q_x), q_y).item()
            
        return s_err, q_err
    
    def run_eval(self, task_batch):
        s_loss_total = 0.0
        q_loss_total = 0.0
        batch_sz = len(task_batch)
        
        for task in task_batch:
            s_err, q_err = self.train_single(
                task['X_support'], task['Y_support'],
                task['X_query'], task['Y_query']
            )
            s_loss_total += s_err
            q_loss_total += q_err
            
        return {
            's_loss_avg': s_loss_total / batch_sz,
            'q_loss_avg': q_loss_total / batch_sz
        }


def extract_tasks(folder='results'):
    """Parses NPZ files into lists of task dictionaries."""
    target_dir = Path(folder)
    train_raw = np.load(target_dir / 'train_tasks.npz')
    test_raw = np.load(target_dir / 'test_tasks.npz')
    
    def formatter(raw_data):
        return [{
            'X_support': raw_data['X_support'][i],
            'Y_support': raw_data['Y_support'][i],
            'X_query': raw_data['X_query'][i],
            'Y_query': raw_data['Y_query'][i],
            'snr': raw_data['snr'][i],
            'num_paths': raw_data['num_paths'][i],
            'noise_scale': raw_data['noise_scale'][i],
        } for i in range(raw_data['X_support'].shape[0])]
        
    return formatter(train_raw), formatter(test_raw)


def render_graphs(metrics, save_dst='results/plot_loss.png'):
    """Generates performance comparison charts."""
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(metrics['maml_q'], label='Meta-Learned', color='indigo')
    ax1.plot(metrics['base_q'], label='Scratch Baseline', color='teal', linestyle='--')
    ax1.set(title='Query Loss Comparison', xlabel='Epoch', ylabel='MSE')
    ax1.legend()
    
    ax2.plot(metrics['maml_s'], label='Meta-Learned', color='indigo')
    ax2.plot(metrics['base_s'], label='Scratch Baseline', color='teal', linestyle='--')
    ax2.set(title='Support Loss Comparison', xlabel='Epoch', ylabel='MSE')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_dst)
    plt.close()


def main():
    print("\n--- Wireless Channel Meta-Learning Initialization ---\n")
    
    # Setup
    compute_dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 50
    b_size = 4
    
    print(f"Target Device: {compute_dev.upper()}")
    print(f"Epochs: {epochs} | Tasks per batch: {b_size}\n")
    
    train_data, test_data = extract_tasks('results')
    print(f"Loaded {len(train_data)} train sets and {len(test_data)} test sets.\n")
    
    # Initialize
    network = ChannelEstModel(in_features=4, out_features=1)
    meta_agent = MetaLearner(network, dev=compute_dev, alpha=0.0001, beta=0.001, steps=1)
    baseline_agent = ScratchTrainer(dev=compute_dev, steps=200)
    
    tracker = {'maml_q': [], 'maml_s': [], 'base_q': [], 'base_s': []}
    
    print("Initiating Meta-Training Sequence...")
    for ep in range(epochs):
        # Sample random tasks
        idx = np.random.choice(len(train_data), size=b_size, replace=False)
        batch = [train_data[i] for i in idx]
        
        # MAML pass
        m_res = meta_agent.meta_step(batch)
        tracker['maml_q'].append(m_res['q_loss_avg'])
        tracker['maml_s'].append(m_res['s_loss_avg'])
        
        # Baseline pass
        b_res = baseline_agent.run_eval(batch)
        tracker['base_q'].append(b_res['q_loss_avg'])
        tracker['base_s'].append(b_res['s_loss_avg'])
        
        if (ep + 1) % 10 == 0:
            print(f"[Epoch {ep + 1:02d}/{epochs}] "
                  f"MAML Query: {m_res['q_loss_avg']:.5f} | "
                  f"Base Query: {b_res['q_loss_avg']:.5f}")
            
    print("\nEvaluating against isolated test sets...")
    final_maml = meta_agent.run_eval(test_data)
    final_base = baseline_agent.run_eval(test_data)
    
    print("\n--- Final Metrics ---")
    print(f"Meta-Learning (MAML) -> Query MSE: {final_maml['q_loss_avg']:.5f}")
    print(f"Train-from-Scratch   -> Query MSE: {final_base['q_loss_avg']:.5f}")
    
    gain = ((final_base['q_loss_avg'] - final_maml['q_loss_avg']) / final_base['q_loss_avg']) * 100
    print(f"\nTotal Performance Gain: {gain:.2f}%")
    
    render_graphs(tracker)
    torch.save(meta_agent.net.state_dict(), 'results/maml_model.pt')
    print("\nVisuals and model weights stored in 'results' directory.")

if __name__ == '__main__':
    main()
