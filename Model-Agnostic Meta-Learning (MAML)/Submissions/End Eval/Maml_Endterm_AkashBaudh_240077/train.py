"""
Meta-learning training for wireless channel estimation using MAML.
Compares MAML (meta-learned initialization) against baseline (train from scratch).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import defaultdict


class ChannelEstimationNetwork(nn.Module):
    """
    Simple neural network for channel estimation.
    
    Architecture:
    Input(input_dim) -> Hidden(64) -> Hidden(32) -> Output(output_dim)
    
    This network learns to estimate channel coefficients from pilot signals.
    """
    
    def __init__(self, input_dim=4, output_dim=1, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def clone(self):
        """Create a deep copy of the network."""
        return deepcopy(self)


class MAMLTrainer:
    """
    Model-Agnostic Meta-Learning (MAML) trainer for channel estimation.
    
    MAML Algorithm (plain English):
    1. Start with shared weights θ
    2. For each task in batch:
       a) Take few gradient steps on support set -> get adapted weights θ'
       b) Compute loss on query set using θ'
    3. Update θ based on query losses (meta-gradient)
    4. Repeat
    
    This creates an initialization that adapts quickly to new tasks.
    """
    
    def __init__(self, model, device='cpu', inner_lr=0.01, outer_lr=0.001, 
                 inner_steps=5):
        """
        Initialize MAML trainer.
        
        Args:
            model: Neural network to train
            device: 'cpu' or 'cuda'
            inner_lr: Learning rate for inner loop (task adaptation)
            outer_lr: Learning rate for outer loop (meta-update)
            inner_steps: Number of gradient steps per task
        """
        self.model = model.to(device)
        self.device = device
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=outer_lr)
        self.loss_fn = nn.MSELoss()
    
    def inner_loop(self, x_support, y_support):
        """
        Adapt to a single task (inner loop).
        
        Takes few gradient steps on support set and returns adapted model.
        
        Args:
            x_support: Support inputs (n_support, input_dim)
            y_support: Support labels (n_support, output_dim)
            
        Returns:
            adapted_model: Network with updated weights
            support_loss: Loss on support set after adaptation
        """
        adapted_model = self.model.clone()
        adapted_optim = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        # Take inner_steps gradient steps on support set
        for _ in range(self.inner_steps):
            adapted_optim.zero_grad()
            support_pred = adapted_model(x_support)
            support_loss = self.loss_fn(support_pred, y_support)
            support_loss.backward()
            adapted_optim.step()
        
        # Final support loss
        with torch.no_grad():
            support_pred = adapted_model(x_support)
            final_support_loss = self.loss_fn(support_pred, y_support).item()
        
        return adapted_model, final_support_loss
    
    def outer_loop(self, tasks_batch):
        """
        Meta-update step (outer loop).
        
        For each task:
        1. Adapt on support set
        2. Compute loss on query set using adapted model
        3. Accumulate query gradient
        
        Then update meta-weights.
        
        Args:
            tasks_batch: List of task dicts with X_support, Y_support, X_query, Y_query
            
        Returns:
            dict with metrics (avg query loss, etc.)
        """
        self.meta_optimizer.zero_grad()
        
        total_query_loss = 0.0
        total_support_loss = 0.0
        num_tasks = len(tasks_batch)
        
        # Process each task
        for task in tasks_batch:
            # Prepare data
            x_support = torch.from_numpy(task['X_support']).float().to(self.device)
            y_support = torch.from_numpy(task['Y_support']).float().to(self.device)
            x_query = torch.from_numpy(task['X_query']).float().to(self.device)
            y_query = torch.from_numpy(task['Y_query']).float().to(self.device)
            
            # Inner loop: adapt to this task
            adapted_model, support_loss = self.inner_loop(x_support, y_support)
            total_support_loss += support_loss
            
            # Compute loss on query set with adapted model
            query_pred = adapted_model(x_query)
            query_loss = self.loss_fn(query_pred, y_query)
            
            # Accumulate gradients (will backprop through adaptation)
            query_loss.backward()
            total_query_loss += query_loss.item()
        
        # Meta-update: update base model weights
        self.meta_optimizer.step()
        
        return {
            'avg_query_loss': total_query_loss / num_tasks,
            'avg_support_loss': total_support_loss / num_tasks,
            'num_tasks': num_tasks
        }
    
    def evaluate(self, tasks_batch):
        """
        Evaluate model on a batch of tasks.
        
        Args:
            tasks_batch: List of task dicts
            
        Returns:
            dict with evaluation metrics
        """
        was_training = self.model.training
        self.model.eval()
        
        total_query_loss = 0.0
        total_support_loss = 0.0
        num_tasks = len(tasks_batch)
        
        for task in tasks_batch:
            x_support = torch.from_numpy(task['X_support']).float().to(self.device)
            y_support = torch.from_numpy(task['Y_support']).float().to(self.device)
            x_query = torch.from_numpy(task['X_query']).float().to(self.device)
            y_query = torch.from_numpy(task['Y_query']).float().to(self.device)
            
            # Adapt on support (need gradients for inner loop)
            adapted_model, support_loss = self.inner_loop(x_support, y_support)
            total_support_loss += support_loss
            
            # Evaluate on query
            with torch.no_grad():
                query_pred = adapted_model(x_query)
                query_loss = self.loss_fn(query_pred, y_query).item()
            total_query_loss += query_loss
        
        self.model.train(was_training)
        
        return {
            'avg_query_loss': total_query_loss / num_tasks,
            'avg_support_loss': total_support_loss / num_tasks,
            'num_tasks': num_tasks
        }


class BaselineTrainer:
    """
    Baseline: Train a fresh model from scratch on each task's support set.
    
    For comparison:
    - Train on task's support set for 200 steps
    - Evaluate on task's query set
    - Do this independently for each task
    
    Good baseline captures "how well can a model learn from scratch in limited data"
    """
    
    def __init__(self, input_dim=4, output_dim=1, device='cpu', 
                 steps_per_task=200):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.steps_per_task = steps_per_task
        self.loss_fn = nn.MSELoss()
    
    def train_on_task(self, x_support, y_support, x_query, y_query):
        """
        Train a single model from scratch and evaluate.
        
        Args:
            x_support, y_support: Support set
            x_query, y_query: Query set
            
        Returns:
            dict with support_loss, query_loss
        """
        # Create fresh model
        model = ChannelEstimationNetwork(self.input_dim, self.output_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        x_support_t = torch.from_numpy(x_support).float().to(self.device)
        y_support_t = torch.from_numpy(y_support).float().to(self.device)
        x_query_t = torch.from_numpy(x_query).float().to(self.device)
        y_query_t = torch.from_numpy(y_query).float().to(self.device)
        
        # Train on support set
        for _ in range(self.steps_per_task):
            optimizer.zero_grad()
            pred = model(x_support_t)
            loss = self.loss_fn(pred, y_support_t)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            support_pred = model(x_support_t)
            support_loss = self.loss_fn(support_pred, y_support_t).item()
            
            query_pred = model(x_query_t)
            query_loss = self.loss_fn(query_pred, y_query_t).item()
        
        return {
            'support_loss': support_loss,
            'query_loss': query_loss
        }
    
    def evaluate(self, tasks_batch):
        """
        Evaluate baseline on a batch of tasks.
        
        Args:
            tasks_batch: List of task dicts
            
        Returns:
            dict with average metrics
        """
        total_support_loss = 0.0
        total_query_loss = 0.0
        
        for task in tasks_batch:
            result = self.train_on_task(
                task['X_support'], task['Y_support'],
                task['X_query'], task['Y_query']
            )
            total_support_loss += result['support_loss']
            total_query_loss += result['query_loss']
        
        return {
            'avg_support_loss': total_support_loss / len(tasks_batch),
            'avg_query_loss': total_query_loss / len(tasks_batch),
            'num_tasks': len(tasks_batch)
        }


def load_dataset(data_path='results'):
    """
    Load training and test tasks from NPZ files.
    
    Args:
        data_path: Directory containing train_tasks.npz and test_tasks.npz
        
    Returns:
        (train_tasks, test_tasks): Lists of task dictionaries
    """
    data_dir = Path(data_path)
    
    train_data = np.load(data_dir / 'train_tasks.npz')
    test_data = np.load(data_dir / 'test_tasks.npz')
    
    # Convert stacked arrays back to list of tasks
    def stack_to_tasks(data):
        n_tasks = data['X_support'].shape[0]
        tasks = []
        for i in range(n_tasks):
            tasks.append({
                'X_support': data['X_support'][i],
                'Y_support': data['Y_support'][i],
                'X_query': data['X_query'][i],
                'Y_query': data['Y_query'][i],
                'snr': data['snr'][i],
                'num_paths': data['num_paths'][i],
                'noise_scale': data['noise_scale'][i],
            })
        return tasks
    
    train_tasks = stack_to_tasks(train_data)
    test_tasks = stack_to_tasks(test_data)
    
    return train_tasks, test_tasks


def plot_results(history, output_path='results/plot_loss.png'):
    """
    Plot training curves comparing MAML vs Baseline.
    
    Args:
        history: Dict with 'maml_query', 'maml_support', 'baseline_query', 'baseline_support'
        output_path: Where to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Query loss
    axes[0].plot(history['maml_query'], label='MAML Query', linewidth=2)
    axes[0].plot(history['baseline_query'], label='Baseline Query', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Query Loss (MSE)')
    axes[0].set_title('Query Loss: MAML vs Baseline')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Support loss
    axes[1].plot(history['maml_support'], label='MAML Support', linewidth=2)
    axes[1].plot(history['baseline_support'], label='Baseline Support', linewidth=2)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Support Loss (MSE)')
    axes[1].set_title('Support Loss: MAML vs Baseline')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    print(f"[OK] Plot saved: {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("MAML Training for Wireless Channel Estimation")
    print("=" * 70)
    print()
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_iterations = 50
    batch_size = 4  # Tasks per batch
    
    print(f"Device: {device}")
    print(f"Meta-learning iterations: {num_iterations}")
    print(f"Batch size: {batch_size} tasks")
    print()
    
    # Load dataset
    print("Loading dataset...")
    train_tasks, test_tasks = load_dataset('results')
    print(f"  Training tasks: {len(train_tasks)}")
    print(f"  Test tasks: {len(test_tasks)}")
    print()
    
    # Initialize models and trainers
    print("Initializing models...")
    base_model = ChannelEstimationNetwork(input_dim=4, output_dim=1)
    
    maml_trainer = MAMLTrainer(
        model=base_model,
        device=device,
        inner_lr=0.0001,  # Reduced to stable learning rate
        outer_lr=0.001,
        inner_steps=1    # Reduced from 5 to prevent adaptation overfitting
    )
    
    baseline_trainer = BaselineTrainer(device=device, steps_per_task=200)
    print()
    
    # Training loop
    print("Training...")
    print("-" * 70)
    
    history = defaultdict(list)
    
    for iteration in range(num_iterations):
        # Sample batch of training tasks
        batch_indices = np.random.choice(len(train_tasks), size=batch_size, replace=False)
        train_batch = [train_tasks[i] for i in batch_indices]
        
        # MAML meta-update
        maml_metrics = maml_trainer.outer_loop(train_batch)
        history['maml_query'].append(maml_metrics['avg_query_loss'])
        history['maml_support'].append(maml_metrics['avg_support_loss'])
        
        # Baseline evaluation (on same batch for fair comparison)
        baseline_metrics = baseline_trainer.evaluate(train_batch)
        history['baseline_query'].append(baseline_metrics['avg_query_loss'])
        history['baseline_support'].append(baseline_metrics['avg_support_loss'])
        
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"  MAML     - Query Loss: {maml_metrics['avg_query_loss']:.6f}, "
                  f"Support Loss: {maml_metrics['avg_support_loss']:.6f}")
            print(f"  Baseline - Query Loss: {baseline_metrics['avg_query_loss']:.6f}, "
                  f"Support Loss: {baseline_metrics['avg_support_loss']:.6f}")
            print()
    
    print("-" * 70)
    print()
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    maml_test = maml_trainer.evaluate(test_tasks)
    baseline_test = baseline_trainer.evaluate(test_tasks)
    
    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)
    print(f"MAML:")
    print(f"  Query Loss:   {maml_test['avg_query_loss']:.6f}")
    print(f"  Support Loss: {maml_test['avg_support_loss']:.6f}")
    print()
    print(f"Baseline (train from scratch):")
    print(f"  Query Loss:   {baseline_test['avg_query_loss']:.6f}")
    print(f"  Support Loss: {baseline_test['avg_support_loss']:.6f}")
    print()
    improvement = (baseline_test['avg_query_loss'] - maml_test['avg_query_loss']) / baseline_test['avg_query_loss'] * 100
    print(f"MAML improvement: {improvement:.1f}%")
    print("=" * 70)
    print()
    
    # Plot results
    plot_results(history)
    
    # Save model
    torch.save(maml_trainer.model.state_dict(), 'results/maml_model.pt')
    print("[OK] Model saved: results/maml_model.pt")
    print()


if __name__ == '__main__':
    main()
