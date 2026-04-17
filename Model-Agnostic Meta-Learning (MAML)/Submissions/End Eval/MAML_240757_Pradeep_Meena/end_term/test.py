import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt

# Network structure must match train.py exactly
class SignalClassifier(nn.Module):
    def __init__(self):
        super(SignalClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3) 
        )

    def forward(self, x):
        return self.net(x)

def functional_forward(x, weights):
    x = F.linear(x, weights['net.0.weight'], weights['net.0.bias'])
    x = F.relu(x)
    x = F.linear(x, weights['net.2.weight'], weights['net.2.bias'])
    x = F.relu(x)
    x = F.linear(x, weights['net.4.weight'], weights['net.4.bias'])
    return x

def test_maml():
    print("Loading test data and model...")
    data_path = os.path.join(os.getcwd(), 'data', 'meta_test_tasks.npz')
    data = np.load(data_path)
    
    X_supp = torch.tensor(data['X_support']) 
    Y_supp = torch.tensor(data['Y_support'])
    X_query = torch.tensor(data['X_query'])
    Y_query = torch.tensor(data['Y_query'])
    
    num_tasks = X_supp.shape[0]
    
    # Load Trained MAML Model
    meta_model = SignalClassifier()
    model_path = os.path.join(os.getcwd(), 'models', 'meta_model.pth')
    meta_model.load_state_dict(torch.load(model_path, weights_only=True))
    
    inner_lr = 0.01
    adaptation_steps = 5
    baseline_steps = 200
    
    # Arrays to track error at each step (0 through 5)
    maml_errors_per_step = np.zeros(adaptation_steps + 1)
    baseline_errors_per_step = np.zeros(adaptation_steps + 1)
    final_baseline_error = 0.0
    
    print("Evaluating on 20 test tasks...")
    
    for task_idx in range(num_tasks):
        x_s, y_s = X_supp[task_idx], Y_supp[task_idx]
        x_q, y_q = X_query[task_idx], Y_query[task_idx]
        
        # -----------------------------------------
        # 1. MAML EVALUATION (Adapting for 5 steps)
        # -----------------------------------------
        fast_weights = {name: param.clone() for name, param in meta_model.named_parameters()}
        
        for step in range(adaptation_steps + 1):
            # Evaluate on Query Set (Calculate Error = 1 - Accuracy)
            with torch.no_grad():
                q_preds = functional_forward(x_q, fast_weights)
                acc = (q_preds.argmax(dim=1) == y_q).float().mean().item()
                maml_errors_per_step[step] += (1.0 - acc)
                
            # Perform 1 Gradient Step on Support Set
            if step < adaptation_steps:
                s_preds = functional_forward(x_s, fast_weights)
                loss = F.cross_entropy(s_preds, y_s)
                grads = torch.autograd.grad(loss, fast_weights.values())
                fast_weights = {name: param - inner_lr * grad 
                                for ((name, param), grad) in zip(fast_weights.items(), grads)}
                                
        # -----------------------------------------
        # 2. BASELINE EVALUATION (Training from Scratch)
        # -----------------------------------------
        baseline_model = SignalClassifier() 
        baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=0.01)
        
        for step in range(baseline_steps + 1):
            # Track early steps for the comparison plot
            if step <= adaptation_steps:
                baseline_model.eval()
                with torch.no_grad():
                    q_preds = baseline_model(x_q)
                    acc = (q_preds.argmax(dim=1) == y_q).float().mean().item()
                    baseline_errors_per_step[step] += (1.0 - acc)
            
            # Train on support set
            if step < baseline_steps:
                baseline_model.train()
                baseline_optimizer.zero_grad()
                s_preds = baseline_model(x_s)
                loss = F.cross_entropy(s_preds, y_s)
                loss.backward()
                baseline_optimizer.step()
                
        # Record final 200-step baseline performance
        baseline_model.eval()
        with torch.no_grad():
            q_preds = baseline_model(x_q)
            acc = (q_preds.argmax(dim=1) == y_q).float().mean().item()
            final_baseline_error += (1.0 - acc)

    # Average the errors over all 20 tasks
    maml_errors_per_step /= num_tasks
    baseline_errors_per_step /= num_tasks
    final_baseline_error /= num_tasks
    final_maml_error = maml_errors_per_step[-1]
    
    print("\n" + "="*40)
    print("RESULTS TABLE NUMBERS (For README.md)")
    print("="*40)
    print(f"MAML Error (5-shots/steps):     {final_maml_error:.4f}")
    print(f"Baseline Error (200-shots/steps): {final_baseline_error:.4f}")
    print("="*40 + "\n")
    
    # Generate Plot 2: Comparison Plot
    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    steps = np.arange(adaptation_steps + 1)
    
    plt.plot(steps, maml_errors_per_step, marker='o', linewidth=2, label='MAML (Meta-Learned)')
    plt.plot(steps, baseline_errors_per_step, marker='x', linestyle='--', color='red', label='Baseline (From Scratch)')
    
    plt.title('MAML vs Baseline Comparison (Signal Classification)')
    plt.xlabel('Number of Adaptation Steps (Shots)')
    plt.ylabel('Classification Error Rate (Lower is Better)')
    plt.xticks(steps)
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(results_dir, 'plot_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot 2 saved correctly in {plot_path}")

if __name__ == "__main__":
    test_maml()