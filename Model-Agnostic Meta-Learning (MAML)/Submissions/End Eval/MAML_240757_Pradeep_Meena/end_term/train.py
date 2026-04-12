import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt

# ✓ DO: Keep your neural network small - 3 layers, 64 neurons each
class SignalClassifier(nn.Module):
    def __init__(self):
        super(SignalClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # 3 classes: BPSK, QPSK, 16QAM
        )

    def forward(self, x):
        return self.net(x)

def functional_forward(x, weights):
    """
    Manual forward pass. 
    This allows us to track gradients through the inner loop weight updates.
    """
    x = F.linear(x, weights['net.0.weight'], weights['net.0.bias'])
    x = F.relu(x)
    x = F.linear(x, weights['net.2.weight'], weights['net.2.bias'])
    x = F.relu(x)
    x = F.linear(x, weights['net.4.weight'], weights['net.4.bias'])
    return x

def train_maml():
    print("Loading data...")
    data_path = os.path.join(os.getcwd(), 'data', 'meta_train_tasks.npz')
    data = np.load(data_path)
    
    X_supp = torch.tensor(data['X_support']) 
    Y_supp = torch.tensor(data['Y_support'])
    X_query = torch.tensor(data['X_query'])
    Y_query = torch.tensor(data['Y_query'])
    
    num_tasks = X_supp.shape[0]
    
    # Step 1: Start with a shared set of weights (Theta)
    meta_model = SignalClassifier()
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.001)
    
    # Hyperparameters
    inner_lr = 0.01
    inner_steps = 5  # DO: Use 3 to 5 inner loop steps
    meta_epochs = 1000  # INCREASED to let MAML fully converge
    tasks_per_batch = 10
    
    history_loss = []
    print("Starting MAML Meta-Training...")
    
    for epoch in range(meta_epochs):
        meta_optimizer.zero_grad()
        meta_loss = 0.0
        
        # Randomly sample a batch of tasks
        task_indices = np.random.choice(num_tasks, tasks_per_batch, replace=False)
        
        for task_idx in task_indices:
            x_s, y_s = X_supp[task_idx], Y_supp[task_idx]
            x_q, y_q = X_query[task_idx], Y_query[task_idx]
            
            # Extract current meta-weights as a dictionary
            fast_weights = {name: param for name, param in meta_model.named_parameters()}
            
            # Step 2: Inner loop - Adapt on the support set
            for _ in range(inner_steps):
                preds = functional_forward(x_s, fast_weights)
                loss = F.cross_entropy(preds, y_s)
                
                # Compute gradients w.r.t fast_weights 
                # create_graph=True is what makes this true MAML (Second-order gradients!)
                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
                
                # Manual inner-loop gradient descent update
                fast_weights = {name: param - inner_lr * grad 
                                for ((name, param), grad) in zip(fast_weights.items(), grads)}
                
            # Step 3: Evaluate adapted weights on the query set
            q_preds = functional_forward(x_q, fast_weights)
            q_loss = F.cross_entropy(q_preds, y_q)
            meta_loss += q_loss
            
        # Average meta-loss over the batch
        meta_loss = meta_loss / tasks_per_batch
        
        # Step 4: Update Theta based on the query set performance
        meta_loss.backward()
        meta_optimizer.step()
        
        history_loss.append(meta_loss.item())
        
        # ✓ DO: Print loss every 10 iterations
        if (epoch + 1) % 10 == 0:
            print(f"Iteration {epoch + 1}/{meta_epochs} - Meta Loss: {meta_loss.item():.4f}")
            
    # Save Model Weights safely
    model_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(meta_model.state_dict(), os.path.join(model_dir, 'meta_model.pth'))
    print("Training complete. Model saved!")
    
    # Generate Plot 1: Training Loss Curve
    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    plt.plot(history_loss, color='blue', label='MAML Meta-Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Meta-Training Iteration')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'plot_loss.png'))
    plt.close()
    print("Plot 1 saved correctly in results/plot_loss.png")

if __name__ == "__main__":
    train_maml()