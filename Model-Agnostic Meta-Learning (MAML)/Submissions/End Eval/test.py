import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.subplots as plt_subs
import matplotlib.pyplot as plt

# Import architectures and utils from our training script
from train import IndoorLocNet, fetch_data_tensors, train_from_scratch

def quick_adapt_eval(meta_model, x_shot, y_shot, x_eval, y_eval, steps=5, lr=0.02):
    """ Takes a meta-trained model, finetunes it briefly on the shots, and checks the eval set. """
    local_net = IndoorLocNet()
    local_net.load_state_dict(meta_model.state_dict())
    opt = optim.Adam(local_net.parameters(), lr=lr)
    mse = nn.MSELoss()
    
    error_curve = []
    
    # Record zero-shot performance (before any room-specific training)
    with torch.no_grad():
        initial_preds = local_net(x_eval)
        error_curve.append(mse(initial_preds, y_eval).item())
    
    # Fast adaptation loop
    for _ in range(steps):
        preds = local_net(x_shot)
        loss = mse(preds, y_shot)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # Track improvement on the eval set after each gradient step
        with torch.no_grad():
            step_eval_preds = local_net(x_eval)
            error_curve.append(mse(step_eval_preds, y_eval).item())
            
    return error_curve[-1], error_curve

def eval_scratch_model(x_shot, y_shot, x_eval, y_eval, step_checkpoints=[0, 1, 2, 3, 4, 5]):
    """ Trains a blank model from scratch to see how it compares to meta-learning. """
    mse = nn.MSELoss()
    errors = []
    
    for n_steps in step_checkpoints:
        blank_net = IndoorLocNet()
        if n_steps > 0:
            opt = optim.Adam(blank_net.parameters(), lr=0.01)
            for _ in range(n_steps):
                preds = blank_net(x_shot)
                loss = mse(preds, y_shot)
                opt.zero_grad()
                loss.backward()
                opt.step()
        
        with torch.no_grad():
            eval_preds = blank_net(x_eval)
            errors.append(mse(eval_preds, y_eval).item())
            
    return errors

if __name__ == "__main__":
    # Lock seeds for consistent chart generation
    torch.manual_seed(42)
    np.random.seed(42)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    model_dir = os.path.join(base_dir, "models")
    viz_dir = os.path.join(base_dir, "results")
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"Locating test datasets...")
    test_suites = {
        '5-shot':  fetch_data_tensors(os.path.join(data_dir, "test_data_5shot.npz")),
        '10-shot': fetch_data_tensors(os.path.join(data_dir, "test_data.npz")),
        '20-shot': fetch_data_tensors(os.path.join(data_dir, "test_data_20shot.npz")),
    }
    
    # Load up the weights we trained in train.py
    print("Loading pre-trained meta models...")
    rep_net = IndoorLocNet()
    rep_net.load_state_dict(torch.load(os.path.join(model_dir, "reptile_weights.pth"), weights_only=True))
    
    maml_net = IndoorLocNet()
    maml_net.load_state_dict(torch.load(os.path.join(model_dir, "maml_weights.pth"), weights_only=True))
    
    adapt_limit = 5
    x_axis_steps = list(range(adapt_limit + 1))
    summary_metrics = {}
    
    # Variables to hold plotting data
    rep_curves, maml_curves, scratch_curves = [], [], []
    
    for scenario_name, (x_s, y_s, x_e, y_e) in test_suites.items():
        n_tasks = x_s.shape[0]
        rep_final_errs, maml_final_errs, scratch_final_errs = [], [], []
        
        print(f"\n[ Evaluating {scenario_name} | {n_tasks} Rooms ]")
        print(f"{'Room ID':<8} {'Scratch MSE':>12} {'Reptile MSE':>12} {'MAML MSE':>12}")
        print("-" * 48)
        
        for task_id in range(n_tasks):
            # Evaluate Reptile
            rep_end, rep_traj = quick_adapt_eval(rep_net, x_s[task_id], y_s[task_id], x_e[task_id], y_e[task_id])
            rep_final_errs.append(rep_end)
            
            # Evaluate MAML
            maml_end, maml_traj = quick_adapt_eval(maml_net, x_s[task_id], y_s[task_id], x_e[task_id], y_e[task_id])
            maml_final_errs.append(maml_end)
            
            # Evaluate Standard Training
            scratch_err = train_from_scratch(x_s[task_id], y_s[task_id], x_e[task_id], y_e[task_id])
            scratch_final_errs.append(scratch_err)
            
            # Collect curve data specifically for the 10-shot scenario to plot later
            if scenario_name == '10-shot':
                rep_curves.append(rep_traj)
                maml_curves.append(maml_traj)
                scratch_curves.append(eval_scratch_model(x_s[task_id], y_s[task_id], x_e[task_id], y_e[task_id], x_axis_steps))
            
            print(f"  {task_id+1:<6} {scratch_err:>12.4f} {rep_end:>12.4f} {maml_end:>12.4f}")
            
        summary_metrics[scenario_name] = {
            'scratch': np.mean(scratch_final_errs),
            'reptile': np.mean(rep_final_errs),
            'maml': np.mean(maml_final_errs),
        }
        
    # Print the final leaderboard
    print(f"\n{'Algorithm':<20} {'5-shot MSE':>12} {'10-shot MSE':>12} {'20-shot MSE':>12}")
    print("-" * 60)
    for method in ['scratch', 'maml', 'reptile']:
        display_name = "Trained from Scratch" if method == 'scratch' else method.upper()
        print(f"{display_name:<20} {summary_metrics['5-shot'][method]:>12.4f} {summary_metrics['10-shot'][method]:>12.4f} {summary_metrics['20-shot'][method]:>12.4f}")
        
    # --- Visualization Generation ---
    print("\nRendering performance charts...")
    
    avg_scratch = np.mean(scratch_curves, axis=0)
    avg_rep = np.mean(rep_curves, axis=0)
    avg_maml = np.mean(maml_curves, axis=0)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x_axis_steps, avg_scratch, label='Scratch Base', marker='o', linestyle='--')
    ax.plot(x_axis_steps, avg_rep, label='Reptile', marker='s')
    ax.plot(x_axis_steps, avg_maml, label='MAML', marker='^')
    
    ax.set_title('Fast Adaptation on Unseen Rooms (10-Shot)')
    ax.set_xlabel('Gradient Steps on Support Set')
    ax.set_ylabel('Mean Squared Error on Target Set')
    ax.legend()
    fig.tight_layout()
    
    plot_out = os.path.join(viz_dir, 'adaptation_curves.png')
    fig.savefig(plot_out, dpi=150)
    print(f"Chart saved successfully -> {plot_out}")