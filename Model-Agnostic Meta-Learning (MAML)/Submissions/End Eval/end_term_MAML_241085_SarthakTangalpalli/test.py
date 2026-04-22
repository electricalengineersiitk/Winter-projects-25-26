import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Import model and utilities from train.py
from train import WirelessNet, load_tasks, train_baseline_on_task

def adapt_and_evaluate(model, x_support, y_support, x_query, y_query, inner_steps=5, inner_lr=0.02):
    """ Adapts a pre-trained model on the support set and evaluates on the query set. """
    adapted_model = WirelessNet()
    adapted_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(adapted_model.parameters(), lr=inner_lr)
    loss_fn = nn.MSELoss()
    
    losses_per_step = []
    
    # Error before adaptation (0-shot)
    with torch.no_grad():
        preds = adapted_model(x_query)
        initial_loss = loss_fn(preds, y_query).item()
    losses_per_step.append(initial_loss)
    
    # Adaptation steps
    for step in range(inner_steps):
        preds = adapted_model(x_support)
        loss = loss_fn(preds, y_support)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            query_preds = adapted_model(x_query)
            query_loss = loss_fn(query_preds, y_query).item()
        losses_per_step.append(query_loss)
    
    return losses_per_step[-1], losses_per_step

def evaluate_baseline_sweep(x_support, y_support, x_query, y_query, step_counts=[0, 1, 2, 3, 4, 5]):
    """ Trains a fresh model from scratch for different step counts. """
    loss_fn = nn.MSELoss()
    losses_per_step = []
    
    for steps in step_counts:
        model = WirelessNet()
        if steps > 0:
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            for _ in range(steps):
                preds = model(x_support)
                loss = loss_fn(preds, y_support)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        with torch.no_grad():
            query_preds = model(x_query)
            query_loss = loss_fn(query_preds, y_query).item()
        losses_per_step.append(query_loss)
    
    return losses_per_step

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    
    print(f"Loading test data from {data_dir}")
    datasets = {
        '5-shot':  load_tasks(os.path.join(data_dir, "test_data_5shot.npz")),
        '10-shot': load_tasks(os.path.join(data_dir, "test_data.npz")),
        '20-shot': load_tasks(os.path.join(data_dir, "test_data_20shot.npz")),
    }
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Load Meta-Learned Weights
    print(f"Loading models from {models_dir}")
    reptile_model = WirelessNet()
    reptile_model.load_state_dict(torch.load(os.path.join(models_dir, "reptile_master.pth"), weights_only=True))
    
    maml_model = WirelessNet()
    maml_model.load_state_dict(torch.load(os.path.join(models_dir, "maml_master.pth"), weights_only=True))
    
    # Start evaluation sweep
    adaptation_steps = 5
    step_range = list(range(adaptation_steps + 1))
    results_table = {}
    
    all_reptile_steps, all_maml_steps, all_baseline_steps = [], [], []
    
    for shot_name, (x_sup, y_sup, x_que, y_que) in datasets.items():
        num_test_tasks = x_sup.shape[0]
        reptile_errors, maml_errors, baseline_errors = [], [], []
        
        print(f"\nEvaluating {shot_name} ({num_test_tasks} tasks)")
        print(f"{'Task':<6} {'Baseline':>12} {'Reptile':>12} {'MAML':>12}")
        print("-" * 48)
        
        for t in range(num_test_tasks):
            rep_final, rep_steps = adapt_and_evaluate(reptile_model, x_sup[t], y_sup[t], x_que[t], y_que[t])
            reptile_errors.append(rep_final)
            
            maml_final, maml_steps = adapt_and_evaluate(maml_model, x_sup[t], y_sup[t], x_que[t], y_que[t])
            maml_errors.append(maml_final)
            
            bl_loss = train_baseline_on_task(x_sup[t], y_sup[t], x_que[t], y_que[t])
            baseline_errors.append(bl_loss)
            
            if shot_name == '10-shot':
                all_reptile_steps.append(rep_steps)
                all_maml_steps.append(maml_steps)
                all_baseline_steps.append(evaluate_baseline_sweep(x_sup[t], y_sup[t], x_que[t], y_que[t], step_counts=step_range))
            
            print(f"  {t+1:<4} {bl_loss:>12.4f} {rep_final:>12.4f} {maml_final:>12.4f}")
        
        results_table[shot_name] = {
            'baseline': np.mean(baseline_errors),
            'reptile': np.mean(reptile_errors),
            'maml': np.mean(maml_errors),
        }
    
    # Summary Table
    print(f"\n{'Method':<30} {'5-shot MSE':>12} {'10-shot MSE':>12} {'20-shot MSE':>12}")
    print("-" * 65)
    for m in ['baseline', 'maml', 'reptile']:
        name = "Baseline" if m == 'baseline' else m.upper()
        print(f"{name:<30} {results_table['5-shot'][m]:>12.4f} {results_table['10-shot'][m]:>12.4f} {results_table['20-shot'][m]:>12.4f}")
    
    # Plotting Section
    print("\nGenerating Plots...")
    x_tr_sup, y_tr_sup, x_tr_que, y_tr_que = load_tasks(os.path.join(data_dir, "train_data.npz"))
    
    # Quick training re-run for loss curve
    reptile_losses, maml_losses = [], []
    model_r, model_m = WirelessNet(), WirelessNet()
    opt_r = optim.Adam(model_r.parameters(), lr=0.001)
    opt_m = optim.Adam(model_m.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    num_train = x_tr_sup.shape[0]
    
    for _ in range(300):
        batch = np.random.choice(num_train, size=10, replace=False)
        
        # Reptile step
        orig_w = [p.clone().detach() for p in model_r.parameters()]
        sum_w = [torch.zeros_like(p) for p in model_r.parameters()]
        total_l = 0
        for idx in batch:
            tm = WirelessNet()
            tm.load_state_dict(model_r.state_dict())
            iopt = optim.Adam(tm.parameters(), lr=0.02)
            for _ in range(5):
                p = tm(x_tr_sup[idx])
                l = loss_fn(p, y_tr_sup[idx])
                iopt.zero_grad(); l.backward(); iopt.step()
            total_l += l.item()
            for i, p in enumerate(tm.parameters()): sum_w[i] += p.data
        for i, p in enumerate(model_r.parameters()): p.grad = (orig_w[i] - sum_w[i]/10)
        opt_r.step(); reptile_losses.append(total_l/10)
        
        # MAML step
        total_ql = 0
        opt_m.zero_grad()
        for idx in batch:
            tm = WirelessNet()
            tm.load_state_dict(model_m.state_dict())
            iopt = optim.Adam(tm.parameters(), lr=0.02)
            for _ in range(5):
                p = tm(x_tr_sup[idx]); l = loss_fn(p, y_tr_sup[idx])
                iopt.zero_grad(); l.backward(); iopt.step()
            qp = tm(x_tr_que[idx]); ql = loss_fn(qp, y_tr_que[idx])
            total_ql += ql.item()
            tm.zero_grad(); ql.backward()
            for mp, tp in zip(model_m.parameters(), tm.parameters()):
                if tp.grad is not None:
                    if mp.grad is None: mp.grad = tp.grad.clone()/10
                    else: mp.grad += tp.grad.clone()/10
        opt_m.step(); maml_losses.append(total_ql/10)
    
    # Save Plot 1
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(reptile_losses, label='Reptile'); ax1.plot(maml_losses, label='MAML')
    ax1.set_title('Meta-Training Loss'); ax1.legend(); fig1.tight_layout()
    p1 = os.path.join(results_dir, 'plot_loss.png')
    fig1.savefig(p1, dpi=150)
    print(f"Saved: {p1}")
    
    # Save Plot 2
    avg_bl = np.mean(all_baseline_steps, axis=0)
    avg_rep = np.mean(all_reptile_steps, axis=0)
    avg_maml = np.mean(all_maml_steps, axis=0)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(step_range, avg_bl, label='Baseline', marker='o')
    ax2.plot(step_range, avg_rep, label='Reptile', marker='s')
    ax2.plot(step_range, avg_maml, label='MAML', marker='^')
    ax2.set_title('Few-Shot Adaptation Comparison'); ax2.legend(); fig2.tight_layout()
    p2 = os.path.join(results_dir, 'plot_comparison.png')
    fig2.savefig(p2, dpi=150)
    print(f"Saved: {p2}\n✓ Done!")
