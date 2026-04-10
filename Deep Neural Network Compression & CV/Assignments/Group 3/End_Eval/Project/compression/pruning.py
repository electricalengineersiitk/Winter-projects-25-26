## This does magnitude based weight-pruning
## pruning is done on the basis of the magnitude of the weights,weights with magnitude less than the threshold are pruned while weights with magnitude greater than the threshold are kept.
## we are essentially figuring out which connections are the weakest (smallest weights) and trim them out. 
## after the trim, we retrain the network so it can adjust to its new, lighter structure.
import torch
import torch.nn as nn
import numpy as np
import copy

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config


def compute_threshold(model, sparsity): # finds the cutoff point for our weights.. it figures out the value where that number of weights will be pruned according to our desired sparsity percentage
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name and ('features' in name or 'classifier' in name):
            if 'bn' not in name and len(param.shape) >= 2:
                all_weights.append(param.data.abs().cpu().numpy().flatten())

    all_weights = np.concatenate(all_weights) # pool all the weights together into one giant list and sort them from smallest to largest
    all_weights.sort()

    # the threshold is the value at the desired sparsity percentile
    threshold_index = int(len(all_weights) * sparsity)
    threshold_index = min(threshold_index, len(all_weights) - 1)
    threshold = float(all_weights[threshold_index])

    print(f"[Pruning] Global threshold: {threshold:.6f}  "
          f"(prune bottom {sparsity*100:.0f}% of {len(all_weights):,} weights)")

    return threshold


def create_pruning_masks(model, threshold): ## creates a mask for each layer, if weight grester than threshold then gets 1 otherwise 0
    masks = {}
    for name, param in model.named_parameters():
        if 'weight' in name and ('features' in name or 'classifier' in name):
            if 'bn' not in name and len(param.shape) >= 2:
                # Keep weights whose absolute value >= threshold
                mask = (param.data.abs() >= threshold).float()
                masks[name] = mask

                kept = mask.sum().item()
                total = mask.numel()
                print(f"  {name:40s}  kept: {kept:8.0f}/{total:8d}  "
                      f"({kept/total*100:5.1f}%)")

    return masks


def apply_pruning_masks(model, masks): ##all pruned connections are zeroed
    for name, param in model.named_parameters():
        if name in masks:
            param.data.mul_(masks[name].to(param.device))


def prune_model(model, sparsity=None): #calculates threshold, builds masks and applies them all at once to shrink the network.
    if sparsity is None:
        sparsity = config.PRUNE_SPARSITY

    print(f"\n{'='*60}")
    print(f"  STAGE 1: NETWORK PRUNING  (target sparsity: {sparsity*100:.0f}%)")
    print(f"{'='*60}")

    # Compute threshold
    threshold = compute_threshold(model, sparsity)

    # Create masks
    print(f"\n[Pruning] Per-layer statistics:")
    masks = create_pruning_masks(model, threshold)

    # Apply masks (zero out weights below threshold)
    apply_pruning_masks(model, masks)

    # Report overall sparse percentage
    total_params = 0
    total_zeros = 0
    for name, mask in masks.items():
        total_params += mask.numel()
        total_zeros += (mask == 0).sum().item()

    actual_sparsity = total_zeros / total_params if total_params > 0 else 0
    print(f"\n[Pruning] Actual sparsity achieved: {actual_sparsity*100:.2f}%  "
          f"({total_zeros:,} / {total_params:,} weights pruned)")
    print(f"[Pruning] Compression from pruning: {1/(1-actual_sparsity):.1f}x")

    return model, masks


def finetune_pruned_model(model, masks, train_loader, test_loader, device,
                          epochs=None, lr=None, evaluate_fn=None):
    ##fine tuning the model again, so it learns how to work with the new (pruned connections) model
    ## smoothly decay the learning rate, preventing overshoot in later epochs
    ## tracks the best model by test accuracy so we dont return an overfitted version
    if epochs is None:
        epochs = config.PRUNE_FINETUNE_EPOCHS
    if lr is None:
        lr = config.PRUNE_FINETUNE_LR

    print(f"\n[Pruning] Fine-tuning pruned model for {epochs} epochs (lr={lr})...")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_acc = 0.0
    best_state = None

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # zero out gradients for pruned weights before optimizer step
            # this prevents momentum from accumulating on dead connections which was causing gradient leakage and destabilising fine-tuning
            for name, param in model.named_parameters():
                if name in masks and param.grad is not None:
                    param.grad.data.mul_(masks[name].to(param.device))

            optimizer.step()

            # re-applies masks after each step to keep pruned weights at 0
            apply_pruning_masks(model, masks)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()

        train_acc = 100.0 * correct / total
        avg_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        # evaluates on test set and saves the best model
        test_acc = 0.0
        if evaluate_fn is not None:
            test_acc = evaluate_fn(model, test_loader, device)
            if test_acc > best_acc:
                best_acc = test_acc
                best_state = copy.deepcopy(model.state_dict())

        print(f"  Epoch [{epoch+1}/{epochs}]  "
              f"Loss: {avg_loss:.4f}  Train Acc: {train_acc:.2f}%  "
              f"Test Acc: {test_acc:.2f}%  LR: {current_lr:.6f}")

    # restore the best model we found during fine-tuning
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  * Restored best pruned model (test acc: {best_acc:.2f}%)")

    return model
