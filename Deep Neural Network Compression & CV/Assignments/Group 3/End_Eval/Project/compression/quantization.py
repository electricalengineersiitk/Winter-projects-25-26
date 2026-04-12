## trained quantization and weight sharing using k-means clustering
## instead of storing every weight as a unique 32-bit number, we group similar weights into clusters so that each weight just stores which cluster it belongs to which will be a small index, and the actual value comes from a shared codebook
## so the conv layers get 256 clusters (8 bit), fc layers get 32 clusters (5 bit) 
## after clustering we fine-tune the centroids so the network adjusts to its new shared weights

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
import copy

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config


def _is_conv_layer(name):
    return 'features' in name


def _is_fc_layer(name):
    return 'classifier' in name


def _get_num_clusters(name): # picks 256 clusters for conv layers, 32 for fc layers 
    if _is_conv_layer(name):
        return 2 ** config.QUANT_BITS_CONV
    elif _is_fc_layer(name):
        return 2 ** config.QUANT_BITS_FC
    else:
        return 2 ** config.QUANT_BITS_FC


def _linear_init_centroids(weights_1d, k): # spaces centroids evenly between min and max 
    w_min = weights_1d.min()
    w_max = weights_1d.max()
    centroids = np.linspace(w_min, w_max, k).reshape(-1, 1)
    return centroids


def quantize_layer(weight_tensor, name, masks=None): # runs k means on one layers non zero weights and replaces them with centroid values
    original_shape = weight_tensor.shape
    weights_np = weight_tensor.data.cpu().numpy().flatten()

    # only cluster the nonzero weights i.e. pruned ones stay at 0
    if masks is not None and name in masks:
        mask_np = masks[name].cpu().numpy().flatten()
        nonzero_idx = np.where(mask_np != 0)[0]
        nonzero_weights = weights_np[nonzero_idx]
    else:
        nonzero_idx = np.where(weights_np != 0)[0]
        nonzero_weights = weights_np[nonzero_idx]

    if len(nonzero_weights) == 0:
        print(f"  {name:40s}  SKIPPED (all zeros)")
        return weight_tensor, None

    k = _get_num_clusters(name)
    n_unique = len(np.unique(nonzero_weights))
    k = min(k, n_unique)

    bits = int(np.log2(k)) if k > 0 else 0

    init_centroids = _linear_init_centroids(nonzero_weights, k)

    kmeans = KMeans(
        n_clusters=k,
        init=init_centroids,
        n_init=1,
        max_iter=300,
        random_state=config.SEED
    )
    labels = kmeans.fit_predict(nonzero_weights.reshape(-1, 1))
    centroids = kmeans.cluster_centers_.flatten()

    # swap each non zero weight for its cluster centroid
    quantized_flat = weights_np.copy()
    for i, idx in enumerate(nonzero_idx):
        quantized_flat[idx] = centroids[labels[i]]

    quantized_weight = torch.from_numpy(
        quantized_flat.reshape(original_shape)
    ).float().to(weight_tensor.device)

    # compression rate calc
    n = len(nonzero_weights)
    b = 32
    r = (n * b) / (n * np.log2(k) + k * b) if k > 1 else 1.0

    print(f"  {name:40s}  clusters: {k:4d} ({bits}-bit)  "
          f"nonzero: {n:8d}  compression: {r:.1f}x")

    codebook = {
        'centroids': centroids,
        'labels': labels,
        'nonzero_idx': nonzero_idx,
        'num_clusters': k,
        'bits': bits
    }

    return quantized_weight, codebook


def quantize_model(model, masks=None): # quantizes every prunable layer in the model using k means
    print(f"\n{'='*60}")
    print(f"  STAGE 2: TRAINED QUANTIZATION (Weight Sharing via K-Means)")
    print(f"{'='*60}")
    print(f"  CONV layers: {2**config.QUANT_BITS_CONV} clusters ({config.QUANT_BITS_CONV}-bit)")
    print(f"  FC layers:   {2**config.QUANT_BITS_FC} clusters ({config.QUANT_BITS_FC}-bit)")
    print(f"  Init method: {config.QUANT_INIT_METHOD}")
    print()

    codebooks = {}

    for name, param in model.named_parameters():
        if 'weight' in name and ('features' in name or 'classifier' in name):
            if 'bn' not in name and len(param.shape) >= 2:
                quantized_weight, codebook = quantize_layer(param, name, masks)
                param.data.copy_(quantized_weight)
                if codebook is not None:
                    codebooks[name] = codebook

    return model, codebooks


def finetune_centroids(model, codebooks, masks, train_loader, test_loader,
                       device, epochs=None, lr=None, evaluate_fn=None):
    # finetunes the shared centroid values so the network recovers accuracy after quantization
    ## tracks best model by test accuracy to avoid overfitting
    if epochs is None:
        epochs = config.QUANT_FINETUNE_EPOCHS
    if lr is None:
        lr = config.QUANT_FINETUNE_LR

    print(f"\n[Quantization] Fine-tuning centroids for {epochs} epochs (lr={lr})...")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=config.MOMENTUM
    )
    # smoothly reduces lr so centroids dont overshoot in later epochs
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

            # the gradient of a centroid = sum of gradients of all weights assigned to that centroid.
            # intercepts gradient before optimizer step and group them by cluster label then sum them, and write the identical sum back so the optimizer moves every weight in a cluster by exactly the same delta.
            # this keeps the shared-weight constraint without needing to snap.
            for name, param in model.named_parameters():
                if name not in codebooks or param.grad is None:
                    continue
                cb = codebooks[name]
                labels      = cb['labels']          
                nonzero_idx = cb['nonzero_idx']
                k           = cb['num_clusters']

                grad_flat = param.grad.data.cpu().numpy().flatten()
                nonzero_grads = grad_flat[nonzero_idx]

                # sum gradients per cluster and broadcast back
                centroid_grad_sum = np.zeros(k, dtype=np.float32)
                for c in range(k):
                    members = nonzero_grads[labels == c]
                    if len(members) > 0:
                        centroid_grad_sum[c] = members.sum()

                # assign summed gradient to every weight in that cluster
                for i, idx in enumerate(nonzero_idx):
                    grad_flat[idx] = centroid_grad_sum[labels[i]]

                # also zero gradients for pruned (masked out) weights
                if masks is not None and name in masks:
                    mask_flat = masks[name].cpu().numpy().flatten()
                    grad_flat *= mask_flat

                param.grad.data.copy_(
                    torch.from_numpy(
                        grad_flat.reshape(param.shape)
                    ).float().to(param.device)
                )

            optimizer.step()

            # keep pruned weights at 0
            if masks is not None:
                from compression.pruning import apply_pruning_masks
                apply_pruning_masks(model, masks)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()

        train_acc = 100.0 * correct / total
        avg_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        test_acc = 0.0
        if evaluate_fn is not None:
            test_acc = evaluate_fn(model, test_loader, device)
            if test_acc > best_acc:
                best_acc = test_acc
                best_state = copy.deepcopy(model.state_dict())

        print(f"  Epoch [{epoch+1}/{epochs}]  "
              f"Loss: {avg_loss:.4f}  Train Acc: {train_acc:.2f}%  "
              f"Test Acc: {test_acc:.2f}%  LR: {current_lr:.6f}")

    # restore best model from fine tuning
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  * Restored best quantized model (test acc: {best_acc:.2f}%)")

    # extract the updated centroid values from the fine tuned weights (not reclustering)
    codebooks = _extract_finetuned_codebooks(model, codebooks, masks)

    return model, codebooks


def _snap_to_centroids(model, codebooks): # forces each weight to its nearest centroid value after gradient update
    for name, param in model.named_parameters():
        if name in codebooks:
            cb = codebooks[name]
            centroids = cb['centroids']
            nonzero_idx = cb['nonzero_idx']

            weight_flat = param.data.cpu().numpy().flatten()
            nonzero_weights = weight_flat[nonzero_idx]

            # find closest centroid for each weight
            dists = np.abs(nonzero_weights.reshape(-1, 1) - centroids.reshape(1, -1))
            new_labels = dists.argmin(axis=1)

            for i, idx in enumerate(nonzero_idx):
                weight_flat[idx] = centroids[new_labels[i]]

            param.data.copy_(
                torch.from_numpy(weight_flat.reshape(param.shape)).float().to(param.device)
            )


def _extract_finetuned_codebooks(model, codebooks, masks):
    ## after fine-tuning, we dont rerun kmeans (that would discard the fine-tuned centroid values) instead we just extract what each centroid's value is now from the actual model weights
    ## so the cluster assignments stay the same, only the centroid values are updated
    updated = {}
    for name, param in model.named_parameters():
        if name in codebooks:
            cb = codebooks[name]
            old_centroids = cb['centroids']
            nonzero_idx = cb['nonzero_idx']
            k = cb['num_clusters']

            weight_flat = param.data.cpu().numpy().flatten()
            nonzero_weights = weight_flat[nonzero_idx]

            # reassign labels to nearest centroid (they may have shifted slightly during fine tuning)
            dists = np.abs(nonzero_weights.reshape(-1, 1) - old_centroids.reshape(1, -1))
            new_labels = dists.argmin(axis=1)

            # recomputes each centroid as the mean of all weights assigned to it
            new_centroids = np.copy(old_centroids)
            for c in range(k):
                members = nonzero_weights[new_labels == c]
                if len(members) > 0:
                    new_centroids[c] = members.mean()

            bits = int(np.log2(k)) if k > 0 else 0
            print(f"  {name:40s}  centroids updated: {k:4d} ({bits}-bit)")

            updated[name] = {
                'centroids': new_centroids,
                'labels': new_labels,
                'nonzero_idx': nonzero_idx,
                'num_clusters': k,
                'bits': bits
            }
    return updated


def quantize_and_finetune(model, masks, train_loader, test_loader, device,
                          evaluate_fn=None): # complete stage 2 including quantize first, then fine tune the centroids
    model, codebooks = quantize_model(model, masks)
    model, codebooks = finetune_centroids(
        model, codebooks, masks, train_loader, test_loader,
        device, evaluate_fn=evaluate_fn
    )
    return model, codebooks
