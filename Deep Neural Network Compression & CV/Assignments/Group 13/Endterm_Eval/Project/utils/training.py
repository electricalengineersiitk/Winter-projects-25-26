import torch
from config import DEVICE, K_MEANS_CLUSTERS, CENTROID_LR
from compression.conv2d import modified_conv2d
from compression.linear import modified_linear

# Standard Training (Used for Stages 0 and 1)
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100. * correct / total

# Trained Quantization (Used for Stage 2.5)
def train_quantized_epoch(model, dataloader, criterion):
    """
    Optimized Trained Quantization using vectorized index summation.
    Aggregates gradients for all weights sharing a centroid.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        model.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # VECTORIZED CENTROID UPDATE 
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, (modified_conv2d, modified_linear)) and m.is_quantized:
                    # Flatten for fast indexing on the M1 GPU
                    w_grad = m.weight.grad.view(-1)
                    indices = m.indices.view(-1)
                    mask = m.mask.view(-1).bool()
                    
                    # 1. Filter for non-pruned connections
                    valid_grads = w_grad[mask]
                    valid_indices = indices[mask]
                    
                    # 2. Sum gradients for all clusters at once
                    cluster_sum = torch.zeros(K_MEANS_CLUSTERS, device=DEVICE)
                    cluster_sum.index_add_(0, valid_indices, valid_grads)
                    
                    # 3. Update centroids while maintaining weight sharing
                    centroid_updates = cluster_sum[indices] * CENTROID_LR
                    m.weight.data.sub_(centroid_updates.view(m.weight.shape) * m.mask)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100. * correct / total