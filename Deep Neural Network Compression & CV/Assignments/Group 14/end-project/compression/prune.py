import numpy as np

def l1_prune(model, masks, prune_rate):
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.abs()
            alive_weights = tensor[masks[name] == 1]

            if len(alive_weights) == 0:
                continue

            threshold = np.percentile(
                alive_weights.cpu().numpy(), prune_rate * 100)

            new_mask = (tensor > threshold).float()
            masks[name] = masks[name] * new_mask

    return masks