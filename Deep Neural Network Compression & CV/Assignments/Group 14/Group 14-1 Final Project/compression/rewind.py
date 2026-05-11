import torch

def rewind_weights(model, initial_state_dict, masks):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                param.copy_(initial_state_dict[name] * masks[name])