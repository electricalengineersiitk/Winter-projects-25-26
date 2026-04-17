import torch

def get_compute_device():
    if torch.cuda.is_available():
        ctx = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        ctx = torch.device("mps")
        print("[Device] Apple Silicon")
    else:
        ctx = torch.device("cpu")
        print("[Device] CPU")
    return ctx