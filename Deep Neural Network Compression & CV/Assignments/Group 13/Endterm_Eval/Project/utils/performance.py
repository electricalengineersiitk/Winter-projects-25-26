import time
import torch
import resource
import os

def measure_performance(model, device, input_size=(1, 3, 32, 32), num_runs=100):
    """
    Measures average inference latency and peak RAM usage.
    """
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)
    
    # 1. Measure Latency
    # Warm up the GPU/MPS first
    for _ in range(10):
        _ = model(dummy_input)
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / num_runs * 1000  # Convert to ms

    # 2. Measure RAM (Peak Resident Set Size)
    # resource.ru_maxrss returns bytes on macOS
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ram_mb = usage / (1024 * 1024) 

    return avg_latency, ram_mb