
import numpy as np
import os
import torch
import psutil

def measure_system_ram():
  
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    ram_mb = mem_info.rss / (1024 * 1024) # RSS is 'Resident Set Size'
    print(f"System RAM Used: {ram_mb:.2f} MB")
    return ram_mb



def measure_model_size(model_path):
    
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"File: {model_path}")
    print(f"Disk Space: {size_mb:.4f} MB")
    return size_mb

def measure_runtime_memory(device):
    
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
        print(f"GPU Memory Allocated: {allocated:.2f} MB")
        print(f"GPU Memory Reserved:  {reserved:.2f} MB")
    