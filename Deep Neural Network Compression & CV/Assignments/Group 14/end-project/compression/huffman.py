import os
import numpy as np

def save_compressed_model(quantized_state_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, **quantized_state_dict)


def compute_compression_stats(model, save_path):
    total_params = sum(p.numel() for p in model.parameters())
    baseline_size_mb = (total_params * 4) / (1024 * 1024)

    compressed_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    compression_ratio = baseline_size_mb / compressed_size_mb

    print("\n--- MEMORY ALLOCATION CHECK ---")
    print(f"Original Model Size (Theoretical): {baseline_size_mb:.2f} MB")
    print(f"Final Compressed Size (NPZ):       {compressed_size_mb:.2f} MB")
    print(f"Compression Ratio Achieved:        {compression_ratio:.2f}x")

    if compression_ratio >= 9.0:
        print("SUCCESS: You hit the 9x compression target!")