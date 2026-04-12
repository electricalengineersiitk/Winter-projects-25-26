import numpy as np
import os

def generate_channel_estimation_tasks(num_tasks, num_support=10, num_query=90, channel_dim=16):
    """
    Generates synthetic wireless tasks for channel estimation.
    A task = one wireless environment with different settings (e.g., SNR, path profile).
    """
    tasks_data = []
    
    for _ in range(num_tasks):
        # Vary SNR between 0 dB and 20 dB across different tasks
        snr_db = np.random.uniform(0, 20)
        snr_linear = 10 ** (snr_db / 10.0)
        
        # Simulating a wireless environment: Power delay profile for the multipath channel
        # Using an exponential decay model common in wireless communications
        pdp = np.exp(-np.arange(channel_dim) / 4.0)
        pdp = pdp / np.sum(pdp)  # Normalize power
        
        X_all = []
        Y_all = []
        
        # Generate total samples (support + query) for this specific environment
        for _ in range(num_support + num_query):
            # True channel H (Y_target)
            # We use a real-valued baseband representation to keep the PyTorch neural network simple
            H = np.random.normal(0, np.sqrt(pdp))
            
            # Pilot observation X = H + noise
            signal_power = np.mean(H**2)
            noise_power = signal_power / snr_linear
            noise = np.random.normal(0, np.sqrt(noise_power), size=channel_dim)
            
            X_obs = H + noise
            
            X_all.append(X_obs)
            Y_all.append(H)
            
        # Convert to float32 which is the standard for PyTorch tensors
        X_all = np.array(X_all, dtype=np.float32)
        Y_all = np.array(Y_all, dtype=np.float32)
        
        # Store as dictionary per the structural requirements
        task_dict = {
            'X_support': X_all[:num_support],
            'Y_support': Y_all[:num_support],
            'X_query': X_all[num_support:],
            'Y_query': Y_all[num_support:],
            'snr_db': snr_db
        }
        tasks_data.append(task_dict)
        
    return tasks_data

def main():
    # Use relative paths so it works on any machine 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Generating 100 training tasks...")
    train_tasks = generate_channel_estimation_tasks(num_tasks=100)
    
    print("Generating 20 test tasks...")
    test_tasks = generate_channel_estimation_tasks(num_tasks=20)
    
    # Save everything to a .npz file
    save_path = os.path.join(current_dir, "channel_data.npz")
    np.savez_compressed(save_path, train=train_tasks, test=test_tasks)
    print(f"Data successfully saved to {save_path}!")
    
    # Optional: ensure files don't exceed GitHub's large file limits
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    # Ensure a fixed seed for reproducibility across test runs
    np.random.seed(42)
    main()
