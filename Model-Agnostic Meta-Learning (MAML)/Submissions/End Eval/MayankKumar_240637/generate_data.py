import numpy as np
import os

def generate_channel_tasks(num_tasks, num_support=10, num_query=50, num_features=64):
    """
    Generates synthetic wireless channel estimation tasks.
    Each task represents a different wireless environment (different SNR, multipath).
    """
    X_support_all, Y_support_all = [], []
    X_query_all, Y_query_all = [], []

    # Time/frequency vector for generating smooth channel responses
    t = np.linspace(0, 1, num_features)

    for _ in range(num_tasks):
        # 1. Randomize Environment Variables for this specific task
        snr_db = np.random.uniform(0, 20)          # SNR between 0dB and 20dB
        noise_std = 10 ** (-snr_db / 20)           # Convert SNR to noise standard deviation
        num_paths = np.random.randint(2, 7)        # 2 to 6 multipath components

        # 2. Create the "Base Channel" (H) for this environment
        # We model the channel as a sum of sine waves (representing different paths/fades)
        H_base = np.zeros(num_features)
        for _ in range(num_paths):
            amplitude = np.random.randn()
            frequency = np.random.uniform(1, 5)
            phase = np.random.uniform(0, 2 * np.pi)
            H_base += amplitude * np.sin(2 * np.pi * frequency * t + phase)
        
        # Normalize the channel power to 1
        H_base = H_base / np.sqrt(np.mean(H_base**2))

        # 3. Generate Support Set (5-10 shots for adaptation)
        # Y is the true channel (with slight time-variations). X is the noisy pilot observation.
        Y_supp = H_base + 0.05 * np.random.randn(num_support, num_features) 
        X_supp = Y_supp + noise_std * np.random.randn(num_support, num_features)

        # 4. Generate Query Set (50-100 samples for evaluation)
        Y_quer = H_base + 0.05 * np.random.randn(num_query, num_features)
        X_quer = Y_quer + noise_std * np.random.randn(num_query, num_features)

        X_support_all.append(X_supp)
        Y_support_all.append(Y_supp)
        X_query_all.append(X_quer)
        Y_query_all.append(Y_quer)

    # Convert lists to NumPy arrays
    return {
        "x_support": np.array(X_support_all, dtype=np.float32),
        "y_support": np.array(Y_support_all, dtype=np.float32),
        "x_query": np.array(X_query_all, dtype=np.float32),
        "y_query": np.array(Y_query_all, dtype=np.float32)
    }

if __name__ == "__main__":
    print("Generating synthetic dataset for Channel Estimation...")
    
    # Generate 100 Training Tasks and 20 Testing Tasks
    train_data = generate_channel_tasks(num_tasks=100, num_support=10, num_query=50)
    test_data = generate_channel_tasks(num_tasks=20, num_support=10, num_query=50)

    # Create a dictionary to hold everything
    dataset = {
        "train_x_support": train_data["x_support"],
        "train_y_support": train_data["y_support"],
        "train_x_query": train_data["x_query"],
        "train_y_query": train_data["y_query"],
        "test_x_support": test_data["x_support"],
        "test_y_support": test_data["y_support"],
        "test_x_query": test_data["x_query"],
        "test_y_query": test_data["y_query"],
    }

    # Save to a single .npz file in the same directory
    save_path = "wireless_dataset.npz"
    np.savez(save_path, **dataset)

    print(f"Dataset successfully saved to: {save_path}")
    print(f"Train tasks shape: {dataset['train_x_support'].shape} -> (tasks, shots, features)")
