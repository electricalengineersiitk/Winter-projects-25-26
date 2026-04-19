import numpy as np
import os

class ChannelDataGenerator:
    """
    Simulates multipath wireless channel environments for Meta-Learning.
    Each 'task' represents a unique environment with specific SNR and PDP.
    """
    def __init__(self, channel_dim=16, decay_factor=4.0):
        self.channel_dim = channel_dim
        self.decay_factor = decay_factor

    def create_tasks(self, num_tasks, num_support=10, num_query=90):
        tasks_data = []
        
        for _ in range(num_tasks):
            # 1. Environment Setup: Randomize SNR (0 to 20 dB)
            snr_db = np.random.uniform(0, 20)
            snr_linear = 10 ** (snr_db / 10.0)
            
            # 2. Power Delay Profile: Exponential decay normalized to unit power
            pdp = np.exp(-np.arange(self.channel_dim) / self.decay_factor)
            pdp /= np.sum(pdp) 
            
            # 3. Generate Channel Realizations
            total_samples = num_support + num_query
            
            # True Channel (H): Ground truth
            # Observation (X): H + Gaussian Noise
            H_true = np.random.normal(0, np.sqrt(pdp), size=(total_samples, self.channel_dim))
            
            signal_power = np.mean(H_true**2)
            noise_std = np.sqrt(signal_power / snr_linear)
            noise = np.random.normal(0, noise_std, size=H_true.shape)
            
            X_obs = H_true + noise

            # 4. Package as Float32 (PyTorch Compatible)
            task_dict = {
                'X_support': X_obs[:num_support].astype(np.float32),
                'Y_support': H_true[:num_support].astype(np.float32),
                'X_query': X_obs[num_support:].astype(np.float32),
                'Y_query': H_true[num_support:].astype(np.float32),
                'snr_db': snr_db
            }
            tasks_data.append(task_dict)
            
        return tasks_data

def main():
    # Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, "channel_data.npz")
    
    generator = ChannelDataGenerator(channel_dim=16)

    print("Generating synthetic wireless environments...")
    train_tasks = generator.create_tasks(num_tasks=100)
    test_tasks = generator.create_tasks(num_tasks=20)

    # Save as compressed archive
    np.savez_compressed(save_path, train=train_tasks, test=test_tasks)
    
    # Final Stats
    file_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"✅ Data saved to: {save_path}")
    print(f"📊 File size: {file_size:.2f} MB")

if __name__ == "__main__":
    np.random.seed(42)
    main()