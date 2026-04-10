import numpy as np
import os

def create_wifi_envs(n_envs, n_shots=10, n_eval=50):
    '''
    Simulates indoor 3D rooms for wireless localization.
    Each "task" (env) gets 4 random Access Points and a unique path loss exponent.
    '''
    # Lists to hold our few-shot splits
    x_shots, y_shots = [], []
    x_queries, y_queries = [], []
    
    for _ in range(n_envs):
        # Drop 4 Access Points randomly in a 100x100x100m space
        routers_pos = np.random.uniform(0, 100, size=(4, 3))
        
        # Room characteristics: 2.0 = open space, up to 4.0 = dense walls/obstacles
        path_loss_exp = np.random.uniform(2.0, 4.0)
        
        # Standard indoor router Tx power (15-20 dBm)
        tx_power = np.random.uniform(15.0, 20.0)
        
        # 40 dB is standard path loss at 1 meter for 2.4GHz Wi-Fi
        base_loss_1m = 40.0 
        
        total_samples = n_shots + n_eval
        
        # Generate random ground-truth user locations (x, y, z)
        user_locs = np.random.uniform(0, 100, size=(total_samples, 3))
        rssi_readings = np.zeros((total_samples, 4))
        
        for i in range(4):
            # Euclidean distance from all user locations to this specific AP
            dists = np.sqrt(np.sum((user_locs - routers_pos[i])**2, axis=1))
            
            # Simulate shadow fading (signal bouncing off walls, people, etc.)
            fading_noise = np.random.normal(0, 4.0, size=total_samples)
            
            # Log-Distance Path Loss model
            # Note: adding 1 to distance to avoid log(0) explosions if user is exactly on the AP
            rx_power = tx_power - base_loss_1m - 10 * path_loss_exp * np.log10(dists + 1) + fading_noise
            rssi_readings[:, i] = rx_power
            
        # Neural nets hate raw RSSI values (-90 to -30). Scale them to roughly [-1, 1].
        rssi_norm = (rssi_readings - (-60.0)) / 30.0 
        
        # Scale physical coords from 100m to [0, 1] range for stable gradients
        locs_norm = user_locs / 100.0            
        
        # Split into support (few-shot context) and query (target evaluation)
        x_shots.append(rssi_norm[:n_shots])
        y_shots.append(locs_norm[:n_shots])
        x_queries.append(rssi_norm[n_shots:])
        y_queries.append(locs_norm[n_shots:])
        
    return {
        'x_support': np.array(x_shots, dtype=np.float32),
        'y_support': np.array(y_shots, dtype=np.float32),
        'x_query': np.array(x_queries, dtype=np.float32),
        'y_query': np.array(y_queries, dtype=np.float32)
    }

if __name__ == "__main__":
    np.random.seed(42) # Lock seed so we can compare models fairly
    
    print("Building training environments...")
    train_data = create_wifi_envs(n_envs=100, n_shots=10, n_eval=50)
    
    print("Building test sets (5, 10, and 20 shot scenarios)...")
    test_5 = create_wifi_envs(n_envs=20, n_shots=5, n_eval=50)
    test_10 = create_wifi_envs(n_envs=20, n_shots=10, n_eval=50)
    test_20 = create_wifi_envs(n_envs=20, n_shots=20, n_eval=50)
    
    # Save everything locally
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    np.savez(os.path.join(data_dir, "train_data.npz"), **train_data)
    np.savez(os.path.join(data_dir, "test_data_5shot.npz"), **test_5)
    np.savez(os.path.join(data_dir, "test_data.npz"), **test_10)
    np.savez(os.path.join(data_dir, "test_data_20shot.npz"), **test_20)
    
    print(f"Dataset generation complete. Saved to {data_dir}/")
