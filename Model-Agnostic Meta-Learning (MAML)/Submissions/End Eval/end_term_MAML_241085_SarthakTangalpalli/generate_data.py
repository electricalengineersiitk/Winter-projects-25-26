import numpy as np
import os

def generate_localization_tasks(num_tasks, num_support=10, num_query=50):
    '''
    Generates synthetic Wireless Localization tasks.
    Each task represents a different room/environment with unique Access Point locations
    and a different Path Loss Exponent (PLE).
    
    Inputs (X): Received Signal Strength (RSS) from 4 Access Points.
    Outputs (Y): The physical user coordinates (x, y, z).
    '''
    tasks_X_support = []
    tasks_Y_support = []
    tasks_X_query = []
    tasks_Y_query = []
    
    for _ in range(num_tasks):
        # Environment (Task) specific parameters:
        # 4 Access Points randomly located in a 100x100x100 volume
        ap_coords = np.random.uniform(0, 100, size=(4, 3))
        
        # Path loss exponent for this specific environment
        # e.g., 2.0 is free space, 3.5 is dense indoor office
        ple = np.random.uniform(2.0, 4.0)
        
        # Transmitter Power (varies realistically for standard 2.4GHz Indoor APs: 15-20 dBm)
        p_tx = np.random.uniform(15.0, 20.0)
        
        # Fixed path loss at reference distance (d0 = 1m) for 2.4GHz is roughly 40 dB
        # Shadow fading noise std dev (sigma) is typically 3-10 dB indoors, we'll use 4.0 dB
        pl_d0 = 40.0
        
        num_samples = num_support + num_query
        
        # User positions Y (x, y, z) coordinates within the volume
        Y = np.random.uniform(0, 100, size=(num_samples, 3))
        X = np.zeros((num_samples, 4))
        
        for i in range(4):
            # Calculate Euclidean distance to AP i
            distances = np.sqrt(np.sum((Y - ap_coords[i])**2, axis=1))
            
            # Realistic indoor Log-Distance Path Loss model:
            # Prx = Ptx - PL(d0) - 10 * ple * log10(d) + shadow_fading
            # We use d + 1 to prevent log(0) at exact 0 distance
            shadow_fading = np.random.normal(0, 4.0, size=num_samples)
            rss = p_tx - pl_d0 - 10 * ple * np.log10(distances + 1) + shadow_fading
            X[:, i] = rss
            
        # Normalize data to improve neural network training stability
        # RSS usually ranges from -30 dBm (close) to -90 dBm (far)
        X_normalized = (X - (-60.0)) / 30.0 # Scale to roughly [-1, 1] range
        Y_normalized = Y / 100.0            # Scale (100x100x100) coordinates to [0, 1]
        
        tasks_X_support.append(X_normalized[:num_support])
        tasks_Y_support.append(Y_normalized[:num_support])
        tasks_X_query.append(X_normalized[num_support:])
        tasks_Y_query.append(Y_normalized[num_support:])
        
    return {
        'x_support': np.array(tasks_X_support, dtype=np.float32),
        'y_support': np.array(tasks_Y_support, dtype=np.float32),
        'x_query': np.array(tasks_X_query, dtype=np.float32),
        'y_query': np.array(tasks_Y_query, dtype=np.float32)
    }

if __name__ == "__main__":
    np.random.seed(42) # For reproducibility
    
    print("Generating training tasks...")
    # 100 tasks, 10 support examples, 50 query examples
    train_data = generate_localization_tasks(num_tasks=100, num_support=10, num_query=50)
    
    print("Generating testing tasks (5-shot)...")
    test_data_5 = generate_localization_tasks(num_tasks=20, num_support=5, num_query=50)
    
    print("Generating testing tasks (10-shot)...")
    test_data_10 = generate_localization_tasks(num_tasks=20, num_support=10, num_query=50)
    
    print("Generating testing tasks (20-shot)...")
    test_data_20 = generate_localization_tasks(num_tasks=20, num_support=20, num_query=50)
    
    # Use absolute paths relative to the script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    os.makedirs(data_dir, exist_ok=True)
    np.savez(os.path.join(data_dir, "train_data.npz"), **train_data)
    np.savez(os.path.join(data_dir, "test_data_5shot.npz"), **test_data_5)
    np.savez(os.path.join(data_dir, "test_data.npz"), **test_data_10)
    np.savez(os.path.join(data_dir, "test_data_20shot.npz"), **test_data_20)
    print(f"Data successfully saved to {data_dir} folder!")
