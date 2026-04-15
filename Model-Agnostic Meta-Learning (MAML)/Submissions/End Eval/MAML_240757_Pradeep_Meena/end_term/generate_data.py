import numpy as np
import os

def generate_symbols(num_samples, modulation):
    """Generates baseband I/Q symbols for a given modulation scheme."""
    if modulation == 'BPSK':
        symbols = np.random.choice([-1, 1], size=num_samples)
        return np.column_stack((symbols, np.zeros(num_samples)))
    elif modulation == 'QPSK':
        symbols_I = np.random.choice([-1, 1], size=num_samples) / np.sqrt(2)
        symbols_Q = np.random.choice([-1, 1], size=num_samples) / np.sqrt(2)
        return np.column_stack((symbols_I, symbols_Q))
    elif modulation == '16QAM':
        levels = [-3, -1, 1, 3]
        symbols_I = np.random.choice(levels, size=num_samples) / np.sqrt(10)
        symbols_Q = np.random.choice(levels, size=num_samples) / np.sqrt(10)
        return np.column_stack((symbols_I, symbols_Q))

def add_awgn_noise(signal, snr_db):
    """Adds Additive White Gaussian Noise based on the SNR."""
    snr_linear = 10 ** (snr_db / 10)
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power / 2), signal.shape)
    return signal + noise

def create_task(snr_db, shots=5, query_size=50):
    """Creates one task (environment) with support and query sets."""
    modulations = ['BPSK', 'QPSK', '16QAM']
    
    X_support, Y_support = [], []
    X_query, Y_query = [], []
    
    for class_idx, mod in enumerate(modulations):
        # Generate clean symbols
        support_clean = generate_symbols(shots, mod)
        query_clean = generate_symbols(query_size, mod)
        
        # Add environment-specific noise
        X_support.append(add_awgn_noise(support_clean, snr_db))
        Y_support.extend([class_idx] * shots)
        
        X_query.append(add_awgn_noise(query_clean, snr_db))
        Y_query.extend([class_idx] * query_size)
        
    # Stack and return
    return (np.vstack(X_support).astype(np.float32), np.array(Y_support, dtype=np.int64),
            np.vstack(X_query).astype(np.float32), np.array(Y_query, dtype=np.int64))

def generate_dataset(num_train=1000,num_test=20):
    print(f"Generating {num_train} training tasks and {num_test} test tasks...")
    
    def build_task_arrays(num_tasks):
        X_s_all, Y_s_all, X_q_all, Y_q_all = [], [], [], []
        for _ in range(num_tasks):
            snr = np.random.uniform(0, 20)
            X_s, Y_s, X_q, Y_q = create_task(snr_db=snr, shots=5, query_size=50)
            X_s_all.append(X_s)
            Y_s_all.append(Y_s)
            X_q_all.append(X_q)
            Y_q_all.append(Y_q)
        # Stack lists into uniform numpy arrays
        return np.array(X_s_all), np.array(Y_s_all), np.array(X_q_all), np.array(Y_q_all)

    # Generate Train and Test arrays
    X_s_tr, Y_s_tr, X_q_tr, Y_q_tr = build_task_arrays(num_train)
    X_s_te, Y_s_te, X_q_te, Y_q_te = build_task_arrays(num_test)
    
    # Save using os.path.join to prevent hardcoding issues
    save_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the arrays to npz files as distinct keyword arguments
    np.savez(os.path.join(save_dir, 'meta_train_tasks.npz'), 
             X_support=X_s_tr, Y_support=Y_s_tr, X_query=X_q_tr, Y_query=Y_q_tr)
             
    np.savez(os.path.join(save_dir, 'meta_test_tasks.npz'), 
             X_support=X_s_te, Y_support=Y_s_te, X_query=X_q_te, Y_query=Y_q_te)
    
    print(f"Dataset successfully saved to {save_dir}")

if __name__ == "__main__":
    generate_dataset()