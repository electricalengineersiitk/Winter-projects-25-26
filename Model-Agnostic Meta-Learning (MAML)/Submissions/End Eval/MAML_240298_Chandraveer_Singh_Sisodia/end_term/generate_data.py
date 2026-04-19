import numpy as np
import os

def generate_channel_tasks(num_tasks, samples_per_task, save_path):
    tasks = []
    for i in range(num_tasks):
        snr = np.random.uniform(0, 20)
        h_true = np.random.randn(1)
        X = np.random.randn(samples_per_task, 1)
        noise_std = 1 / np.sqrt(10**(snr/10))
        Y = h_true * X + np.random.normal(0, noise_std, (samples_per_task, 1))

        # Split: 10 for support (adaptation), 50 for query (evaluation)
        tasks.append({
            'x_s': X[:10], 'y_s': Y[:10],
            'x_q': X[10:], 'y_q': Y[10:],
            'h_true': h_true
        })
    np.save(save_path, tasks)

if __name__ == "__main__":
    # Using os.path.join to avoid hardcoding
    data_path = os.path.join("end_term", "data")
    if not os.path.exists(data_path): os.makedirs(data_path)
    generate_channel_tasks(100, 60, os.path.join(data_path, "train_tasks.npy"))
    generate_channel_tasks(20, 60, os.path.join(data_path, "test_tasks.npy"))
    print("Data generated successfully!")
