import numpy as np
import os
os.makedirs("data", exist_ok=True)

def generate_task(num_support=15, num_query=50, input_dim=10):

    A = np.random.uniform(0.5, 2.0)
    phase = np.random.uniform(0, np.pi)
    
    def get_y(x):
        return A * np.sin(x + phase) + 0.5 * x 

    X_support = np.random.uniform(-5, 5, (num_support, input_dim))
    Y_support = get_y(X_support) + 0.02 * np.random.randn(num_support, input_dim)

    X_query = np.random.uniform(-5, 5, (num_query, input_dim))
    Y_query = get_y(X_query) + 0.02 * np.random.randn(num_query, input_dim)

    return X_support, Y_support, X_query, Y_query

if __name__ == "__main__":
    train_tasks = [generate_task() for _ in range(1000)]
    test_tasks = [generate_task() for _ in range(100)]
    np.savez("data/dataset.npz", train_tasks=np.array(train_tasks, dtype=object), test_tasks=np.array(test_tasks, dtype=object))
    print(" High-quality data generated!")