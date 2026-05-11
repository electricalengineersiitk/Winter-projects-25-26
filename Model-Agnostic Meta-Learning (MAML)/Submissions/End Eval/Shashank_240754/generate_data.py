import numpy as np
import os

np.random.seed(42)

def generate_task(n_support=10, n_query=50, n_anchors=5):
    anchors = np.random.uniform(-10, 10, (n_anchors, 2))

    def simulate(n_samples):
        positions = np.random.uniform(-10, 10, (n_samples, 2))
        dists = np.linalg.norm(positions[:, None, :] - anchors[None, :, :], axis=2)
        
        rssi = 1 / (dists**2 + 1e-3)
        noise = np.random.normal(0, 0.01, rssi.shape)

        # ✅ NORMALIZATION (IMPORTANT)
        rssi = (rssi - np.mean(rssi)) / (np.std(rssi) + 1e-6)
        positions = positions / 10.0

        return rssi + noise, positions

    X_s, Y_s = simulate(n_support)
    X_q, Y_q = simulate(n_query)

    return X_s, Y_s, X_q, Y_q


def create_dataset(n_tasks):
    return [generate_task() for _ in range(n_tasks)]


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    train_tasks = create_dataset(100)
    test_tasks = create_dataset(20)

    np.save("data/train.npy", np.array(train_tasks, dtype=object))
    np.save("data/test.npy", np.array(test_tasks, dtype=object))

    print("Data generated successfully!")