"""
Creates synthetic datasets for few-shot wireless channel estimation.
Generates isolated tasks, each representing a unique transmission environment.
"""

import numpy as np
from pathlib import Path
import argparse


class EnvSimulator:
    """
    Simulates unique wireless environments for meta-learning.
    Generates pilot signals (inputs) and corresponding channel responses (labels)
    under varying noise, SNR, and multipath conditions.
    """
    def __init__(self, in_features=4, out_features=1, seed=None):
        self.in_features = in_features
        self.out_features = out_features
        
        if seed is not None:
            np.random.seed(seed)

    def _create_pilots(self, num_samples, snr_db, noise_std):
        """Generates random pilot signals with additive white Gaussian noise."""
        raw_signal = np.random.randn(num_samples, self.in_features)
        
        # Calculate noise power from SNR
        noise_var = 10 ** (-snr_db / 10)
        noisy_signal = raw_signal + np.sqrt(noise_var) * np.random.randn(num_samples, self.in_features)
        
        return noisy_signal

    def _compute_response(self, pilots, h_coeffs):
        """Calculates the true channel response for given pilot signals."""
        responses = np.zeros((pilots.shape[0], self.out_features))
        
        for idx, h in enumerate(h_coeffs[:self.out_features]):
            responses[:, 0] += np.real(h) * pilots[:, idx % self.in_features]
            
        return responses

    def build_environment_profile(self, n_supp=8, n_query=64, snr_db=10, paths=3, n_scale=0.1):
        """Constructs a single few-shot task (support and query sets)."""
        # Create a unique underlying channel function for this task
        h_base = np.random.randn(paths) + 1j * np.random.randn(paths)
        h_norm = h_base / np.linalg.norm(h_base)
        
        # Build Support Set (Few-shot samples for adaptation)
        supp_x = self._create_pilots(n_supp, snr_db, n_scale)
        supp_y = self._compute_response(supp_x, h_norm)
        
        # Build Query Set (Samples for evaluation)
        query_x = self._create_pilots(n_query, snr_db, n_scale)
        query_y = self._compute_response(query_x, h_norm)
        
        return {
            'X_support': supp_x.astype(np.float32),
            'Y_support': supp_y.astype(np.float32),
            'X_query': query_x.astype(np.float32),
            'Y_query': query_y.astype(np.float32),
            'snr': snr_db,
            'num_paths': paths,
            'noise_scale': n_scale
        }

    def generate_task_batch(self, batch_size, n_supp=8, n_query=64):
        """Creates a diverse batch of tasks with randomized environmental parameters."""
        task_list = []
        
        for _ in range(batch_size):
            task = self.build_environment_profile(
                n_supp=n_supp,
                n_query=n_query,
                snr_db=np.random.uniform(5.0, 20.0),
                paths=np.random.randint(2, 6),
                n_scale=np.random.uniform(0.05, 0.20)
            )
            task_list.append(task)
            
        return task_list


def export_npz(train_batch, test_batch, out_folder='results'):
    """Compiles task lists into compressed arrays and saves to disk."""
    dest_path = Path(out_folder)
    dest_path.mkdir(exist_ok=True, parents=True)
    
    def compile_dict(task_list):
        keys = ['X_support', 'Y_support', 'X_query', 'Y_query', 'snr', 'num_paths', 'noise_scale']
        return {k: np.stack([t[k] for t in task_list]) for k in keys}
        
    compiled_train = compile_dict(train_batch)
    compiled_test = compile_dict(test_batch)
    
    loc_train = dest_path / 'train_tasks.npz'
    loc_test = dest_path / 'test_tasks.npz'
    
    np.savez_compressed(loc_train, **compiled_train)
    np.savez_compressed(loc_test, **compiled_test)
    
    print(f"\n[SUCCESS] Train tasks saved -> {loc_train}")
    print(f"   -> Count: {compiled_train['X_support'].shape[0]}")
    print(f"   -> Support Matrix: {compiled_train['X_support'].shape[1:]}")
    print(f"   -> Query Matrix: {compiled_train['X_query'].shape[1:]}")
    
    print(f"\n[SUCCESS] Test tasks saved -> {loc_test}")
    print(f"   -> Count: {compiled_test['X_support'].shape[0]}")


def inspect_environments(task_list, check_count=3):
    """Outputs environmental parameters to verify task heterogeneity."""
    print("\n--- Environmental Heterogeneity Check ---")
    
    limit = min(check_count, len(task_list))
    for i in range(limit):
        env = task_list[i]
        print(f"Environment {i + 1}:")
        print(f"  > SNR: {env['snr']:.2f} dB | Multipath Elements: {env['num_paths']}")
        print(f"  > Target Distribution (Min/Max): [{env['Y_support'].min():.3f}, {env['Y_support'].max():.3f}]\n")


def main():
    print("=" * 65)
    print(" Meta-Learning Wireless Data Simulator ")
    print("=" * 65 + "\n")

    parser = argparse.ArgumentParser(description="Builds synthetic datasets for channel estimation.")
    parser.add_argument("--train-tasks", type=int, default=100, help="Train task count")
    parser.add_argument("--test-tasks", type=int, default=20, help="Test task count")
    parser.add_argument("--n-support", type=int, default=8, help="Samples per support set")
    parser.add_argument("--n-query", type=int, default=64, help="Samples per query set")
    parser.add_argument("--seed", type=int, default=None, help="Randomization seed")
    parser.add_argument("--output-dir", default="results", help="Target output folder")
    
    cfg = parser.parse_args()

    # Bootstrap Simulator
    sim = EnvSimulator(in_features=4, out_features=1, seed=cfg.seed)

    print(f"Constructing {cfg.train_tasks} training environments...")
    train_data = sim.generate_task_batch(cfg.train_tasks, cfg.n_support, cfg.n_query)

    print(f"Constructing {cfg.test_tasks} testing environments...")
    test_data = sim.generate_task_batch(cfg.test_tasks, cfg.n_support, cfg.n_query)

    # Save and Validate
    export_npz(train_data, test_data, out_folder=cfg.output_dir)
    
    print("Validating Training Set Diversity:")
    inspect_environments(train_data, check_count=2)
    
    print("Validating Test Set Diversity:")
    inspect_environments(test_data, check_count=2)

    print(f"[COMPLETE] Simulation finished. Generated {len(train_data) + len(test_data)} total environments.")


if __name__ == '__main__':
    main()