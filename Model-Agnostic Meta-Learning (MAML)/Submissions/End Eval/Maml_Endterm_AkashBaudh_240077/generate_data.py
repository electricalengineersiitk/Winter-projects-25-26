"""
Generate synthetic wireless channel estimation tasks for meta-learning.
Each task represents one wireless environment with different conditions (SNR, noise, paths).
"""

import numpy as np
from pathlib import Path


class WirelessTaskGenerator:
    """
    Generates meta-learning tasks for wireless channel estimation.
    
    Task definition:
    - One task = one wireless environment with fixed parameters (SNR, num_paths, noise_level)
    - Support set: Small number of pilot observations for fast adaptation (5-10 samples)
    - Query set: Larger number of samples for performance evaluation (50-100 samples)
    - All tasks share the same underlying structure but with varied parameters
    """
    
    def __init__(self, input_dim=4, output_dim=1, random_seed=None):
        """
        Initialize task generator.
        
        Args:
            input_dim: Input dimension (pilot observations)
            output_dim: Output dimension (channel coefficients)
            random_seed: Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.random_seed = random_seed
        # If a seed is provided, make generation reproducible. Otherwise use random state.
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate_single_task(self, n_support=8, n_query=64, snr=10, num_paths=3, noise_scale=0.1):
        """
        Generate a single meta-learning task.
        
        A task consists of:
        - X_support: Support set inputs (n_support, input_dim)
        - Y_support: Support set labels (n_support, output_dim)
        - X_query: Query set inputs (n_query, input_dim)
        - Y_query: Query set labels (n_query, output_dim)
        
        Args:
            n_support: Number of support samples (5-10)
            n_query: Number of query samples (50-100)
            snr: Signal-to-noise ratio in dB
            num_paths: Number of multipath components (varies task difficulty)
            noise_scale: Gaussian noise standard deviation
            
        Returns:
            Dictionary with X_support, Y_support, X_query, Y_query
        """
        # Sample random channel coefficients (task-specific parameters)
        # These define the underlying true function for this task
        channel_coeffs = np.random.randn(num_paths) + 1j * np.random.randn(num_paths)
        channel_coeffs = channel_coeffs / np.linalg.norm(channel_coeffs)  # Normalize
        
        # Generate support set
        X_support = self._generate_pilot_signals(n_support, snr, noise_scale)
        Y_support = self._estimate_channel(X_support, channel_coeffs)
        
        # Generate query set from SAME task (same parameters)
        X_query = self._generate_pilot_signals(n_query, snr, noise_scale)
        Y_query = self._estimate_channel(X_query, channel_coeffs)
        
        return {
            'X_support': X_support.astype(np.float32),
            'Y_support': Y_support.astype(np.float32),
            'X_query': X_query.astype(np.float32),
            'Y_query': Y_query.astype(np.float32),
            'snr': snr,
            'num_paths': num_paths,
            'noise_scale': noise_scale
        }
    
    def _generate_pilot_signals(self, n_samples, snr, noise_scale):
        """
        Generate pilot signals (sub-carriers with known values).
        
        X represents observations in the wireless channel.
        Shape: (n_samples, input_dim)
        """
        # Random pilot signal matrix
        X = np.random.randn(n_samples, self.input_dim)
        
        # Add noise based on SNR
        noise_power = 10 ** (-snr / 10)
        X = X + np.sqrt(noise_power) * np.random.randn(n_samples, self.input_dim)
        
        return X
    
    def _estimate_channel(self, X, channel_coeffs):
        """
        Compute channel estimates based on pilots and true channel.
        
        Y represents the true channel responses corresponding to inputs X.
        Shape: (n_samples, output_dim)
        """
        # Simple linear relationship: different feature combinations
        # In real wireless: Y = H @ X^H (Hermitian transpose)
        # Simplified here to regressors on pilot signals
        Y = np.zeros((X.shape[0], self.output_dim))
        
        for i, coeff in enumerate(channel_coeffs[:self.output_dim]):
            Y[:, 0] += np.real(coeff) * X[:, i % self.input_dim]
        
        return Y
    
    def generate_task_distribution(self, n_tasks, n_support=8, n_query=64):
        """
        Generate multiple tasks with VARIED parameters.
        
        Diversity is achieved by:
        - Varying SNR (5 to 20 dB)
        - Varying number of paths (2 to 5)
        - Varying noise scale (0.05 to 0.2)
        
        Args:
            n_tasks: Number of tasks to generate
            n_support: Support set size per task
            n_query: Query set size per task
            
        Returns:
            List of task dictionaries
        """
        tasks = []
        
        for _ in range(n_tasks):
            # Sample task parameters - vary them to ensure diversity
            snr = np.random.uniform(5, 20)  # SNR in dB
            num_paths = np.random.randint(2, 6)  # 2-5 paths
            noise_scale = np.random.uniform(0.05, 0.2)
            
            task = self.generate_single_task(
                n_support=n_support,
                n_query=n_query,
                snr=snr,
                num_paths=num_paths,
                noise_scale=noise_scale
            )
            tasks.append(task)
        
        return tasks


def save_dataset(train_tasks, test_tasks, output_dir='results'):
    """
    Save dataset in structured NPZ format.
    
    Args:
        train_tasks: List of training tasks
        test_tasks: List of test tasks
        output_dir: Directory to save files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Combine tasks into arrays
    train_data = {
        'X_support': np.stack([t['X_support'] for t in train_tasks]),
        'Y_support': np.stack([t['Y_support'] for t in train_tasks]),
        'X_query': np.stack([t['X_query'] for t in train_tasks]),
        'Y_query': np.stack([t['Y_query'] for t in train_tasks]),
        'snr': np.array([t['snr'] for t in train_tasks]),
        'num_paths': np.array([t['num_paths'] for t in train_tasks]),
        'noise_scale': np.array([t['noise_scale'] for t in train_tasks]),
    }
    
    test_data = {
        'X_support': np.stack([t['X_support'] for t in test_tasks]),
        'Y_support': np.stack([t['Y_support'] for t in test_tasks]),
        'X_query': np.stack([t['X_query'] for t in test_tasks]),
        'Y_query': np.stack([t['Y_query'] for t in test_tasks]),
        'snr': np.array([t['snr'] for t in test_tasks]),
        'num_paths': np.array([t['num_paths'] for t in test_tasks]),
        'noise_scale': np.array([t['noise_scale'] for t in test_tasks]),
    }
    
    # Save as NPZ files (compressed NumPy format)
    train_path = output_path / 'train_tasks.npz'
    test_path = output_path / 'test_tasks.npz'
    
    np.savez_compressed(train_path, **train_data)
    np.savez_compressed(test_path, **test_data)
    
    print(f"[OK] Training tasks saved: {train_path}")
    print(f"  Shape: {train_data['X_support'].shape[0]} tasks")
    print(f"  Support set: {train_data['X_support'].shape[1:]} per task")
    print(f"  Query set:   {train_data['X_query'].shape[1:]} per task")
    print()
    print(f"[OK] Test tasks saved: {test_path}")
    print(f"  Shape: {test_data['X_support'].shape[0]} tasks")


def verify_dataset_diversity(tasks, n_samples=5):
    """
    Verify that tasks are different (check diversity).
    
    Args:
        tasks: List of task dictionaries
        n_samples: Number of tasks to inspect
    """
    print("\nDataset Diversity Check:")
    print("-" * 60)
    
    for i in range(min(n_samples, len(tasks))):
        task = tasks[i]
        print(f"Task {i}:")
        print(f"  SNR: {task['snr']:.1f} dB")
        print(f"  Paths: {task['num_paths']}")
        print(f"  Noise: {task['noise_scale']:.3f}")
        print(f"  Support loss range: [{task['Y_support'].min():.4f}, {task['Y_support'].max():.4f}]")
        print()


if __name__ == '__main__':
    import argparse
    print("=" * 60)
    print("MAML Wireless Channel Estimation Dataset Generator")
    print("=" * 60)
    print()

    parser = argparse.ArgumentParser(description="Generate meta-learning tasks for MAML experiments.")
    parser.add_argument("--train-tasks", type=int, default=100, help="Number of training tasks to generate (default: 100)")
    parser.add_argument("--test-tasks", type=int, default=20, help="Number of test tasks to generate (default: 20)")
    parser.add_argument("--n-support", type=int, default=8, help="Support set size per task (shots)")
    parser.add_argument("--n-query", type=int, default=64, help="Query set size per task")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility (default: random)")
    parser.add_argument("--output-dir", default="results", help="Directory to save generated datasets")
    args = parser.parse_args()

    # Initialize generator (use None seed by default -> new random data each run)
    generator = WirelessTaskGenerator(input_dim=4, output_dim=1, random_seed=args.seed)

    # Generate training and test tasks
    print(f"Generating {args.train_tasks} training tasks (support={args.n_support}, query={args.n_query})...")
    train_tasks = generator.generate_task_distribution(
        n_tasks=args.train_tasks,
        n_support=args.n_support,
        n_query=args.n_query
    )

    print(f"Generating {args.test_tasks} test tasks...")
    test_tasks = generator.generate_task_distribution(
        n_tasks=args.test_tasks,
        n_support=args.n_support,
        n_query=args.n_query
    )

    # Save dataset
    save_dataset(train_tasks, test_tasks, output_dir=args.output_dir)

    # Verify diversity
    verify_dataset_diversity(train_tasks, n_samples=3)
    verify_dataset_diversity(test_tasks, n_samples=3)

    print("[OK] Dataset generation complete!")
    print(f"  Total training tasks: {len(train_tasks)}")
    print(f"  Total test tasks: {len(test_tasks)}")
