# MAML for Wireless Channel Estimation

Meta-learning implementation for fast adaptation to wireless channel estimation tasks.

## Quick Start

Clone the repository and run the pipeline. By default the pipeline generates new random data on each run (use `--seed` to reproduce):

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python generate_data.py --seed 123

# 3. Train meta-learning model (uses stable hyperparameters)
python train.py

# 4. Test on new tasks (evaluates MAML vs baseline)
python test.py
```

After ~2 minutes, results are generated in `results/`:
- `train_tasks.npz` / `test_tasks.npz` – generated meta-learning dataset
- `plot_loss.png` – training loss curve
- `plot_comparison.png` – MAML vs baseline comparison
- `maml_model.pt` – trained meta-learning model

**Note:** All data, models, and plots are generated fresh by running the pipeline. No pre-saved files are included.

**Hyperparameters (stable settings for channel estimation):**
- `inner_lr = 0.0001` – Task adaptation learning rate (preventing divergence)
- `inner_steps = 1` – Gradient steps per task (preventing overfitting)
- `outer_lr = 0.001` – Meta-learning rate
- `meta_iterations = 100` – Meta-training iterations

## Problem Setup

### What is Channel Estimation?

Wireless signals travel through environments with obstacles, reflections, and fading. We need to **estimate the channel** using pilot signals to recover the transmitted message.

**Mathematical model:**
$$Y = XH + \text{noise}$$

Where:
- `X` = pilot signals (known by receiver)
- `H` = channel coefficients (unknown, what we estimate)
- `Y` = received signals (noisy observations)

### Why Meta-Learning?

Different wireless environments have different channels (indoor/outdoor, distance, obstacles, etc.). After observing just a **few pilots**, a good estimator should quickly adapt to new channels.

**MAML learns how to adapt quickly** by training on many channel tasks and learning an initialization that takes good inner-loop steps.

## Dataset Structure

### One Task = One Wireless Environment

Each task represents a channel with fixed parameters:
- **Channel coefficients H**: Randomly sampled per task (simulates different environments)
- **SNR (Signal-to-Noise Ratio)**: Varies 5–20 dB (difficulty level)
- **Number of paths**: 2–5 multipath components (channel complexity)
- **Noise scale**: 0.05–0.2 standard deviation

### Support Set (Few-Shot Adaptation)

Task contains **8 pilot observations** for the model to adapt:
- `X_support`: (8, 4) pilot signals
- `Y_support`: (8, 1) received signals

### Query Set (Generalization Evaluation)

Larger evaluation set (**64 samples**) to measure post-adaptation performance:
- `X_query`: (64, 4) pilots
- `Y_query`: (64, 1) received signals

Both come from the **same task's true channel**, ensuring the model learns the right task.

## Dataset Diversity

To ensure meta-learning benefits, tasks vary in:

| Parameter | Range | Effect |
|-----------|-------|--------|
| SNR | 5–20 dB | Difficulty: higher SNR = easier |
| Paths | 2–5 | Complexity: more paths = harder |
| Noise | 0.05–0.2 | Measurement uncertainty |

**Total:** 100 training tasks + 20 test tasks (120 different channel realizations)

## Algorithm

### MAML (Model-Agnostic Meta-Learning)

Plain English:

1. **Start** with shared weights θ (the meta-model)
2. **For each task** in the batch:
   - Take 5 gradient steps on support set → adapted weights θ'
   - Compute loss on query set using θ'
3. **Update θ** so that taking those few adaptation steps gives good query performance
4. **Repeat** for many iterations

This creates an initialization that **learns how to adapt quickly**.

### Baseline (For Comparison)

Train an independent model from scratch on each task's support set for 200 steps. Shows how well you can do without meta-learning.

**Meta-learning wins when:** Small support set + many diverse tasks → model learns generalizable adaptation strategy.

## Training Details

**MAML hyperparameters:**
- Inner learning rate: 0.01 (adaptation step size)
- Outer learning rate: 0.001 (meta-update)
- Inner steps: 5 (gradient steps per task)
- Batch size: 4 tasks
- Iterations: 50

**Network:**
- Input: 4 (pilot features)
- Hidden: 64 → 32 neurons
- Output: 1 (channel estimate)
- Activation: ReLU
- Loss: MSE

## Results

After training, `plot_loss.png` shows:

- **Top plot**: Query loss (generalization) – measure of how well adapted model performs
- **Bottom plot**: Support loss (fitting) – how well model fits the adaptation data

Comparison:
- **MAML**: Learns meta-initialization → adapts quickly with few steps
- **Baseline**: Trains from scratch → requires more data or steps to adapt
Expected behavior:
- MAML query loss should decrease with meta-learning iterations
- Baseline query loss comes from training 200 steps per task (more stable but worse at very few samples)

## README: Parts 1–6 (Quick Reference)

Part 1 — What did you build?

This repository implements a meta-learning approach (MAML) for wireless channel estimation. The goal is to learn an initialization that adapts quickly to new channel realizations using a few pilot observations. A baseline model (train from scratch on each task's support set) is included for comparison.

Part 2 — How to set it up

Copy-paste commands to reproduce locally:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo/end_term
pip install -r requirements.txt
```

Part 3 — How to generate data

Run (one-step pipeline recommended):

```bash
# Generate data (use --seed to reproduce the same dataset)
python generate_data.py --seed 42
```

This creates synthetic wireless channel estimation tasks and saves them to `results/`:
- `train_tasks.npz`: training tasks (stacked arrays)
- `test_tasks.npz`: test tasks

By default `generate_data.py` uses a random seed (so each run produces new data). To reproduce an identical dataset use `--seed <INT>`.

Each task contains: `X_support` (8, 4), `Y_support` (8, 1), `X_query` (64, 4), `Y_query` (64, 1), plus `snr`, `num_paths`, `noise_scale`.

Part 4 — How to train

Run:

```bash
python train.py
```

Key settings used in the included run:
- Inner learning rate (adaptation): 0.01
- Outer learning rate (meta-update): 0.001
- Inner steps per task: 5
- Batch size: 4 tasks
- Meta-iterations: 50

Part 5 — How to test

Run:

```bash
python test.py
```

`test.py` does four things:
1. **Evaluates the trained meta-model** on 20 test tasks (5 inner gradient steps per task)
2. **Compares against baseline** (trains fresh model on support set for 200 steps)
3. **Generates both required plots:**
   - `results/plot_loss.png` - training loss curve
   - `results/plot_comparison.png` - MAML vs Baseline comparison
4. **Computes few-shot results** (5-shot and 20-shot scenarios)

Printed outputs include:
- Per-task results table with adapted/baseline loss and improvement %
- Summary averages
- Per-SNR performance breakdown
- Metric interpretation guide

Part 6 — Your results (actual numbers)

The table below compares average query MSE for 5-shot and 20-shot settings (lower is better):

| Method | 5-shot Error (MSE) | 20-shot Error (MSE) |
|--------|--------------------:|---------------------:|
| Baseline (from scratch) | 0.051593 | 0.004661 |
| MAML (meta-learned init) | 0.133398 | 0.242135 |

Notes:
- Numbers were computed locally using `compute_table_numbers.py` which runs 20 tasks for each setting and reports averaged MSE values.
- In this run the baseline outperformed MAML on average; this indicates the meta-training configuration may require tuning (more iterations, different inner/outer rates, or first-order MAML variants).

How to reproduce the few-shot table:

The 5-shot and 20-shot results are now computed automatically as part of `test.py`:

```bash
python train.py    # Train the meta-model
python test.py     # Run all evaluations, generate plots, and print few-shot table
```

The few-shot table will be printed in the console output under "Few-shot Results".

### Latest quantitative results (from most recent run)

The following numbers were produced by running `python train.py` and `python test.py` in this repository (local run):

- **Average Query Loss (MAML with adaptation):** 0.134141
- **Average Query Loss (Baseline trained on support):** 0.031457
- **Average Improvement:** -552.4% (baseline outperformed MAML in this run — see notes below)

Notes: the baseline here is a fresh model trained on each task's support set for 200 steps (same as in `train.py`). The negative improvement indicates the meta-learning configuration needs hyperparameter tuning or more meta-training iterations for this synthetic setup.

## Files
Evaluation, plotting, and few-shot computations
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── results/
    ├── train_tasks.npz    # Training tasks (generated)
    ├── test_tasks.npz     # Test tasks (generated)
    ├── plot_loss.png      # Training loss curve
    ├── plot_comparison.png # MAML vs Baseline comparison
    └── maml_model.pt      # Trained MAML weights
```

**Note:** `compute_table_numbers.py` and `plot_results.py` have been merged into `test.py` for a streamlined pipeline. results/
    ├── train_tasks.npz    # Training tasks (generated)
    ├── test_tasks.npz     # Test tasks (generated)
    ├── plot_loss.png      # Training curves
    └── maml_model.pt      # Trained MAML weights
```

## Reproducibility Checklist

✓ Channel model: Y = XH + noise (realistic wireless)  
✓ Task variability: SNR, paths, noise vary per task  
✓ Pilot signals: Randomly generated, consistent dims  
✓ Support/query split: Small support → large query  
✓ Consistency: Same structure across tasks, only params vary  
✓ Regression: Continuous channel estimation with MSE  
✓ Realism: Balanced difficulty (not trivial, not impossible)  
✓ Independence: Train/test split, no overlap  
✓ Dimensional clarity: Input (*, 4) → Output (*, 1)  
✓ Meta-learning fit: Few samples, many tasks, learns adaptation  

## Next Steps

### Improve MAML Performance
- Tune inner/outer learning rates
- Increase meta-training iterations
- Experiment with different inner step counts
- Use GPU for faster training

### Extend Dataset
- Vary input dimension (more/fewer subcarriers)
- Add different channel models (Rayleigh, Rician fading)
- Simulate more realistic noise patterns (colored noise, interference)

### Advanced Algorithms
- Implement Reptile (first-order MAML)
- Try FOMAML or other variants
- Multi-task learning baselines

## References

- Finn et al. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (ICML 2017)
- Wireless channel estimation fundamentals
- Few-shot learning with deep networks

---

**Status:** Initial implementation with generation and MAML training. Ready for hyperparameter tuning and extension.
