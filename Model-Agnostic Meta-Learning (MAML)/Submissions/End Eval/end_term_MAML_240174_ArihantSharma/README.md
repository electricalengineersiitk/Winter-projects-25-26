## MAML for Wireless Channel Estimation

## Part 1 — What did I build?

This project applies **Model-Agnostic Meta-Learning (MAML)** to the problem of **wireless channel estimation** in OFDM systems. A shared neural network is meta-trained across 100 synthetic wireless environments (tasks), each with different SNR levels and multipath conditions, so that at test time it can adapt to a brand-new, unseen environment using only a handful of pilot observations — without retraining from scratch.

## Part 2 — How to set it up

```bash
## Part 2 — How to set it up

git clone https://github.com/arihant174/Winter-projects-25-26.git
cd Winter-projects-25-26/"Model-Agnostic Meta-Learning (MAML)"/Submissions/"End Eval"/end_term
pip install -r requirements.txt

## Part 3 — How to generate data

```bash
python generate_data.py
```

This script creates two `.npz` files inside a `data/` folder: `train_tasks.npz` (100 tasks) and `test_tasks.npz` (20 tasks). Each task represents one wireless environment with a unique SNR (0–20 dB) and number of multipath components (2–8). Every task contains a **support set** (10 pilot/channel pairs for adaptation) and a **query set** (80 pairs for evaluation). Inputs are complex pilot observations stacked as real vectors; labels are the true channel coefficients.

## Part 4 — How to train

```bash
python train.py
```

Key settings (edit at top of `train.py`):

| Parameter | Value |
|---|---|
| Meta learning rate | `1e-3` (Adam) |
| Inner learning rate | `0.01` (SGD) |
| Inner adaptation steps | `5` |
| Meta-training iterations | `500` |
| Tasks per meta-update | `8` |
| Algorithm | MAML (set `USE_REPTILE = True` for Reptile) |

The script prints loss every 10 iterations and saves the model to `results/meta_model.pt` and the loss curve to `results/plot_loss.png`.

## Part 5 — How to test

```bash
python test.py
```

The script loads the meta-trained model, adapts it on each of the 20 unseen test tasks for 1 / 3 / 5 / 10 / 20 gradient steps, and evaluates NMSE on the query set. It also trains a baseline model from scratch for comparison. A table is printed to the terminal and a comparison plot is saved to `results/plot_comparison.png`.

## Part 6 — Results


| Method | 5-step NMSE | 20-step NMSE |
|---|---|---|
| Baseline (from scratch) | −5.2 dB | −10.8 dB |
| **MAML (meta-trained)** | **−13.4 dB** | **−19.7 dB** |

Lower NMSE (dB) = better channel estimation accuracy. MAML reaches higher accuracy with far fewer adaptation steps because its initialisation is already close to a good solution for any task in the distribution.
