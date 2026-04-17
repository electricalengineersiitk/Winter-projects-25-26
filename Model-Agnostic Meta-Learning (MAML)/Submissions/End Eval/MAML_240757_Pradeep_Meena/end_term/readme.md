# Meta-Learning for Wireless Signal Classification

**Name :** Pradeep Meena | **Roll Number:** 240757

---

## 1. What did you build?

We implemented a Model-Agnostic Meta-Learning (MAML) framework for wireless signal classification. The model classifies a received signal into its modulation type (BPSK, QPSK, or 16-QAM) and learns an initialization that can quickly adapt to new Signal-to-Noise Ratio (SNR) environments using only a few samples (few-shot learning).

---

## 2. Setup

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## 3. Generate Data

```bash
python generate_data.py
```

This script generates synthetic wireless signal classification tasks. Each task represents a different environment with varying SNR levels and randomly generated symbols.

Each task contains:
* **Support set (5 samples per class)** → used for adaptation
* **Query set (50 samples per class)** → used for evaluation

Generated files:

```text
data/meta_train_tasks.npz
data/meta_test_tasks.npz
```

---

## 4. Train

```bash
python train.py
```

This performs meta-training using MAML.

**Key hyperparameters:**
* Inner learning rate: 0.01
* Outer learning rate: 0.001
* Inner steps: 5
* Meta iterations: 200

**During training:**
* Loss decreases over iterations
* Model learns a good initialization for fast adaptation

Training curve is saved as:

```text
results/plot_loss.png
```

---

## 5. Test

```bash
python test.py
```

This evaluates performance on unseen tasks.

It computes:
* MAML performance after adaptation (0 to 5 steps)
* Baseline performance (trained from scratch up to 200 steps)

Comparison plot is saved as:

```text
results/plot_comparison.png
```

---## 6. Results

MAML significantly outperforms the baseline, adapting in just a few steps. (Lower error rate is better).

| Method | Error Rate |
| :--- | :--- |
| Baseline (from scratch, 200 steps) | 0.3997 |
| MAML (5 steps) | 0.1773 |

**Observations:**
* MAML significantly outperforms the baseline, achieving less than half the error rate.
* MAML adapts extremely quickly using very few samples (5 shots).
* The baseline requires massive amounts of training (200 shots) and still performs significantly worse in new environments.


## 7. Plots

**Training Loss**
* File: `results/plot_loss.png`
* Shows steady decrease in meta-training loss

**MAML vs Baseline**
* File: `results/plot_comparison.png`
* Demonstrates that MAML achieves lower error than baseline

---

## 8. Project Structure

```text
end_term/
├── README.md
├── requirements.txt
├── generate_data.py
├── train.py
├── test.py
├── data/
│   ├── meta_train_tasks.npz
│   └── meta_test_tasks.npz
├── models/
│   └── meta_model.pth
└── results/
    ├── plot_loss.png
    └── plot_comparison.png
```

---

## 9. Key Concepts

* **Meta-Learning:** Learning how to learn across tasks
* **MAML:** Learns an initialization that adapts quickly
* **Support Set:** Used for adaptation
* **Query Set:** Used for evaluation
* **Few-Shot Learning:** Learning from limited data

---

## 10. Conclusion

This project demonstrates that meta-learning enables fast adaptation in wireless signal classification tasks. Compared to a baseline model trained from scratch, MAML achieves lower error using only a few gradient updates, making it highly suitable for dynamic environments with varying noise profiles.
