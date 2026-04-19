# Meta-Learning for Wireless Localization

## 1. What did you build?

We implemented a Model-Agnostic Meta-Learning (MAML) framework for wireless localization.
The model predicts user positions from signal features and learns an initialization that can quickly adapt to new environments using only a few samples (few-shot learning).

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

This script generates synthetic wireless localization tasks.
Each task represents a different environment with randomly placed anchors.

Each task contains:

* **Support set (10 samples)** → used for adaptation
* **Query set (50 samples)** → used for evaluation

Generated files:

```
data/train.npy
data/test.npy
```

---

## 4. Train

```bash
python train.py
```

This performs meta-training using MAML.

### Key hyperparameters:

* Inner learning rate: **0.005**
* Outer learning rate: **0.001**
* Inner steps: **5**
* Meta iterations: **150**

During training:

* Loss decreases over iterations
* Model learns a good initialization for fast adaptation

Training curve is saved as:

```
results/plot_loss.png
```

---

## 5. Test

```bash
python test.py
```

This evaluates performance on unseen tasks.

It computes:

* MAML performance after adaptation (1, 3, 5 steps)
* Baseline performance (trained from scratch)

Comparison plot is saved as:

```
results/plot_comparison.png
```

---

## 6. Results

| Method         | Error |
| -------------- | ----- |
| Baseline       | ~0.84 |
| MAML (1 step)  | ~0.63 |
| MAML (3 steps) | ~0.43 |
| MAML (5 steps) | ~0.38 |

### Observations:

* MAML significantly outperforms the baseline
* Error decreases as the number of adaptation steps increases
* MAML adapts quickly using very few samples
* Baseline requires more training and performs worse

---

## 7. Plots

### Training Loss

* File: `results/plot_loss.png`
* Shows steady decrease in meta-training loss

### MAML vs Baseline

* File: `results/plot_comparison.png`
* Demonstrates that MAML achieves lower error than baseline

---

## 8. Project Structure

```
end_term/
├── README.md
├── requirements.txt
├── generate_data.py
├── train.py
├── test.py
├── data/
│   ├── train.npy
│   └── test.npy
├── models/
│   └── maml.pth
└── results/
    ├── plot_loss.png
    └── plot_comparison.png
```

---

## 9. Key Concepts

* **Meta-Learning**: Learning how to learn across tasks
* **MAML**: Learns an initialization that adapts quickly
* **Support Set**: Used for adaptation
* **Query Set**: Used for evaluation
* **Few-Shot Learning**: Learning from limited data

---

## 10. Conclusion

This project demonstrates that meta-learning enables fast adaptation in wireless localization tasks.
Compared to a baseline model trained from scratch, MAML achieves lower error using only a few gradient updates, making it suitable for dynamic environments.

---

## 11. Bonus Insight

Reptile (a first-order meta-learning method) was also explored conceptually as a simpler alternative to MAML.