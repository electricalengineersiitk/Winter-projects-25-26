# Meta-Learning for Wireless Systems (Channel Estimation)

## 1. What did I build?

This project implements a meta-learning model using **First-Order MAML (FOMAML)** for the task of **wireless channel estimation**.
The model learns a shared initialization that can quickly adapt to a new wireless environment using only a few samples.

---

## 2. How to set it up

```bash
git clone https://github.com/electricalengineersiitk/Winter-projects-25-26.git
cd "Winter-projects-25-26/Model-Agnostic Meta-Learning (MAML)/Submissions/End Eval/MAML_240882_RohithRajeevNambiar"
pip install -r requirements.txt
```

---

## 3. How to generate data

```bash
python generate_data.py
```

This script creates synthetic wireless tasks.
Each task simulates a different wireless environment with:

* Random channel (H)
* Random SNR (noise level)

Each task contains:

* Support set (k samples) for adaptation (Plots are drawn for k = 20, 20-shot)
* Query set (50 samples) for evaluation

It generates:

* 100 training tasks
* 20 test tasks

All data is saved in `data/tasks.npz`.

---

## 4. How to train

```bash
python train.py
```

This script:

* Trains a meta-learning model using FOMAML
* Uses:

  * Inner learning rate = 0.01
  * 3 inner steps
  * 200 meta-iterations

It also:

* Saves the trained model to `model.pth`
* Saves training loss plot to `results/plot_loss.png`

---

## 5. How to test

```bash
python test.py
```

This script:

* Evaluates MAML on unseen tasks
* Adapts using 1, 3, 5, and 10 steps
* Compares with baseline performance

It outputs:

* Average loss values
* Comparison plot saved as `results/plot_comparison_db.png`

---

## 6. Results

The model is evaluated using **NMSE (in dB)** across different adaptation steps for both 5-shot and 20-shot settings.

### 5-shot Results

| Adaptation Steps | MAML (dB) | Baseline (dB) |
| ---------------- | --------- | ------------- |
| 1                | 5.42      | 5.63          |
| 3                | 3.85      | 4.97          |
| 5                | 3.06      | 4.51          |
| 10               | 2.48      | 3.27          |

### 20-shot Results

| Adaptation Steps | MAML (dB) | Baseline (dB) |
| ---------------- | --------- | ------------- |
| 1                | 17.38     | 3.81          |
| 3                | -0.58     | 1.66          |
| 5                | -3.00     | -1.43         |
| 10               | -4.59     | -3.79         |

MAML achieves lower NMSE than the baseline in both 5-shot and 20-shot settings, demonstrating faster adaptation.

---

## Project Structure


```
end_term/
├── README.md
├── requirements.txt
├── generate_data.py
├── train.py
├── test.py
├── data/
│   └── tasks.npz
├── results/
│   ├── plot_loss.png
│   └── plot_comparison.png
├── model.pth
```

---

## Notes

* The model uses a simple 3-layer neural network (64 hidden units).
* Tasks simulate wireless environments using a linear channel model.
* MAML enables fast adaptation using very few samples compared to training from scratch.
