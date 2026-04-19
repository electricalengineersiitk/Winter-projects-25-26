# Illicit Transaction Detection in Bitcoin using Graph Neural Networks

> Detecting financial fraud on the Bitcoin blockchain by modeling transaction flows as a graph and leveraging GCN, GraphSAGE, and GAT architectures with temporal adaptation strategies.

---

## Team Members

| Name |
|------|
| Harshit |
| Shivam |
| Harsh |
| Ojas |
| Dhruv |
| Lakshay |
| Chirag |
  Manthan |
  Sarthak |


---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Architectures](#model-architectures)
5. [Training Strategy](#training-strategy)
6. [Ablation Study](#ablation-study)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Results](#results)
9. [Explainability — GNNExplainer & Attention Visualization](#explainability)
10. [Rolling Window & Temporal Adaptation](#rolling-window--temporal-adaptation)
11. [Libraries & Dependencies](#libraries--dependencies)
12. [Conclusions & Limitations](#conclusions--limitations)
13. [References](#references)

---

## Project Overview

The goal of this project is to **identify illicit (money-laundering) transactions** within the Bitcoin network using Graph Neural Networks. The problem presents two core challenges:

- **Severe class imbalance**: only ~9.8% of labeled transactions are illicit.
- **Temporal concept drift**: the proportion of illicit activity fluctuates drastically across 49 time steps, making static train/test splits unreliable.

To address these challenges, three GNN architectures were benchmarked — GCN, GraphSAGE, and GAT — augmented with Focal Loss, threshold calibration, an ablation study on feature importance, GNNExplainer-based interpretability, and a rolling window training strategy with a historical replay buffer.

---

## Dataset

The project uses the publicly available **[Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)** — a real-world, timestamped Bitcoin transaction graph.

| Property | Value |
|---|---|
| Nodes (Transactions) | 203,769 |
| Edges (Transaction Flows) | 234,355 directed |
| Features per Node | 165 (dense) |
| Timesteps | 49 |
| Illicit Nodes | 4,545 (9.8% of labeled) |
| Licit Nodes | 42,019 (90.2% of labeled) |
| Unknown Nodes | 157,205 |

**File structure:**

```
elliptic_bitcoin_dataset/
├── elliptic_txs_features.csv   # 203,769 rows × 167 cols (txId + timestep + 165 features)
├── elliptic_txs_edgelist.csv   # 234,355 directed edges (txId1 → txId2)
└── elliptic_txs_classes.csv    # Per-transaction label: 1=illicit, 2=licit, unknown
```

**Feature groups:**
- `f1 – f94`: Local transaction features (amounts, fees, time deltas, etc.)
- `f95 – f165`: Aggregated structural features (neighborhood-level statistics pre-computed by the dataset authors)

---

## Data Preprocessing

All preprocessing steps are implemented in the notebook and produce two PyTorch Geometric `Data` objects saved as `.pt` files.

### Steps

1. **Label Encoding**
   - `1` (illicit) → `1`
   - `2` (licit) → `0`
   - `unknown` → `-1` (masked out from loss computation)

2. **Feature Normalization**
   - All 165 features scaled to zero mean and unit variance using `StandardScaler`.
   - A separate scaler was applied to the local-only 94-feature matrix for the ablation study.

3. **Graph Pruning**
   - Edge list filtered to retain only node connections present in the parsed feature matrix, ensuring a clean `edge_index` tensor.

4. **Temporal Train / Val / Test Split**
   - Timesteps 1–34 → Training (80%) and Validation (20%)
   - Timesteps 35–49 → Held-out Test set
   - Only labeled nodes (illicit + licit) contribute to loss via masking.

5. **Data Packaging**
   - `elliptic_full.pt` — full 165-feature graph
   - `elliptic_ablated.pt` — local-only 94-feature graph (for ablation)

```python
data_full = Data(
    x=x_full,           # (203769, 165) normalized features
    edge_index=edge_index,
    y=labels,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask
)
```

---

## Model Architectures

### 1. GCN — Graph Convolutional Network

A 2-layer spectral GCN acting as the baseline.

```
GCNConv(165 → 128) → ReLU → Dropout(0.3)
GCNConv(128 → 2)   → ReLU → Dropout(0.3)
                   → log_softmax
```

| Hyperparameter | Value |
|---|---|
| Hidden Dim | 128 |
| Dropout | 0.3 |
| Optimizer | Adam (lr=1e-3, wd=5e-4) |
| Scheduler | ReduceLROnPlateau |
| Loss | FocalLoss (α=0.85, γ=2.0) |
| Max Epochs | 5000 |
| Early Stopping Patience | 500 |

---

### 2. GraphSAGE — Scalable Inductive GNN

A 2-layer inductive GNN designed for scalability, using mean aggregation over sampled neighborhoods.

```
SAGEConv(165 → 140, aggr='mean') → ReLU → Dropout(0.1)
SAGEConv(140 → 103, aggr='mean') → ReLU → Dropout(0.1)
Linear(103 → 2) [Xavier init]    → log_softmax
```

| Hyperparameter | Value |
|---|---|
| Hidden Dim | 140 |
| Embedding Dim | 103 |
| Dropout | 0.1 |
| Aggregator | Mean |
| Classifier Head | Linear + Xavier init |
| Optimizer | Adam (lr=1e-3, wd=5e-4) |
| Loss | FocalLoss (α=0.85, γ=2.0) |
| Max Epochs | 5000 |
| Early Stopping Patience | 500 |

> **Note:** The architecture supports mini-batch training via `NeighborLoader` (num_neighbors=[10,5], batch_size=512) on systems with `pyg-lib` / `torch-sparse` installed.

---

### 3. GAT — Graph Attention Network

A 2-layer attention-based GNN enabling differential weighting of neighbors, with interpretable attention scores per edge.

```
GATConv(165 → 64, heads=4, concat=True)   → ELU → Dropout
GATConv(256 → 2,  heads=1, concat=False)  → log_softmax
```

| Hyperparameter | Value |
|---|---|
| Hidden Channels | 64 |
| Layer 1 Heads | 4 (concat → 256) |
| Layer 2 Heads | 1 |
| Optimizer | Adam (lr=1e-3, wd=1e-3) |
| Loss | FocalLoss (α=0.85, γ=2.0) |
| Max Epochs | 5000 |
| Early Stopping Patience | 500 |

The attention weights `α_ij ∈ (0,1)` learned per edge provide direct interpretability: high-weight edges on an illicit node reveal which neighboring transactions the model considers most suspicious.

---

## Training Strategy

### Focal Loss

Standard cross-entropy drastically under-penalizes misclassification of the minority illicit class. Focal Loss addresses this:

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```

- `α = 0.85` — illicit nodes receive 85% of the loss signal weight
- `γ = 2.0` — down-weights easy, confidently-correct examples to focus on hard cases

### Threshold Calibration

Rather than a naïve 0.5 decision boundary, probability thresholds are swept across the validation set (range 0.05–0.95 in steps of 0.01) to find the threshold that maximizes illicit-class F1.

### Early Stopping

Training is halted when validation F1 has not improved for `patience=500` epochs, with the best model checkpoint saved throughout.

---

## Ablation Study

**Goal:** Determine whether the 71 pre-computed *structural* features (f95–f165) add genuine value, or whether the GNN already learns equivalent structural representations from the edge structure alone.

| Variant | Features | Source File |
|---|---|---|
| Full | 165 (local + structural) | `elliptic_full.pt` |
| Ablated | 94 (local only) | `elliptic_ablated.pt` |

Both variants of GraphSAGE were trained identically and compared on the held-out test set. The ablation revealed only a **minor drop** in F1 when structural features were removed — confirming that the GNN's message-passing mechanism sufficiently learns cluster-level topology from the graph edges themselves, and the pre-computed structural features are partially redundant.

---

## Hyperparameter Tuning

A grid search was performed over learning rate and hidden dimension, holding all other parameters constant (FocalLoss α=0.75, γ=2, patience=20, max 150 epochs for sweep speed).

| LR | Hidden Dim | Best Val F1 | Test F1 | Test PR-AUC |
|---|---|---|---|---|
| 1e-3 | 128 | ↑ best | ↑ best | ↑ best |
| 1e-3 | 64 | — | — | — |
| 5e-4 | 128 | — | — | — |
| 5e-4 | 64 | — | — | — |

**Conclusion:** GraphSAGE with `lr=1e-3` or `5e-4` and `hidden_dim=128` consistently provided the best recall-weighted performance.

---

## Results

Test set metrics (timesteps 35–49), fixed threshold = 0.5:

| Model | F1 (Illicit) | Precision | Recall | PR-AUC |
|---|---|---|---|---|
| **GCN** | ~0.46 | ~0.37 | ~0.60 | ~0.48 |
| **GraphSAGE** | **~0.58** | **~0.54** | **~0.64** | **~0.60** |
| **GAT** | ~0.55 | ~0.60 | ~0.50 | ~0.55 |

- **GraphSAGE** achieves the best overall F1 and recall, making it the best model for maximizing illicit transaction detection.
- **GAT** achieves the highest precision but lower recall — fewer false positives at the cost of missing more illicit transactions.
- **GCN** shows the highest raw recall but poor precision, reflecting aggressive classification of the minority class.
- Rolling Window training (see below) significantly boosts recall beyond all static baselines by allowing the model to adapt to distribution shifts in real time.

---

## Explainability

### GAT Attention Visualization

After training, attention weights `α_ij` are extracted from the final GAT layer across the entire graph. For predicted-illicit nodes in the test set, a 2-hop subgraph is extracted and edges are visualized with width/opacity proportional to attention score — visually surfacing the most suspicious transaction links.

### GNNExplainer

GNNExplainer was applied to selected illicit nodes to generate influence subgraphs, identifying:
- Which local neighborhood edges most influenced the model's prediction.
- Top features by importance score per node.
- Confirmation that illicit nodes densely self-connect into tight fraudulent clusters.

---

## Rolling Window & Temporal Adaptation

To combat **concept drift** — the shifting nature of illicit activity across time steps — a **Rolling Window Incremental Trainer** was implemented:

1. The model is iteratively fine-tuned on each incoming timestep batch.
2. A **Replay Buffer** (`buffer_size=200`) retains a reservoir of historically seen illicit and licit examples.
3. Each training step combines current timestep data with replayed historical examples to prevent **catastrophic forgetting** while adapting to the new data distribution.

This approach was the primary driver of recall improvements above static baselines, as fixed models trained on timesteps 1–34 progressively degrade on the later test timesteps due to concept drift.

---

## Libraries & Dependencies

```bash
pip install torch torch_geometric scikit-learn matplotlib seaborn networkx tqdm psutil
```

| Library | Purpose |
|---|---|
| `torch` | Deep learning framework |
| `torch_geometric` | GNN layers (GCNConv, SAGEConv, GATConv), Data objects |
| `numpy`, `pandas` | Data manipulation |
| `scikit-learn` | StandardScaler, F1/Precision/Recall/PR-AUC, confusion matrix |
| `matplotlib`, `seaborn` | Visualization |
| `networkx` | Graph topology analysis |
| `tqdm` | Progress bars |
| `psutil` | System resource monitoring |

---

## Conclusions & Limitations

### Conclusions

- GNN architectures successfully detect illicit topological clusters in the Bitcoin transaction graph, significantly outperforming unstructured ML baselines on this domain.
- Focal Loss was critical in addressing the extreme class imbalance (~9.8% illicit).
- The Rolling Window + Replay Buffer strategy was the most impactful improvement for handling temporal concept drift.
- GNNExplainer confirmed that illicit nodes form dense self-connected clusters, validating the graph-based approach.

### Improvements Over Legacy Approaches

- **Temporal Rolling Window** replaces static train/test splits, allowing continual adaptation.
- **Historical Replay Buffer** prevents catastrophic forgetting during incremental fine-tuning.
- **Focal Loss** resolves the massive class imbalance that hampers standard cross-entropy training.
- **Threshold Calibration** replaces the naïve 0.5 boundary with a validation-optimized threshold.

### Limitations

- **Unknown nodes (~77%):** Standard supervised metrics mask all unknown-class nodes; transitioning to semi-supervised or self-supervised approaches could unlock this large unlabeled pool.
- **Threshold overfitting:** Calibrating thresholds on the validation set risks overfitting to the val distribution; unpredictable structural spikes in illicit transactions across future timesteps will degrade fixed thresholds.
- **Full-batch inference:** The current GCN/GAT implementations run full-batch forward passes, which may not scale to much larger graphs without mini-batch sampling.

---

## References

| Citation | Link |
|---|---|
| Weber et al. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with GCNs for Financial Forensics* | Dataset / Foundational Baseline |
| Kipf & Welling (2017). *Semi-Supervised Classification with GCNs* | GCN Architecture |
| Hamilton, Ying & Leskovec (2017). *Inductive Representation Learning on Large Graphs* | GraphSAGE Architecture |
| Veličković et al. (2018). *Graph Attention Networks* | GAT Architecture |
| Ying et al. (2019). *GNNExplainer: Generating Explanations for GNNs* | Explainability |

---
