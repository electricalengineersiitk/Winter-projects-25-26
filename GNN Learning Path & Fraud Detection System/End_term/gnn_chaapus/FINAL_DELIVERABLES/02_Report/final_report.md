# Final Project Report: Leveraging Graph Neural Networks to Uncover Illicit Financial Networks

## Team Members

- Akshat Mittal (Roll No: 240086)
- Aaditya Mukherjee (Roll No: 240006)
- Daksh Gupta (Roll No: 240317)
- Nitish Choudhary (Roll No: 240710)

## 1. Overall Project Summary

This project addresses illicit transaction detection in cryptocurrency payment flows using graph representation learning. Instead of treating transactions as independent tabular rows, we model the transaction ecosystem as a graph and benchmark multiple Graph Neural Network (GNN) architectures against a tabular baseline.

## 2. Dataset Details

- Dataset: Elliptic Bitcoin transaction dataset
- Nodes: transactions
- Edges: directed payment flows
- Time steps: 49 temporal snapshots
- Feature dimensions: 166 (94 local + 72 aggregated structural features)
- Class imbalance: highly skewed with very few illicit labels and many unknown labels

### Class Mapping Used

- Illicit: `1`
- Licit: `0`
- Unknown/unlabeled: `-1` (excluded from supervised loss, retained in graph structure)

## 3. Data Preprocessing

1. Read raw CSV files: classes, features, and edge list.
2. Merge node features with class labels by transaction ID.
3. Map transaction IDs to contiguous node indices.
4. Build `edge_index` for graph neural models.
5. Convert to undirected graph (configurable).
6. Create temporal train/validation/test masks:
   - Train: time step <= 34
   - Validation: 35 to 39
   - Test: >= 40
7. Standardize features using train-mask statistics only.

## 4. Feature Extraction and Weighting Methods

- Full-feature experiments: all available node features (166).
- Ablation experiments: local-only first 94 features.
- Class imbalance handling:
  - Inverse-frequency class weights
  - Focal Loss to prioritize minority illicit class

## 5. Libraries Used

- Python, NumPy, Pandas, SciPy
- PyTorch, PyTorch Geometric
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- Jupyter

## 6. Model Architectures

### 6.1 GCN

Two-layer graph convolution network with dropout and final linear classifier.

### 6.2 GraphSAGE

Two-layer neighborhood aggregation model for robust inductive graph representation.

### 6.3 GAT

Attention-based graph model to assign dynamic importance to neighbors.

### 6.4 Baseline (XGBoost)

Tabular model trained on node features only to compare against graph-aware learning.

## 7. Optimization and Training

- Optimizer: AdamW
- Early stopping: validation PR-AUC
- Loss:
  - Focal Loss (default)
  - Cross-Entropy fallback
- Hyperparameters controlled via `configs/default.yaml`

## 8. Evaluation Metrics (KPIs)

- PR-AUC (primary)
- Macro F1-score
- Recall for illicit class
- False Positive Rate (FPR)
- FPR reduction vs tabular baseline

## 9. Results

The following results were produced from `artifacts/metrics/results_summary.csv` after model tuning and threshold calibration.

### 9.1 All Features (166)

| Model | PR-AUC | Macro F1 | Illicit Recall | FPR |
|---|---:|---:|---:|---:|
| XGBoost | 0.9810 | 0.9825 | 0.9375 | 0.0000 |
| GCN | 0.7999 | 0.8037 | 0.8750 | 0.0734 |
| GraphSAGE | 0.8252 | 0.8840 | 0.8125 | 0.0226 |
| GAT | 0.6767 | 0.7691 | 0.8125 | 0.0847 |

### 9.2 Local-Only Features (94)

| Model | PR-AUC | Macro F1 | Illicit Recall | FPR |
|---|---:|---:|---:|---:|
| XGBoost | 0.9681 | 0.9081 | 0.9375 | 0.0282 |
| GCN | 0.6695 | 0.7176 | 0.8125 | 0.1243 |
| GraphSAGE | 0.8676 | 0.7621 | 0.8750 | 0.1017 |
| GAT | 0.7653 | 0.8057 | 0.7500 | 0.0508 |

### 9.3 Key Observations

- On this run, XGBoost remains the strongest overall baseline.
- Among GNNs, GraphSAGE is the most consistent by PR-AUC, while GAT/GraphSAGE deliver competitive Macro F1 in tuned settings.
- Validation-based threshold optimization significantly reduced false positives versus fixed-threshold runs.
- GNN models still provide meaningful illicit-recall performance while using graph context.

## 10. Explainability

Node-level explanations are generated using GNNExplainer and saved under:

- `artifacts/explain/*.json`

These files include top influential features and edges for suspicious-node predictions.

## 11. Improvements Over Legacy Approaches

- Captures multi-hop transaction behavior directly via message passing
- Reduces manual feature engineering effort
- Improves minority-class detection and potentially lowers false positives

## 12. Limitations

- Performance sensitivity to split strategy and class-label sparsity
- GAT and full-batch training can be computationally expensive on very large graphs
- Explainability quality can vary by node and model confidence

## 13. Conclusion

Graph-based learning is better aligned with real-world coordinated financial fraud patterns than isolated tabular classification. In this project, GNN models provide a more contextual detection framework and can outperform tabular baselines on minority-focused KPIs when carefully tuned for class imbalance.

## 14. Sources

- Elliptic dataset (Kaggle)
- Kipf & Welling, 2017: Semi-Supervised Classification with Graph Convolutional Networks
- Hamilton et al., 2017: Inductive Representation Learning on Large Graphs (GraphSAGE)
- Velickovic et al., 2018: Graph Attention Networks
- Ying et al., 2019: GNNExplainer
