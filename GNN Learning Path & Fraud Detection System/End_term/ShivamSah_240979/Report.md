# Comprehensive Report: Illicit Transaction Detection in Bitcoin using GNNs

## 1. Members of the Team
- **Harshit**
- **Shivam**
- **Harsh**
- **Ojas**
- **Dhruv**
- **Lakshay**
- **Chirag**
- **Manthan**
- **Sarthak**


## 2. Overall Project Summary
The primary objective of this project is to identify illicit transactions within the Bitcoin network using Graph Neural Networks (GNNs). The dataset exhibits extreme class imbalance (only ~9.8% of labeled transactions are illicit) and significant temporal concept drift, as the proportion of illicit activity fluctuates drastically across time. To address these challenges, we benchmarked three robust GNN architectures: a Baseline Graph Convolutional Network (GCN), a scalable GraphSAGE model, and an explainable Graph Attention Network (GAT).

Key innovations in our approach include using Focal Loss to combat class imbalance, conducting an ablation study to isolate the impact of structural features, utilizing GNNExplainer to visualize influence subgraphs, and employing a rolling window training strategy coupled with a replay buffer to incrementally adapt to distribution shifts over time.

## 3. Details of the Dataset
The analysis was performed on the **Elliptic Bitcoin Dataset**.
- **Nodes (Transactions):** 203,769
- **Edges (Transaction Flows):** 234,355
- **Features:** 165 dense features per node (94 local transaction features and 71 aggregated structural features).
- **Temporal Aspect:** 49 distinct timesteps, showcasing visible concept drift.
- **Classes (Labels):**
  - **Unknown:** 157,205
  - **Licit (Class 2):** 42,019 (90.2% of labeled nodes)
  - **Illicit (Class 1):** 4,545 (9.8% of labeled nodes)

## 4. Details of Data Preprocessing
- **Encoding:** String classes mapped to integers (Illicit=1, Licit=0, Unknown=-1).
- **Normalization:** The features were scaled to have zero mean and unit variance using `StandardScaler`. This was done for both the full 165-feature matrix and the local-only 94-feature matrix.
- **Graph Pruning:** The edge list was filtered to strictly maintain node connections existing within our parsed feature matrix.
- **Temporal Train/Test Split:** 
  - Timesteps 1 to 34 were primarily allocated for Training (80%) and Validation (20%).
  - Timesteps 35 to 49 were designated as the held-out Test set.
- **Masking:** Only labeled nodes (Illicit and Licit) contributed to the training loss. "Unknown" nodes were structurally present but masked out from classification targets.
- **Data Packaging:** Processed tensors were saved as internal PyTorch Geometric `Data` objects (`elliptic_full.pt` and `elliptic_ablated.pt`).

## 5. Feature Extraction and Weighing Methods
- **Focal Loss:** Standard cross-entropy loss drastically under-penalized the model for missing the minority illicit class. We implemented Focal Loss with `alpha` ranges (e.g., 0.75 - 0.85) to weigh illicit nodes heavily, and `gamma=2.0` to down-weight confident, easy-to-classify samples.
- **Ablation Study:** We manually segregated the feature space into *Local* (f1-f94) and *Structural* (f95-f165) sets to ablate and test the GraphSAGE model's dependency on pre-calculated structural attributes.
- **Threshold Calibration:** Instead of a naïve 0.5 decision boundary, probability thresholds were swept dynamically across the validation set (e.g., picking 0.3 or similar parameters) to optimize for the `F1 (illicit)` score.

## 6. A List of All Libraries Used
- **Data processing & Manipulation:** `numpy`, `pandas`
- **Visualization:** `matplotlib`, `seaborn`, `networkx`
- **Deep Learning / Graph Analytics:** `torch` (PyTorch), `torch_geometric` (PyG)
- **Machine Learning Utilities:** `scikit-learn` (`StandardScaler`, `f1_score`, `precision_score`, `recall_score`, `average_precision_score`, `confusion_matrix`, `classification_report`)
- **System/Utilities:** `os`, `random`, `time`, `warnings`, `tqdm`, `psutil`

## 7. The Architecture of the Model Employed
Three model architectures were evaluated and optimized:
1. **GCN (Graph Convolutional Network):** A 2-layer implementation bridging node neighborhoods via spectral aggregations (`hidden_dim`=128, `dropout`=0.3), ending in a softmax output.
2. **GraphSAGE:** Designed for scalability, using 2 `SAGEConv` layers (`hidden_dim`=140, `embedding_dim`=103, `dropout`=0.1) alongside an aggregator mean function. It featured a linear output head randomly initialized via Xavier Initialization.
3. **GAT (Graph Attention Network):** Included multi-head attention allowing differential neighbor weighting (`hidden_channels`=64, `heads`=4 on the first layer, and 1 head on the classifier output layer).

For ultimate evaluation, a **Rolling Window Incremental Trainer** was built. It iteratively finetunes the base model on incoming timesteps, leveraging a Replay Buffer of `buffer_size=200` historically illicit and licit examples alongside current ones to avoid catastrophic forgetting while adjusting to concept drift.

## 8. The Final Results Achieved
Standard temporal test set (timesteps > 34) metrics without the rolling window evaluated strictly on Fixed baselines:
- **GCN:** F1 ~0.46, Precision ~0.37, Recall ~0.60, PR-AUC ~0.48
- **GraphSAGE:** F1 ~0.58, Precision ~0.54, Recall ~0.64, PR-AUC ~0.60
- **GAT:** F1 ~0.55, Precision ~0.60, Recall ~0.50, PR-AUC ~0.55

Hyperparameter tuning concluded that GraphSAGE with learning rates of `1e-3` or `5e-4` and hidden dimensions up to `128` provided maximized recall. Furthermore, the Ablation study revealed a minor drop in F1 when eliminating structural features, implying that while structural features provide extra non-graph signals, the GNN sufficiently learns cluster structures inherently through edge convolutions.

*Note: The comprehensive Rolling Window evaluation significantly boosted recall above general baseline thresholds by retraining on the fly.*

## 9. Conclusions, Improvements from Legacy Approaches, and Limitations
- **Conclusions:** The GNN architectures highly successfully discovered illicit topological clusters, demonstrating their prowess in the financial forensics domain over unstructured machine learning. The GAT network—visualized via GNNExplainer—extracted top features and verified that illicit nodes densely self-connect into fraudulent clusters. 
- **Improvements from Legacy Approaches:** Moving away from static Train/Test splits, the adoption of a Temporal Rolling Window alongside a Historical Replay Buffer was the ultimate catalyst for countering chronological concept drift. Applying Focal Loss significantly solved traditional machine learning shortfalls regarding the massive class disparity. 
- **Limitations:** The majority (~77%) of transactions are marked "Unknown." Standard supervised metrics simply ignore them; transitioning this pipeline completely into self-supervised or semi-supervised architectures could yield superior data mining. Furthermore, adapting validation thresholds against the future test set introduces risk, as unpredictable structural spikes in illicit transactions will degrade fixed-bound thresholds.

## 10. Sources
- **Dataset / Foundational Baseline:** Weber et al. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics*.
- **GCN Framework:** Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*.
- **GraphSAGE Framework:** Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs*.
- **GAT Framework:** Veličković, P., et al. (2018). *Graph Attention Networks*.
- **Explainability:** Ying, Z., et al. (2019). *GNNExplainer: Generating Explanations for Graph Neural Networks*.
