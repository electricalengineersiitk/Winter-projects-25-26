# Leveraging Graph Neural Networks to Detect Illicit Financial Transactions

## Project Overview
This project applies **Graph Neural Networks (GNNs)** to detect illicit transactions in financial networks. Unlike traditional machine learning models that treat transactions independently, this approach models the **relational structure** of transactions as a graph to uncover hidden fraud patterns.

The project is built on the **Elliptic Bitcoin Transaction Dataset**, a benchmark dataset for graph-based fraud detection.

---

## Problem Statement
Traditional fraud detection systems suffer from:
- Ignoring relationships between transactions  
- Heavy reliance on manual feature engineering  
- High false positive rates  

This project solves these issues by:
- Modeling transactions as a **graph**
- Using **GNN architectures** to learn from both features and network structure

---

## Methodology

### 1. Graph Construction
- Nodes → Transactions  
- Edges → Payment flows  
- Converted tabular data into graph format using **PyTorch Geometric**

### 2. Feature Engineering
- 166 features per node:
  - 94 local features  
  - 72 aggregated structural features  

### 3. Models Implemented
- **GCN (Graph Convolutional Network)** – baseline model  
- **GraphSAGE** – inductive learning & scalability  
- **GAT (Graph Attention Network)** – weighted neighbor importance  

### 4. Training Strategy
- Semi-supervised learning (many unlabeled nodes)
- Class imbalance handled using:
  - Weighted loss / Focal loss
- Temporal splits used for realistic evaluation

---

## Dataset Details

- **Dataset**: Elliptic Bitcoin Dataset  
- **Nodes**: ~203K transactions  
- **Edges**: ~234K connections  
- **Classes**:
  - 2% Illicit  
  - 21% Licit  
  - 77% Unknown  

---

## Evaluation Metrics

Due to class imbalance, accuracy is not reliable. We use:

- **F1 Score (Macro)**  
- **Precision-Recall AUC (PR-AUC)**

---

## Results

- GNN-based models outperform traditional ML approaches  
- Better detection of fraud rings via structural learning  
- Reduced false positives compared to tabular models  


---

## Tech Stack

### Libraries Used
- Python  
- PyTorch  
- PyTorch Geometric  
- NumPy  
- Pandas  
- Matplotlib / Seaborn  
- Scikit-learn  

---

## How to Run

### 1. Install dependencies
```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install pandas numpy scikit-learn matplotlib
```
---
## References
### Dataset
- **Elliptic Dataset**  
  https://www.elliptic.co/resources/elliptic-dataset-cryptocurrency-transactions  

---

### Research Papers
- **Kipf, T. N., & Welling, M. (2017)**  
  *Semi-Supervised Classification with Graph Convolutional Networks*  
  https://arxiv.org/abs/1609.02907  

- **Hamilton, W. L., Ying, R., & Leskovec, J. (2017)**  
  *Inductive Representation Learning on Large Graphs (GraphSAGE)*  
  https://arxiv.org/abs/1706.02216  

- **Veličković, P., et al. (2018)**  
  *Graph Attention Networks (GAT)*  
  https://arxiv.org/abs/1710.10903  

- **McAuley, J., et al. / Elliptic (2019)**  
  *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks*  
  https://arxiv.org/abs/1908.02591  

---

### Libraries & Documentation
- **PyTorch Documentation**  
  https://pytorch.org/docs/stable/index.html  

- **PyTorch Geometric Documentation**  
  https://pytorch-geometric.readthedocs.io  

- **Scikit-learn Documentation**  
  https://scikit-learn.org/stable/  

- **Pandas Documentation**  
  https://pandas.pydata.org/docs/  

- **NumPy Documentation**  
  https://numpy.org/doc/  