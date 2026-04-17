# Detecting Illicit Bitcoin Transactions using Graph Neural Networks

## Author
Manish Kajla (240622)

## Project Overview

In modern financial systems, especially in cryptocurrency networks like Bitcoin, fraudulent activities such as money laundering are not isolated events. They often occur in the form of coordinated patterns or networks of transactions.

Traditional machine learning models treat each transaction independently, which makes it difficult to detect such coordinated fraud. This project explores a better approach — **Graph Neural Networks (GNNs)** — which can naturally model relationships between transactions.

In this project, we treat the entire Bitcoin transaction system as a **graph**, where:
- Each node represents a transaction  
- Each edge represents the flow of money between transactions  

The goal is to classify each transaction as:
- **Licit (0)** → Normal transaction  
- **Illicit (1)** → Fraudulent transaction  


## Objectives

The main goals of this project were:

- To understand how fraud detection can be modeled as a **graph-based problem**
- To implement and compare different GNN architectures:
  - GCN (Graph Convolutional Network)
  - GraphSAGE
  - GAT (Graph Attention Network)
- To compare GNN models with a traditional ML model (XGBoost)
- To analyze whether GNNs can learn structural patterns automatically
- To improve interpretability using **GNNExplainer**

## Dataset

We used the **Elliptic Bitcoin Transaction Dataset**, which is a benchmark dataset for fraud detection.

Key details:
- ~203,000 transactions (nodes)
- ~234,000 connections (edges)
- 166 features per transaction
- Only ~2% transactions are fraudulent (high class imbalance)

This dataset is challenging because:
- It is highly imbalanced  
- Many nodes are unlabeled  
- Fraud patterns evolve over time  


## Methodology

### 1. Data Preprocessing
- Merged multiple dataset files (features, edges, labels)
- Converted transaction IDs into numerical indices
- Normalized features for stable training
- Split labeled data into train, validation, and test sets

### 2. Handling Class Imbalance

Since fraudulent transactions are very rare, we used **Focal Loss**, which:
- Reduces importance of easy (majority) examples  
- Focuses more on difficult (minority) examples  

This helps improve detection of illicit transactions.


### 3. Models Implemented

#### GCN (Graph Convolutional Network)
- Aggregates information from neighboring nodes
- Simple but limited (cannot handle unseen data well)

#### GraphSAGE (Best Model)
- Samples neighbors instead of using full graph
- Can generalize to new, unseen transactions
- Best suited for real-world applications

#### GAT (Graph Attention Network)
- Uses attention mechanism
- Assigns importance to different neighbors
- More complex but unstable in this dataset

#### XGBoost (Baseline)
- Traditional machine learning model
- Uses tabular features only (no graph structure)


### 4. Ablation Study

To test whether GNNs actually learn structure:

- Removed 72 engineered structural features  
- Trained GraphSAGE again  

Result: Only a small drop in performance  

This proves that:
> GNNs can automatically learn structural relationships from the graph itself.


### 5. Explainability

We used **GNNExplainer** to understand model decisions.

It helps identify:
- Important features  
- Important neighboring transactions  

This is crucial for real-world applications like fraud detection, where explanations are necessary.


## Results

| Model       | Macro F1 | PR-AUC |
|------------|----------|--------|
| XGBoost     | 0.98     | 0.98   |
| GraphSAGE   | 0.92     | 0.92   |
| GCN         | 0.82     | 0.76   |
| GAT         | 0.80     | 0.63   |

### Key Observations:
- GraphSAGE performed best among GNN models  
- XGBoost performed better overall due to engineered features  
- GNNs still proved powerful in capturing graph structure  


## Key Insights

- Fraud detection is fundamentally a **network problem**, not just a classification problem  
- GNNs eliminate the need for heavy manual feature engineering  
- GraphSAGE is ideal for real-time fraud detection systems  
- Interpretability (via GNNExplainer) is critical for financial applications  


## How to Run the Project

### 1. Install dependencies
```bash
pip install torch torch-geometric scikit-learn pandas numpy matplotlib seaborn xgboost
