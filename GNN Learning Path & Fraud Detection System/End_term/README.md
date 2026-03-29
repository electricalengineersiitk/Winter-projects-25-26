# Illicit Transaction Detection using GNN + Random Forest

## Overview
This project aims to detect illicit (fraudulent) Bitcoin transactions using a hybrid machine learning approach that combines Random Forest and Graph Neural Networks (GNNs).

Traditional models analyze transactions independently, but fraud often occurs in interconnected networks. This project models transactions as a graph to capture relational patterns.

---

## Problem Statement
Financial fraud detection systems based on traditional machine learning suffer from key limitations:

- They assume transactions are independent
- They fail to capture network relationships
- They require heavy manual feature engineering
- They produce high false positive rates

To overcome these issues, this project uses Graph Neural Networks to incorporate structural information from transaction networks :contentReference[oaicite:0]{index=0}.

---

## Dataset
The project uses the Elliptic Bitcoin dataset.

### Dataset Details
- Nodes: 203,769 transactions
- Edges: 234,355 transaction flows
- Features: 166 per node
  - 94 local features
  - 72 structural features
- Labels:
  - 1 → Illicit
  - 2 → Licit
  - Unknown → Unlabeled
- Time steps: 49

The dataset is highly imbalanced, with only about 2% illicit transactions :contentReference[oaicite:1]{index=1}.

---

## Methodology

### 1. Data Preprocessing
- Map transaction IDs to integer indices
- Extract feature matrix
- Convert labels:
  - Illicit → 1
  - Licit → 0
  - Unknown → -1

---

### 2. Random Forest Model
- Train Random Forest on labeled data
- Learn patterns from tabular features
- Generate probability of each transaction being illicit

---

### 3. Feature Augmentation
- Append Random Forest probability to feature vector
- Final feature vector:
  original features + RF probability

---

### 4. Graph Construction
- Nodes represent transactions
- Edges represent transaction flows
- Create edge_index for graph structure

---

### 5. Graph Neural Network (GCN)
- Use Graph Convolutional Network
- Aggregate information from neighboring nodes
- Learn structural patterns of fraud

---

### 6. Training
- Train only on labeled nodes
- Use CrossEntropyLoss
- Optimize using Adam optimizer

---

### 7. Evaluation
Due to class imbalance, accuracy is not suitable.

Metrics used:
- F1 Score
- Precision
- Recall
- PR-AUC

---

## Results
- Achieved F1 Score ≈ 0.87
- Hybrid RF + GCN performs better than individual models
- Improved fraud detection performance

---

## Key Insights
- Random Forest captures feature-level patterns
- GNN captures graph relationships
- Combining both improves classification significantly

---

## Technologies Used
- Python
- PyTorch
- PyTorch Geometric
- Scikit-learn
- Pandas
- NumPy

---

## Project Pipeline
