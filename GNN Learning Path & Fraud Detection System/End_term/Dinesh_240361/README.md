# Illicit Transaction Detection using Random Forest + GNN

## Overview
This project implements a hybrid model to detect illicit Bitcoin transactions using the Elliptic dataset. The approach combines a Random Forest model with a Graph Neural Network (GCN). The Random Forest captures patterns in transaction features, while the GNN captures relationships between transactions in the graph.

### Data Loading
The dataset is loaded from three CSV files. The features file contains transaction features, the classes file contains labels, and the edgelist file defines the connections between transactions. Each transaction initially has a unique ID which is not directly usable by the model.

### Feature Processing
The first step in the code maps transaction IDs to integer indices. This is required because graph-based frameworks expect nodes to be indexed numerically. After creating this mapping, the transaction ID column is removed and only numerical features are retained.

These features are converted into a tensor so they can be used by the neural network.

### Label Processing
The labels in the dataset are given as:
- 1 for illicit
- 2 for licit
- unknown for unlabeled data

In the code, these are converted into:
- 1 for illicit
- 0 for licit
- -1 for unknown

The unknown nodes are not used during training, but they are still included in the graph so that they can contribute to neighborhood information.

### Random Forest Model
A Random Forest classifier is trained using only the labeled nodes. This model learns patterns directly from the feature values of transactions.

After training, the model outputs a probability score for each transaction indicating how likely it is to be illicit. This probability is not the final prediction but is used as an additional feature.

### Feature Augmentation
The probability output from the Random Forest is appended to the original feature vector of each node. This creates a richer representation that combines raw features with learned predictions.

This step is important because it injects feature-based knowledge into the graph model.

### Graph Construction
The graph is constructed using the edgelist file. Each row represents a connection between two transactions.

Using the previously created mapping, transaction IDs are converted into indices and stored in an edge_index tensor. This tensor defines how nodes are connected in the graph.

A data object is then created containing:
- Node features
- Edge connections
- Labels

### Train-Test Split
Since many nodes are unlabeled, only labeled nodes are used for training and evaluation. The code creates two masks:
- train_mask for training nodes
- test_mask for evaluation nodes

This ensures the model does not train on unknown labels.

### GNN Model Definition
The Graph Neural Network used is a Graph Convolutional Network (GCN). It consists of two layers.

The first layer transforms input features into a hidden representation. The second layer produces output scores for classification.

Each node updates its representation by aggregating information from its neighbors. This allows the model to learn patterns based on both node features and graph structure.

### Model Training
The model is trained using cross-entropy loss and the Adam optimizer.

During each epoch:
- A forward pass is performed on the graph
- Loss is calculated using only training nodes
- Backpropagation updates the model parameters

Over multiple epochs, the model learns to distinguish between licit and illicit transactions.

### Evaluation
After training, predictions are made for the test nodes.

The performance is evaluated using F1 score, which is more suitable than accuracy because the dataset is highly imbalanced.

## Key Idea Behind the Code
The core idea is to combine two types of learning:

- Random Forest learns patterns from transaction features
- GNN learns patterns from transaction relationships

By adding the Random Forest output as a feature, the GNN benefits from both perspectives.

## Results
The model achieves an F1 score of approximately 0.87, showing strong performance for fraud detection.

## Conclusion
This implementation demonstrates that combining traditional machine learning with graph-based learning significantly improves performance in detecting illicit transactions. The approach effectively captures both local feature patterns and global network structure.