# Deep Model Compression on MNIST

## Overview

This repository implements a deep learning pipeline for model compression on the MNIST dataset using:

- Weight Pruning (removing small-magnitude weights)
- Weight Quantization (K-means clustering of weights)
- Compressed Storage (CSR sparse format)

The objective is to reduce model size and computation while maintaining good accuracy.


## What This Repo Does

- Trains a CNN model on MNIST  
- Applies global pruning to introduce sparsity  
- Applies quantization to reduce precision  
- Saves the model in a compressed sparse format  
- Allows loading the compressed model back  


## Project Structure

├── main.py 
├── config.py 

├── compression/
│ ├── init.py
│ ├── prune.py 
│ ├── quantization.py 
│ ├── linear.py 
│ └── conv2d.py 

├── data/
│ ├── init.py
│ └── data_loader.py 

├── models/
│ ├── init.py
│ └── mnist.py 

├── utils/
│ ├── training.py 
│ └── test_eval.py 


## How to Run

### 1. Install dependencies

pip install torch torchvision numpy scikit-learn scipy

Running the project

## ☁️ Running on Google Colab

You can run this project easily on Google Colab by uploading the project ZIP file.

### Steps:

1. Open Google Colab

2. Upload your project ZIP file:

from google.colab import files
uploaded = files.upload() 

!unzip project_final.zip

%cd project_final

!pip install torch torchvision numpy scikit-learn scipy

!python main.py

## Results ->

-> Baseline Model

Final Test Accuracy: 99.22%

-> After Pruning (98% Sparsity)

Final Test Accuracy: 98.59%

Sparsity Achieved: 98%

-> After Quantization (K = 4 clusters)

Final Test Accuracy: 99.22%

-> Compressed Model Performance

Reloaded Model Accuracy: 98.25%

# Model Size Comparison

Model Type	            File Size
Original (.pth)	        1.6701 MB
Compressed (.npz)	      1.1453 MB

# Compression Achieved: 1.46× reduction

# Resource Usage
GPU Memory Allocated: 26.83 MB
GPU Memory Reserved: 132.00 MB


