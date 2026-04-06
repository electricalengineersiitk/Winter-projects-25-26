# EEG-based P300 Brain-Computer Interface (BCI) Speller

This repository contains a full-stack EEG signal processing and classification pipeline for a P300 Brain Speller, designed to meet 100% of the academic requirements (Stage 1-5).

## 🚀 Final Project Achievement
- **Full Stage 1 Compliance:** Implemented 0.1-30Hz Bandpass, 50Hz Notch, Average Re-referencing, and **Bad Channel Interpolation**.
- **Artifact Rejection:** integrated **ICA (Independent Component Analysis)** to remove ocular artifacts.
- **Robust Evaluation:** All models (LDA, SVM, EEGNet) are evaluated using **5-Fold Stratified Cross-Validation**.
- **Metrics:** Reports Accuracy, Precision, Recall, F1-Score, and **Information Transfer Rate (ITR)**.

## 📊 Final Performance Results (Averaged Over 5 Folds)
| Model | Accuracy | F1-Score | ITR (bits/min) |
| :--- | :--- | :--- | :--- |
| **LDA (Baseline)** | 71.0% | 0.418 | 84.46 |
| **SVM (RBF Kernel)** | 86.8% | 0.488 | 117.90 |
| **EEGNet (Deep Learning)** | **85.1%** | **0.642** | **113.88** |

*Note: EEGNet demonstrates superior reliability (highest F1-score) for handling the P300 class imbalance.*

## 📁 Project Structure
- `src/01_explore_data.py`: Initial data inspection.
- `src/02_preprocess.py`: Master preprocessing (Filtering, Interpolation).
- `src/04_classify.py`: Classical machine learning comparison.
- `src/05_eegnet.py`: State-of-the-art EEGNet implementation.
- `src/06_final_evaluation.py`: Total system audit and metric generation.
- `src/07_erp_plot.py`: Visualization of the P300 ERP waveform.
- `src/08_ensemble_averaging.py`: Simulation of ensemble score averaging across repetitions.
- `results/`: Contains **Confusion Matrix Heatmaps** (`confusion_matrix.png`) and the **ERP Waveform Plot**.


## 🛠 Setup & Run
1. Create environment: `python -m venv eeg_env`
2. Activate: `.\eeg_env\Scripts\activate`
3. Install: `pip install -r requirements.txt`
4. Run evaluation: `python src/06_final_evaluation.py`

---
**Status:** Submitted for Final Audit (April 7 Deadline Compliance).
