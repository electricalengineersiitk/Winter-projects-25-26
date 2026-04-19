# EEG Brain Speller Project (P300 Paradigm)

## 1. Project Overview
This project implements an EEG-based brain speller, a Brain-Computer Interface (BCI) system that allows a user to type characters using brain signals (P300 ERP component) with no physical movement.

## 2. Signal Processing Pipeline
I have implemented the pipeline as follows to ensure high signal quality and classification accuracy:

* **Preprocessing:** * Applied a 0.1–30 Hz bandpass filter to isolate P300 frequencies.
    * Notch filter at 50 Hz to remove power line noise.
    * Average re-referencing for improved signal-to-noise ratio (SNR).
* **Artifact Rejection:** Used Independent Component Analysis (ICA) to identify and remove eye-blink and muscle artifacts.
* **Epoching:** Extracted epochs locked to stimulus onset (-200 ms to +800 ms) with baseline correction.
* **Feature Extraction:** * Downsampled data to reduce feature dimensionality.
    * Implemented **Xdawn spatial filtering** (with stable 'lwf' estimator) to enhance the P300 ERP component.
* **Classification:** Used a Linear Discriminant Analysis (LDA) classifier within a Scikit-Learn pipeline.

## 3. Results
The model was evaluated using a held-out test set (20% split) with the following performance metrics:

| Metric | Value |
| :--- | :--- |
| **Classification Accuracy** | 93.00% |
| **Information Transfer Rate (ITR)** | 133.35 bits/minute |

### Confusion Matrix
(The confusion matrix generated in the notebook shows high precision for both Target and Non-Target classes, with minimal false positives.)

## 4. Environment & Dependencies
* Python 3.10+
* Libraries: `mne`, `moabb`, `numpy`, `scikit-learn`, `pyriemann`, `matplotlib`

## 5. Instructions to Run
1. Open the provided `.ipynb` file in Google Colab.
2. Run the first cell to install all required dependencies.
3. Execute the cells sequentially to download the BNCI2014_009 dataset and process the data.
