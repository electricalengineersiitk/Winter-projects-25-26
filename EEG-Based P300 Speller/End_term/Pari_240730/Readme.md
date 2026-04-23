#  EEG-Based P300 Speller — Methodology

##  Environment Setup & Imports
All required Python libraries are installed and imported at the beginning of the notebook:

- MNE  
- MOABB  
- scikit-learn  
- pyriemann  
- braindecode  
- PyTorch  

---

##  Data Loading & Exploration
- EEG data for **Subject 1** is loaded using MOABB.
- Initial visualization includes:
  - Raw EEG signal plots
  - Power Spectral Density (PSD)

These help assess signal quality and identify noise sources such as **50 Hz powerline interference**.

---

##  Preprocessing Pipeline
The raw EEG data undergoes the following preprocessing steps:

1. **Resampling**
   - Downsampled to **256 Hz** to reduce computational cost.

2. **Bandpass Filtering**
   - Frequency range: **0.1 – 30 Hz**
   - Focuses on relevant brain activity (P300 lies in low frequencies).

3. **Notch Filtering**
   - Applied at **50 Hz** to remove powerline noise.

4. **Re-referencing**
   - Converted to **common average reference (CAR)**.

5. **Bad Channel Detection & Interpolation**
   - Noisy channels are automatically detected and corrected.

6. **ICA Artifact Removal**
   - Independent Component Analysis (ICA) removes:
     - Eye movement artifacts
     - Muscle artifacts

---

##  Epoching & Baseline Correction
- EEG is segmented into epochs around stimulus events:
  - Time window: **-200 ms to +800 ms**
- Baseline correction is applied using the **pre-stimulus interval**.

---

##  ERP Visualization (P300 Component)
- Event-Related Potentials (ERPs) are computed for:
  - **Target stimuli**
  - **Non-target stimuli**
- Visualizations include:
  - Channel-wise ERP plots
  - Grand average ERP

 The **P300 peak (~300 ms)** is clearly highlighted for target stimuli.

---

##  Feature Extraction
Three feature extraction techniques are implemented:

### 1. Downsampled Waveform
- Epochs are flattened after temporal downsampling.

### 2. Xdawn Spatial Filtering
- Enhances ERP signals by maximizing signal-to-noise ratio.

### 3. Riemannian Covariance Features
- Uses covariance matrices
- Transformed into tangent space for classification

---

##  Classification Models
Models are trained and evaluated using **Stratified K-Fold Cross-Validation**:

###  Linear Discriminant Analysis (LDA)
- Baseline classifier commonly used in BCI

###  Support Vector Machine (SVM)
- Uses **RBF kernel**
- Handles non-linear patterns

###  EEGNet (PyTorch)
- Lightweight CNN for EEG signals
- Uses **class-weighted cross-entropy** to handle imbalance

---

##  Ensemble / Score Averaging
- Combines predictions from:
  - LDA  
  - SVM  
  - EEGNet  

- Uses **probability averaging**

 Particularly effective for:
- P300 spellers
- Multi-trial character selection

---

##  Summary
This pipeline demonstrates a complete workflow for EEG-based P300 classification:

- Robust preprocessing  
- Multiple feature extraction strategies  
- Classical + deep learning models  
- Ensemble learning for improved accuracy  

---


##  Results Summary

All models are evaluated on a **held-out test set**, using standard classification metrics:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Information Transfer Rate (ITR)  

---

###  Final Results

| Model          | Accuracy | Precision | Recall | F1-Score | ITR (bits/min) |
|---------------|----------|----------|--------|----------|----------------|
| LDA (Xdawn)   | 0.8966   | 0.6500   | 0.8667 | 0.7429   | 124.78         |
| SVM (RBF)     | 0.8621   | 1.0000   | 0.2000 | 0.3333   | 116.51         |
| EEGNet        | 0.9655   | 0.8333   | 1.0000 | 0.9091   | 143.30         |
| **Ensemble**  | **0.9770** | **0.9333** | **0.9333** | **0.9333** | **146.82** |

---

##  Key Observations

-  **EEGNet** achieves the highest accuracy and ITR  
  → Demonstrates the strength of deep learning in EEG-based BCI systems  

-  **Ensemble Model** performs exceptionally well  
  → Benefits from combining multiple classifiers  

-  **ITR (Information Transfer Rate)**  
  → Measures how fast a user can communicate using the BCI  
  → **ITR > 20 bits/min is considered competitive**  
  → All models significantly exceed this threshold  

---
