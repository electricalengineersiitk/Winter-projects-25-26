# **EEG-Based P300 Brain Speller System**
### **Brain-Computer Interface (BCI) for Text Communication**

This repository contains a complete end-to-end pipeline for an EEG-based Brain-Computer Interface. The system utilizes the **P300 Event-Related Potential (ERP)** to allow users to "type" characters by simply focusing on them as they flash on a 6×6 grid.

---

## **1. Project Overview**
The core of this project is the detection of the **"Oddball" effect**. When a user focuses on a specific "Target" character among "Non-Target" characters, their brain produces a distinct positive voltage deflection roughly 300ms after the stimulus—the **P300**. 

By processing noisy EEG signals, extracting spatial features, and applying machine learning, this system identifies the intended character with high precision.

### **Key Technical Achievements:**
* **Peak Accuracy:** 92.4% (SVM-RBF).
* **Spatial Filtering:** Implemented **Xdawn** to isolate P300 components from background noise.
* **Comparative Analysis:** Evaluated LDA, SVM, and Deep Learning (EEGNet) models.
* **Metric Tracking:** Calculated Information Transfer Rate (ITR) to measure communication speed.

---

## **2. System Architecture**
The project is divided into four main stages:

1.  **Preprocessing:** 0.1–30Hz bandpass filtering, 50Hz notch filtering, and ICA-based artifact removal.
2.  **Epoching:** Time-locking signal windows (-200ms to 800ms) relative to each flash.
3.  **Feature Extraction:** Downsampling and Xdawn spatial covariance mapping.
4.  **Classification:** Ensemble scoring across flash repetitions to determine the final character.

---

## **3. Environment Setup**

### **Prerequisites**
* **Python:** 3.9 or 3.10 (Recommended)
* **Virtual Environment:** Highly recommended to avoid dependency conflicts.

### **Installation**
1. **Clone/Download** the repository to your local machine.
2. **Create a virtual environment:**
   ```cmd
   python -m venv eeg_env
   eeg_env\Scripts\activate  # Windows
