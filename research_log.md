# EEG Brain Speller — Research & Development Log

## 📅 Log Entry: 2026-03-24
- **Baseline:** Created `04_classify.py`. Used simple **Linear Discriminant Analysis (LDA)** as the baseline model.
- **Accuracy:** Observed initial single-trial accuracy around 0.70.
- **Problem:** Significant P300 class imbalance (1:5 Target vs Non-Target ratio).
- **Decision:** Shifted to **Class-Weighted SVM** and **EEGNet** to handle the minority class more effectively.

## 📅 Log Entry: 2026-03-26
### Phase: Signal Processing & The "Xdawn Fallback"
- **Implementation:** Built a pipeline with 0.1-30Hz Bandpass, 50Hz Notch, and Average Re-referencing.
- **Pivot (The Fallback):** Originally intended to use **Xdawn Spatial Filtering** for feature extraction as recommended in the docs.
- **Insight:** Running Xdawn on average-referenced data caused a `LinAlgError` (Rank Deficiency).
- **Resolution:** Fell back to **Downsampling (Decimation)**. This proved to be more mathematically robust while still significantly reducing data dimensionality (87.5% reduction).

## 📅 Log Entry: 2026-04-06
### Phase: Final A+ Polish & Master Sync
- [x] **Final Quality Audit:** Full review against `output.txt` rubric.
- [x] **Master Sync:** Synchronized `05_eegnet.py` and `06_final_evaluation.py` with identical preprocessing.
- [x] **Bad Channel Interpolation:** Implemented variance-based bad channel identification to satisfy Stage 1 requirements.
- [x] **ICA Artifact Rejection:** Applied ICA to remove eye blinks (~2 components excluded).
- [x] **Evaluation Upgrade:** Moved EEGNet to 5-Fold Stratified Cross-Validation for scientific rigor.
- [x] **Visuals:** Generated `results/erp_waveform.png` to visually confirm P300 component detection.
- [x] **Confusion Matrix:** Aggregated EEGNet predictions across 5-folds to plot a Seaborn heatmap (`results/confusion_matrix.png`), fully matching Stage 5 requirements.
- [x] **ITR Validation:** Confirmed Information Transfer Rate (ITR) calculations across all models.
- [x] **Ensemble Simulation:** Demonstrated Accuracy boost via score averaging in `08_ensemble_averaging.py`.

---
**Status:** Project is 100% compliant with mentor's final stage requirements. Ready for submission.
