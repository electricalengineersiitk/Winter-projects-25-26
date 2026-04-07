import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from mne.preprocessing import Xdawn

# Local Imports (Zero-Leakage & Modular)
from preprocess import get_clean_data, apply_bad_channel_interpolation, apply_spatial_ica
from models import EEGNet, get_lda_pipeline, get_svm_pipeline

def get_character_prediction(probs, y_test, flash_per_char=12):
    """
    Groups flash predictions into symbol-level decisions (N=36).
    Standard P300 Ensemble Logic: Max prob in Cycle = Predicted Target.
    """
    n_chars = len(probs) // flash_per_char
    correct_chars = 0
    for i in range(n_chars):
        # Extract 12 flashes for this character
        c_probs = probs[i*flash_per_char : (i+1)*flash_per_char]
        c_labels = y_test[i*flash_per_char : (i+1)*flash_per_char]
        
        # Predicted: Flash index with max target probability
        pred_idx = np.argmax(c_probs)
        # Truth: Flash index with label 1 (Target)
        true_idx = np.where(c_labels == 1)[0]
        
        if len(true_idx) > 0 and pred_idx == true_idx[0]:
            correct_chars += 1
            
    return correct_chars / n_chars if n_chars > 0 else 0

def get_symbol_itr(n, acc, dur=12.0):
    """Physically valid ITR (N=36, T=1character_time) in bits/min."""
    if acc <= 1/n: return 0
    if acc >= 0.999: acc = 0.999
    bits = np.log2(n) + acc*np.log2(acc) + (1-acc)*np.log2((1-acc)/(n-1))
    return bits * (60.0 / dur)

def run_benchmarking():
    datasets = ["BNCI2014_009"] # , "EPFLP300"]
    os.makedirs('results', exist_ok=True)
    all_summary = []

    for ds_name in datasets:
        for subj in range(1, 2): # Subject 1 as benchmark
            print(f"\n--- [ {ds_name} ] Subject {subj} ---")
            epochs, X, y = get_clean_data(ds_name, subj)
            skf = StratifiedKFold(n_splits=5, shuffle=False)
            
            # --- Classical Models Comparison ---
            models_list = [
                ("LDA", get_lda_pipeline()),
                ("SVM", get_svm_pipeline()),
                ("Xdawn+LDA", get_lda_pipeline()) # Xdawn features handled in CV
            ]

            for name, clf in models_list:
                print(f"  Training {name}...")
                metrics = []
                subject_probs = []
                subject_y = []

                for train_idx, test_idx in skf.split(X, y):
                    # Data Slicing
                    e_tr, e_te = epochs[train_idx].copy(), epochs[test_idx].copy()
                    y_tr, y_te = y[train_idx], y[test_idx]

                    # 1. Zero-Leakage Preprocessing (Per-fold)
                    e_tr, e_te = apply_bad_channel_interpolation(e_tr, e_te)
                    e_tr, e_te = apply_spatial_ica(e_tr, e_te)
                    
                    # 2. Model Specific Features
                    if "Xdawn" in name:
                        try:
                            # Scientific Fix: correct_overlap=False and reduced components for stability
                            xd = Xdawn(n_components=2, correct_overlap=False).fit(e_tr, y_tr)
                            X_tr = xd.transform(e_tr).reshape(len(e_tr), -1)
                            X_te = xd.transform(e_te).reshape(len(e_te), -1)
                        except Exception as e:
                            print(f"      [Warning] Xdawn failed ({e}), falling back to waveform features.")
                            X_tr = e_tr.get_data().reshape(len(e_tr), -1)
                            X_te = e_te.get_data().reshape(len(e_te), -1)
                    else:
                        X_tr = e_tr.get_data().reshape(len(e_tr), -1)
                        X_te = e_te.get_data().reshape(len(e_te), -1)

                    # 3. Classify
                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_te)
                    # For character ensemble logic
                    subject_probs.extend(clf.predict_proba(X_te)[:, 1])
                    subject_y.extend(y_te)
                    
                    metrics.append([
                        accuracy_score(y_te, y_pred),
                        recall_score(y_te, y_pred),
                        precision_score(y_te, y_pred, zero_division=0),
                        f1_score(y_te, y_pred, zero_division=0)
                    ])

                # Aggregated Analysis
                avg_m = np.mean(metrics, axis=0)
                
                # 4. Correct ITR (N=36) via Character-Level Aggregation
                char_acc = get_character_prediction(np.array(subject_probs), np.array(subject_y))
                itr = get_symbol_itr(36, char_acc, dur=12.0)
                
                # 5. Confusion Matrix (Requirement: Visual CM)
                y_pred_all = (np.array(subject_probs) > 0.5).astype(int)
                cm = confusion_matrix(subject_y, y_pred_all)
                plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f"CM: {name} ({ds_name})")
                plt.xlabel("Predicted"); plt.ylabel("True")
                plt.savefig(f"results/cm_{ds_name}_{name}.png")
                plt.close()

                print(f"    - F1: {avg_m[3]:.3f} | Char Acc: {char_acc*100:.1f}% | ITR: {itr:.2f} bpm")
                all_summary.append([ds_name, subj, name, avg_m[0], avg_m[3], char_acc, itr])

    # Final Report Save
    if all_summary:
        df = pd.DataFrame(all_summary, columns=['Dataset', 'Subject', 'Model', 'Acc', 'F1', 'Char_Acc', 'ITR_N36'])
        df.to_csv('results/all_subject_results.csv', index=False)
        print("\n[DONE] Benchmark Complete. Results and Confusion Matrices saved to results/ folder.")

if __name__ == "__main__":
    run_benchmarking()
