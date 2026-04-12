import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

from preprocess import get_clean_data, apply_bad_channel_interpolation, apply_spatial_ica, apply_average_reference
from features import apply_xdawn, extract_riemannian_covariances, extract_p300_features
from models import EEGNet, get_lda_pipeline, get_svm_pipeline, get_eegnet_pipeline, get_riemannian_pipeline

def get_character_prediction(probs, y_test, flash_ids, n_reps=10):
    unique_f = np.sort(np.unique(flash_ids))
    if len(unique_f) < 12: return 0 
    rows_ids = unique_f[:6]
    cols_ids = unique_f[6:]
    flash_per_char = 12 * n_reps
    n_chars = len(probs) // flash_per_char
    correct_chars = 0
    for i in range(n_chars):
        start = i * flash_per_char
        end = (i + 1) * flash_per_char
        char_probs = probs[start:end]
        char_labels = y_test[start:end]
        char_flashes = flash_ids[start:end]
        agg_probs = {}
        target_row = -1
        target_col = -1
        for p, l, f in zip(char_probs, char_labels, char_flashes):
            if f not in agg_probs: agg_probs[f] = []
            agg_probs[f].append(p)
            if l == 1:
                if f in rows_ids: target_row = f
                if f in cols_ids: target_col = f
        mean_probs = {f: np.mean(v) for f, v in agg_probs.items()}
        pred_row = rows_ids[np.argmax([mean_probs.get(r, 0) for r in rows_ids])]
        pred_col = cols_ids[np.argmax([mean_probs.get(c, 0) for c in cols_ids])]
        if pred_row == target_row and pred_col == target_col:
            correct_chars += 1
    return correct_chars / n_chars if n_chars > 0 else 0

def get_symbol_itr(n, acc, dur=2.1):
    if acc <= 1/n: return 0
    if acc >= 0.999: acc = 0.999
    bits = np.log2(n) + acc*np.log2(acc + 1e-10) + (1-acc)*np.log2((1-acc)/(n-1) + 1e-10)
    return bits * (60.0 / dur)

def run_benchmarking():
    datasets = ["BNCI2014_009", "EPFLP300"]
    os.makedirs('results', exist_ok=True)
    all_summary = []
    
    # Testing subjects 1 to 3 to meet subject-level accuracy requirements
    test_range = range(1, 4) 
    
    for ds_name in datasets:
        for subj in test_range: 
            print(f"\\n--- [ {ds_name} ] Subject {subj} ---")
            
            try:
                epochs, X, y = get_clean_data(ds_name, subj)
            except Exception as e:
                print(f"  Skipping subject {subj} due to loading error: {e}")
                continue
                
            skf = GroupKFold(n_splits=5)
            groups = epochs.metadata['char_id'].values
            
            models_list = [
                ("LDA", get_lda_pipeline()),
                ("SVM", get_svm_pipeline()),
                ("Xdawn+LDA", get_lda_pipeline()), 
                ("Riemannian MDM", get_riemannian_pipeline()),
                ("EEGNet", get_eegnet_pipeline())
            ]
            
            for name, clf in models_list:
                print(f"  Training {name}...")
                metrics = []
                subject_probs = []
                subject_y = []
                subject_flashes = []
                
                for train_idx, test_idx in skf.split(X, y, groups=groups):
                    e_tr, e_te = epochs[train_idx].copy(), epochs[test_idx].copy()
                    y_tr, y_te = y[train_idx], y[test_idx]
                    
                    e_tr, e_te = apply_bad_channel_interpolation(e_tr, e_te)
                    e_tr, e_te = apply_average_reference(e_tr, e_te)
                    e_tr, e_te = apply_spatial_ica(e_tr, e_te)
                    
                    if "Xdawn" in name:
                        X_tr, X_te = apply_xdawn(e_tr, y_tr, e_te)
                    elif "Riemannian" in name:
                        X_tr = extract_riemannian_covariances(e_tr.get_data())
                        X_te = extract_riemannian_covariances(e_te.get_data())
                    elif "EEGNet" in name:
                        X_tr_data = e_tr.get_data()
                        X_te_data = e_te.get_data()
                        mu = np.mean(X_tr_data)
                        sd = np.std(X_tr_data)
                        X_tr = ((X_tr_data - mu) / sd)[:, np.newaxis, :, :].astype(np.float32)
                        X_te = ((X_te_data - mu) / sd)[:, np.newaxis, :, :].astype(np.float32)
                        y_tr = y_tr.astype(np.int64)
                        y_te = y_te.astype(np.int64)
                    else:
                        X_tr = extract_p300_features(e_tr.get_data())
                        X_te = extract_p300_features(e_te.get_data())
                        
                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_te)
                    
                    subject_probs.extend(clf.predict_proba(X_te)[:, 1])
                    subject_y.extend(y_te)
                    subject_flashes.extend(e_te.metadata['flash_id'].values)
                    
                    metrics.append([
                        accuracy_score(y_te, y_pred),
                        recall_score(y_te, y_pred),
                        precision_score(y_te, y_pred, zero_division=0),
                        f1_score(y_te, y_pred, zero_division=0)
                    ])
                    
                avg_m = np.mean(metrics, axis=0)
                char_acc = get_character_prediction(np.array(subject_probs), np.array(subject_y), np.array(subject_flashes))
                itr = get_symbol_itr(36, char_acc, dur=21.0)
                y_pred_all = (np.array(subject_probs) > 0.5).astype(int)
                cm = confusion_matrix(subject_y, y_pred_all)
                
                plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f"CM: {name} ({ds_name} S{subj})")
                plt.xlabel("Predicted"); plt.ylabel("True")
                plt.savefig(f"results/cm_{ds_name}_S{subj}_{name}.png")
                plt.close()
                print(f"    - F1: {avg_m[3]:.3f} | Char Acc: {char_acc*100:.1f}% | ITR: {itr:.2f} bpm")
                
                all_summary.append([ds_name, subj, name, avg_m[0], avg_m[3], char_acc, itr])
                
    if all_summary:
        df = pd.DataFrame(all_summary, columns=['Dataset', 'Subject', 'Model', 'Acc', 'F1', 'Char_Acc', 'ITR_N36'])
        df.to_csv('results/all_subject_results.csv', index=False)
        print("\\n[DONE] Benchmark Complete. Results and Confusion Matrices saved to results/ folder.")

if __name__ == "__main__":
    run_benchmarking()
