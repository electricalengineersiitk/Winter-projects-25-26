"""
evaluate.py — Metrics, ITR, confusion matrix, character decoding
"""
import os, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)

os.makedirs('results', exist_ok=True)
# P300 character grid 
CHAR_MATRIX = np.array([
    ['A','B','C','D','E','F'],
    ['G','H','I','J','K','L'],
    ['M','N','O','P','Q','R'],
    ['S','T','U','V','W','X'],
    ['Y','Z','1','2','3','4'],
    ['5','6','7','8','9','_']
])

def information_transfer_rate(N: int, P: float, T: float) -> float:
    """
    ITR in bits/minute.
    N = number of symbols (36 for 6×6 grid)
    P = classification accuracy  (0 – 1)
    T = trial duration in seconds
    """
    if P <= 0 or P >= 1:
        return 0.0
    B = (math.log2(N)
         + P * math.log2(P)
         + (1 - P) * math.log2((1 - P) / (N - 1)))
    return B * (60.0 / T)

# Classification report 
def evaluate_classifier(model, X, y_true,
                         model_name: str = 'Model',
                         save_plot:  bool = True) -> dict:
    """
    Predict, print full report, save confusion matrix.
    Returns dict with accuracy, auc, itr.
    """
    y_pred = model.predict(X)
    acc    = accuracy_score(y_true, y_pred)

    try:
        probs = model.predict_proba(X)[:, 1]
        auc   = roc_auc_score(y_true, probs)
    except Exception:
        auc = float('nan')

    # Trial time: 10 repetitions × 12 flashes × ~125 ms each
    T_trial = 10 * 12 * 0.125
    itr_val = information_transfer_rate(N=36, P=acc, T=T_trial)

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  AUC      : {auc:.4f}")
    print(f"  ITR      : {itr_val:.1f} bits/min")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred,
                                 target_names=['NonTarget','Target']))

    if save_plot:
        _plot_confusion(y_true, y_pred, model_name)

    return {'accuracy': acc, 'auc': auc, 'itr': itr_val}

def _plot_confusion(y_true, y_pred, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NonTarget', 'Target'],
                yticklabels=['NonTarget', 'Target'])
    plt.title(f'{model_name} — Confusion Matrix')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    fname = f"results/{model_name.replace(' ', '_')}_confusion.png"
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"  → Saved: {fname}")

# ERP visualisation 
def plot_erp(X: np.ndarray, y: np.ndarray,
             fs: int = 256, channel_idx: int = 10,
             subject_id: int = 1):
    """
    Plot average ERP for Target vs NonTarget at one channel.
    """
    tmin, tmax = -0.1, 0.8
    times = np.linspace(tmin, tmax, X.shape[2])
    target     = X[y == 1, channel_idx, :].mean(axis=0)
    non_target = X[y == 0, channel_idx, :].mean(axis=0)

    plt.figure(figsize=(9, 4))
    plt.plot(times * 1000, target,     label='Target',     linewidth=2)
    plt.plot(times * 1000, non_target, label='Non-Target', linewidth=2,
             linestyle='--')
    plt.axvline(0,   color='k', linestyle=':', linewidth=1)
    plt.axvline(300, color='gray', linestyle=':', linewidth=1, label='~P300')
    plt.xlabel('Time (ms)'); plt.ylabel('Amplitude (µV)')
    plt.title(f'ERP — Subject {subject_id} — Channel {channel_idx}')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    fname = f'results/erp_subject_{subject_id:02d}.png'
    plt.savefig(fname, dpi=100); plt.close()
    print(f"  → ERP saved: {fname}")

# Cross-subject summary table 
def print_summary_table(results: dict):
    """
    results: {subject_id: {'lda': {...}, 'svm': {...}}}
    """
    print(f"\n{'Subject':>8} {'LDA Acc':>10} {'SVM Acc':>10} "
          f"{'LDA ITR':>10} {'SVM ITR':>10}")
    print('-' * 55)
    for sid, r in results.items():
        print(f"{sid:>8} "
              f"{r['lda']['accuracy']*100:>9.1f}% "
              f"{r['svm']['accuracy']*100:>9.1f}% "
              f"{r['lda']['itr']:>9.1f}  "
              f"{r['svm']['itr']:>9.1f}")