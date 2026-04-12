from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def char_accuracy(pred, true):
    n = min(len(pred), len(true))
    correct = sum(p == t for p, t in zip(pred[:n], true[:n]))
    acc = correct / n
    print(f"Accuracy: {acc*100:.2f}% ({correct}/{n})")
    return acc

def char_confusion(pred, true, save_path=None):
    n = min(len(pred), len(true))
    pred, true = list(pred[:n]), list(true[:n])
    labels = sorted(set(true))
    cm = confusion_matrix(true, pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Character Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()

def classification_metrics(model, X, y):
    y_pred = model.predict(X)
    p, r, f1 = precision_score(y, y_pred), recall_score(y, y_pred), f1_score(y, y_pred)
    print(f"P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
    return p, r, f1

def stratified_kfold_cv(model, X, y, k=5):
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    res = cross_validate(model, X, y, cv=cv,
                         scoring=['accuracy', 'precision', 'recall', 'f1'])
    print(f"{k}-Fold | Acc={res['test_accuracy'].mean():.4f} "
          f"P={res['test_precision'].mean():.4f} "
          f"R={res['test_recall'].mean():.4f} "
          f"F1={res['test_f1'].mean():.4f}")
    return res

def plot_confusion_matrix(model, X, y, save_path=None):
    cm = confusion_matrix(y, model.predict(X))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-P300', 'P300'],
                yticklabels=['Non-P300', 'P300'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Epoch Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()

def calculate_itr(acc, n_reps=15, N=36, flash_duration=0.175):
    T = n_reps * 12 * flash_duration
    if acc <= 0 or acc >= 1:
        itr = 0.0
    else:
        itr = (np.log2(N) + acc * np.log2(acc)
               + (1 - acc) * np.log2((1 - acc) / (N - 1))) * (60 / T)
    print(f"ITR: {itr:.2f} bits/min")
    return itr

def plot_accuracy_vs_reps(subject, reps_list, acc_list, save_path=None):
    plt.figure(figsize=(6, 4))
    plt.plot(reps_list, acc_list, marker='o')
    plt.xlabel("Number of Repetitions")
    plt.ylabel("Accuracy")
    plt.title(f"Subject {subject} — Accuracy vs Repetitions")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()

def plot_itr_vs_reps(subject, reps_list, itr_list, save_path=None):
    plt.figure(figsize=(6, 4))
    plt.plot(reps_list, itr_list, marker='s', color='orange')
    plt.xlabel("Number of Repetitions")
    plt.ylabel("ITR (bits/min)")
    plt.title(f"Subject {subject} — ITR vs Repetitions")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()