from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
from sklearn.model_selection import StratifiedKFold,cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def char_accuracy(pred,true):
    n=min(len(pred),len(true))
    correct=sum(p==t for p,t in zip(pred[:n],true[:n]))
    acc=correct/n
    print(f"Character Accuracy: {acc*100:.2f}% ({correct}/{n})")
    return acc

def print_comparison(pred,true,name=""):
    print(f"\n--- {name} ---")
    print("Predicted :",pred)
    print("True      :",true)
    acc=char_accuracy(pred,true)
    return acc

def calculate_itr(P,N=36,n_reps=15,flash_duration=0.175):
    T=n_reps*12*flash_duration
    if P<=0 or P>=1:
        itr=0.0
    else:
        itr=(np.log2(N)+P*np.log2(P)+(1-P)*np.log2((1-P)/(N-1)))*(60/T)
    print(f"ITR       : {itr:.2f} bits/minute  (N={N}, T={T:.1f}s, reps={n_reps})")
    return itr

def classification_metrics(model,X,y):
    y_pred=model.predict(X)
    p=precision_score(y,y_pred)
    r=recall_score(y,y_pred)
    f1=f1_score(y,y_pred)
    print(f"Precision : {p:.4f}")
    print(f"Recall    : {r:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    return p,r,f1

def stratified_kfold_cv(model,X,y,k=5):
    cv=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)
    results=cross_validate(model,X,y,cv=cv,scoring=['accuracy','precision','recall','f1'])
    print(f"\n--- {k}-Fold Stratified Cross-Validation ---")
    print(f"Accuracy  : {results['test_accuracy'].mean():.4f} ± {results['test_accuracy'].std():.4f}")
    print(f"Precision : {results['test_precision'].mean():.4f} ± {results['test_precision'].std():.4f}")
    print(f"Recall    : {results['test_recall'].mean():.4f} ± {results['test_recall'].std():.4f}")
    print(f"F1 Score  : {results['test_f1'].mean():.4f} ± {results['test_f1'].std():.4f}")
    return results

def plot_confusion_matrix(model,X,y,subject="",mode=""):
    y_pred=model.predict(X)
    cm=confusion_matrix(y,y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
                xticklabels=['Non-P300','P300'],
                yticklabels=['Non-P300','P300'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — Subject {subject} {mode.upper()}")
    plt.tight_layout()
    filename=f"results/confusion_matrix_Subject{subject}_{mode.upper()}.png"
    plt.savefig(filename)
    plt.show()
    print(f"Confusion matrix saved to {filename}")