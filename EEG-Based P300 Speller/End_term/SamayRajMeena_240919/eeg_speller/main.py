"""
main.py — Full EEG Brain Speller Pipeline
Dataset : BNCI2014_009 (P300 speller, 10 subjects, MOABB)
"""
import os, sys, argparse
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')

os.makedirs('results', exist_ok=True)
os.makedirs('models',  exist_ok=True)

from src.preprocess import download_dataset, load_subject, apply_notch_to_epochs, apply_ica_to_epochs
from src.features   import extract_features
from src.models     import build_lda, build_svm
from src.evaluate   import (evaluate_classifier, plot_erp,
                             print_summary_table, information_transfer_rate)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

# CLI 
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=int, nargs='+',
                    default=list(range(1, 11)),
                    help='Subject IDs to process (1-10)')
parser.add_argument('--method', type=str, default='downsample',
                    choices=['downsample', 'pca', 'xdawn'],
                    help='Feature extraction method')
parser.add_argument('--skip-download', action='store_true')
args = parser.parse_args()
if not args.skip_download:
    download_dataset()

# Main loop 
all_results = {}
for sid in args.subjects:
    print(f"\n{'#'*60}")
    print(f"  SUBJECT {sid:02d}")
    print(f"{'#'*60}")

    # 1.Load epochs 
    X, y, meta = load_subject(sid)
    X = apply_notch_to_epochs(X)   # extra 50 Hz notch
    X = apply_ica_to_epochs(X)

    # 2.ERP plot 
    plot_erp(X, y, subject_id=sid)

    # 3.Train / test split (80 / 20) 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4.Feature extraction 
    print(f"\nFeature extraction method: {args.method}")
    X_tr, X_te, scaler, transformer = extract_features(
        X_train, y_train, X_test, method=args.method
    )

    # 5.Cross-validation on training set
    print("\n── 5-Fold Cross-Validation (training set) ──")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, clf in [('LDA', build_lda()), ('SVM', build_svm())]:
        scores = cross_val_score(clf, X_tr, y_train,
                                 cv=cv, scoring='accuracy', n_jobs=-1)
        print(f"  {name}: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

    #  6.Train final models 
    lda = build_lda(); lda.fit(X_tr, y_train)
    svm = build_svm(); svm.fit(X_tr, y_train)

    # 7.Test-set evaluation 
    print("\n── Test-set evaluation ──")
    r_lda = evaluate_classifier(lda, X_te, y_test, f'LDA S{sid:02d}')
    r_svm = evaluate_classifier(svm, X_te, y_test, f'SVM S{sid:02d}')
    all_results[sid] = {'lda': r_lda, 'svm': r_svm}

    # 8.Save models
    joblib.dump({'model': lda, 'scaler': scaler},
                f'models/lda_subject_{sid:02d}.pkl')
    joblib.dump({'model': svm, 'scaler': scaler},
                f'models/svm_subject_{sid:02d}.pkl')
    print(f"  Models saved → models/lda_subject_{sid:02d}.pkl")
    print(f"                 models/svm_subject_{sid:02d}.pkl")

# 9. Summary table
print(f"\n{'='*60}")
print("  FINAL SUMMARY ACROSS ALL SUBJECTS")
print(f"{'='*60}")
print_summary_table(all_results)
print("\n Pipeline complete! Results saved to results/ and models/\n")