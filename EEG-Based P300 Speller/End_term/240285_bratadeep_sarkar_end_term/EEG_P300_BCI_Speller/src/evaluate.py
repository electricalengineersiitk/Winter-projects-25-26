import os
import sys
from pathlib import Path

# Add project root to sys.path for robust running from any location
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import config
from preprocess import get_clean_data
from engine import run_model_evaluation
from models import get_lda_pipeline, get_svm_pipeline, get_eegnet_pipeline, get_riemannian_pipeline
from utils import setup_environment, get_symbol_itr, get_character_prediction

def run_benchmarking():
    """
    Main entry point for full BCI pipeline benchmarking.
    Runs all models across all subjects and datasets.
    """
    all_summary = []

    for ds_name in config.DATASETS:
        for subj in config.TEST_SUBJECTS:
            print(f"\n--- [ {ds_name} ] Subject {subj} ---")

            try:
                # Load CLEAN data (ICA/Filtering already done in raw stage)
                epochs, X, y = get_clean_data(ds_name, subj)
            except Exception as e:
                print(f"  Skipping subject {subj}: {e}")
                continue

            n_chans, n_times = X.shape[1], X.shape[2]

            models_list = [
                ("LDA",            get_lda_pipeline()),
                ("SVM",            get_svm_pipeline()),
                ("Xdawn+LDA",      get_lda_pipeline()),
                ("Riemannian MDM", get_riemannian_pipeline()),
                ("EEGNet",         get_eegnet_pipeline(in_chans=n_chans, input_window_samples=n_times)),
            ]

            for name, clf in models_list:
                print(f"  Evaluating {name}...")
                
                # CALL CENTRALIZED ENGINE
                results = run_model_evaluation(epochs, X, y, clf, name)
                
                avg_m = results['metrics']
                char_acc = get_character_prediction(results['probs'], results['true_y'], results['flash_ids'])
                itr = get_symbol_itr(36, char_acc, dur=config.TRIAL_DURATION)

                # Save metrics for summary
                all_summary.append([
                    ds_name, subj, name,
                    avg_m[0], avg_m[3], avg_m[2], avg_m[1],
                    char_acc, itr
                ])

                # Confusion matrix generation
                y_pred_all = (results['probs'] > 0.5).astype(int)
                cm = confusion_matrix(results['true_y'], y_pred_all)
                plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f"CM: {name} ({ds_name} S{subj})")
                plt.xlabel("Predicted"); plt.ylabel("True")
                plt.savefig(config.RESULTS_DIR / f"cm_{ds_name}_S{subj}_{name}.png")
                plt.close()

                print(f"    Acc: {avg_m[0]:.3f} | F1: {avg_m[3]:.3f} | ITR: {itr:.2f} bpm")

    # Save CSV results
    if all_summary:
        df = pd.DataFrame(
            all_summary,
            columns=['Dataset', 'Subject', 'Model', 'Acc', 'F1', 'Precision', 'Recall', 'Char_Acc', 'ITR_N36']
        )
        df.to_csv(config.RESULTS_DIR / 'all_subject_results.csv', index=False)
        print(f"\n[DONE] Benchmark complete. Results saved to {config.RESULTS_DIR}")

        # Post-run: Comparative ERP Plots
        try:
            from visualization import plot_dataset_erp
            print("\n[PLOTS] Generating ERP Comparison...")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            plot_dataset_erp(ax1, 'BNCI2014_009', subj=1)
            plot_dataset_erp(ax2, 'EPFLP300', subj=1)
            plt.suptitle('P300 Grand Average Comparison', fontsize=16, y=1.02)
            plt.tight_layout()
            plt.savefig(config.RESULTS_DIR / 'grand_average_comparison.png', bbox_inches='tight')
            plt.close()
        except:
            pass

if __name__ == "__main__":
    setup_environment()
    run_benchmarking()
