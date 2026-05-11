import sys
from pathlib import Path

# Path hack for root run
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import config
from preprocess import get_clean_data
from engine import run_model_evaluation
from models import get_svm_pipeline
from utils import setup_environment, get_symbol_itr, get_character_prediction

def run_ensemble_benchmark():
    """
    Standalone script for the Ensemble Speller benchmark.
    Uses unified engine to ensure zero leakage and consistent preprocessing.
    """
    print("\n=== P300 Ensemble Speller Benchmark (SVM) ===")
    results_list = []
    
    # Run specifically for the Standard Ensemble model (SVM with RBF)
    clf = get_svm_pipeline()
    
    for ds_name in config.DATASETS:
        for subj in config.TEST_SUBJECTS:
            print(f"  Crunching ensemble for {ds_name} Subject {subj}...")
            
            try:
                epochs, X, y = get_clean_data(ds_name, subj)
            except: 
                continue
                
            # EXECUTE ENGINE
            results = run_model_evaluation(epochs, X, y, clf, "SVM_Ensemble")
            
            char_acc = get_character_prediction(results['probs'], results['true_y'], results['flash_ids'])
            itr = get_symbol_itr(36, char_acc, dur=config.TRIAL_DURATION)
            
            results_list.append({
                'dataset': ds_name,
                'subject': subj,
                'char_acc': char_acc,
                'itr': itr
            })

    if results_list:
        df = pd.DataFrame(results_list)
        df.to_csv(config.RESULTS_DIR / 'ensemble_results.csv', index=False)
        print(f"\n[DONE] Ensemble results saved to {config.RESULTS_DIR / 'ensemble_results.csv'}")

if __name__ == "__main__":
    setup_environment()
    run_ensemble_benchmark()
